"""
Drift Policy 

- Using 1-step drift model (no multi-step flow)
- No distillation loss
- Drift loss with Q-learning
"""

import copy
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value
from utils.drift_loss import drift_loss


class DriftAgent(flax.struct.PyTreeNode):
    """Drift Policy agent with drift model."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _get_bon_samples(self, use_q_bon: bool):
        key = "q_bon" if use_q_bon else "eval_bon"
        n = self.config[key]
        if n <= 0:
            n = self.config["actor_num_samples"]
        return n

    def critic_loss(self, batch, grad_params, rng):
        """Compute the Drift critic loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]

        rng, sample_rng = jax.random.split(rng)
        next_obs = batch["next_observations"][..., -1, :]
        next_actions = self.sample_actions(next_obs, sample_rng, use_q_bon=True)

        next_qs = self.network.select("target_critic")(next_obs, actions=next_actions)
        if self.config["q_agg"] == "min":
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch["rewards"][..., -1] + (
            self.config["discount"] ** self.config["horizon_length"]
        ) * batch["masks"][..., -1] * next_q

        q = self.network.select("critic")(
            batch["observations"], actions=batch_actions, params=grad_params
        )
        valid = batch["valid"][..., -1] if "valid" in batch else jnp.ones_like(target_q)
        critic_loss = (jnp.square(q - target_q) * valid).mean()

        info = {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "tgt_q_mean": next_q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }
        return critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute actor loss with drift model."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        batch_size, action_dim = batch_actions.shape

        drift_mode = self.config.get("drift_mode", "none")
        if drift_mode in ("hard", "soft"):
            return self._actor_loss_online(batch, batch_actions, grad_params, rng, drift_mode)

        _, drift_rng = jax.random.split(rng)

        # Generate multiple samples per observation.
        gen_per_label = self.config.get("gen_per_label", 8)
        obs_repeated = jnp.repeat(batch["observations"], gen_per_label, axis=0)
        drift_noises = jax.random.normal(drift_rng, (batch_size * gen_per_label, action_dim))

        # Get actions from drift model
        drift_actions_all = self.network.select("actor_drift")(
            obs_repeated, drift_noises, params=grad_params
        )
        drift_actions_all = jnp.clip(drift_actions_all, -1, 1)
        # Reshape to [B, gen_per_label, action_dim]
        gen_samples = drift_actions_all.reshape(batch_size, gen_per_label, action_dim)

        # Reuse the same generated actions for Q loss over all generated samples.
        qs_all = self.network.select("critic")(
            obs_repeated, actions=drift_actions_all
        )
        q_all = getattr(jnp, self.config["q_agg"])(qs_all, axis=0).reshape(
            batch_size, gen_per_label
        )

        q_loss = -q_all.mean()
        if self.config["normalize_q_loss"]:
            lam = jax.lax.stop_gradient(1 / (jnp.abs(q_all).mean()))
            q_loss = lam * q_loss
        q_loss = self.config["q_alpha"] * q_loss

        # Positive samples: dataset actions [B, 1, action_dim]
        pos_samples = jnp.expand_dims(batch_actions, axis=1)

        # Compute drift loss (per sample in batch)
        # Note: drift_loss already normalizes internally by scale_inputs
        drift_loss_val, drift_info = drift_loss(
            gen=gen_samples,
            fixed_pos=pos_samples,
            R_list=tuple(self.config.get("drift_temps", [0.1])),
            plus_only=bool(self.config.get("drift_plus_only", False)),
        )

        # Apply alpha to match Q loss magnitude.
        alpha = self.config.get("alpha", 1.0)
        actor_drift_loss = alpha * drift_loss_val.mean()

        # Total loss (no distillation)
        actor_loss = actor_drift_loss + q_loss

        info = dict(
            actor_loss=actor_loss,
            actor_drift_loss=actor_drift_loss,
            drift_scale=drift_info.get("scale", 0.0),
            q_loss=q_loss,
        )
        # Add per-temperature losses
        for key, val in drift_info.items():
            if key.startswith("loss_"):
                info[f"drift_{key}"] = val
        return actor_loss, info

    def _actor_loss_online(self, batch, batch_actions, grad_params, rng, mode):
        """Compute online actor loss with Q-based positive selection + paper-style neg.

        Phase 1: generate N samples, score with Q, pick top-a as fixed positives
                 (all stop-gradient; N-pool is used only for ranking).
        Phase 2: generate b fresh samples from new noise as gen; Q loss is computed
                 on these b samples. Negatives are handled implicitly by drift_loss
                 via its internal old_gen (paper Algorithm 1: y_neg = x with
                 self-exclusion on the diagonal).
          - hard: uniform weights on positives
          - soft: Q-value-based weights on positives

        If `bc_observations` / `bc_actions` are present in `batch`, the entire
        actor loss (Phase 1 / Phase 2 / BC drift target) is computed on those
        offline-only (obs, action) pairs; the Phase 2 `gen_samples` are reused
        for the BC drift target (no extra actor forward pass). Otherwise the
        actor loss falls back to using the main (mixed) batch's observations
        and actions throughout.
        """
        if "bc_observations" in batch and "bc_actions" in batch:
            obs = batch["bc_observations"]
            bc_actions_seq = batch["bc_actions"]
            if self.config["action_chunking"]:
                actor_actions = jnp.reshape(
                    bc_actions_seq, (bc_actions_seq.shape[0], -1)
                )
            else:
                actor_actions = bc_actions_seq[..., 0, :]
        else:
            obs = batch["observations"]
            actor_actions = batch_actions

        batch_size, action_dim = actor_actions.shape
        rng, sel_rng, drift_rng = jax.random.split(rng, 3)

        N = self.config.get("online_n_samples", 32)
        a = self.config.get("online_pos_samples", 4)
        b = self.config.get("gen_per_label", 8)
        neg_b = self.config.get("online_neg_samples", 4)

        # --- Phase 1: N-sample pool for positive selection (stop-gradient) ---
        obs_rep_N = jnp.repeat(obs, N, axis=0)
        noises_sel = jax.random.normal(sel_rng, (batch_size * N, action_dim))
        sel_actor_name = "actor_drift_ema" if self.config.get("use_actor_ema", False) else "actor_drift"
        sel_acts = self.network.select(sel_actor_name)(obs_rep_N, noises_sel)
        sel_acts = jnp.clip(sel_acts, -1, 1)
        sel_pool = jax.lax.stop_gradient(
            sel_acts.reshape(batch_size, N, action_dim)
        )  # [B, N, D]

        qs_sel = self.network.select("critic")(obs_rep_N, actions=sel_acts)
        q_sel = getattr(jnp, self.config["q_agg"])(qs_sel, axis=0).reshape(
            batch_size, N
        )
        cand_qs = jax.lax.stop_gradient(q_sel)  # [B, N]

        sorted_idx = jnp.argsort(cand_qs, axis=-1)
        pos_idx = sorted_idx[:, -a:]  # highest Q
        neg_idx = sorted_idx[:, :neg_b]
        pos_acts = jnp.take_along_axis(sel_pool, pos_idx[..., None], axis=1)  # [B, a, D]
        neg_acts = jnp.take_along_axis(sel_pool, neg_idx[..., None], axis=1)  # [B, neg_b, D]

        # Dataset target action (kept separate from pos_acts). Sourced from
        # `actor_actions` so the BC target lives on the same observation
        # distribution as `obs` (offline if bc fields are present, else mixed).
        target_act = jnp.expand_dims(actor_actions, axis=1)  # [B, 1, D]

        # --- Phase 2: b fresh samples for drift gen + Q loss ---
        obs_rep_b = jnp.repeat(obs, b, axis=0)
        noises_drift = jax.random.normal(drift_rng, (batch_size * b, action_dim))
        gen_acts = self.network.select("actor_drift")(
            obs_rep_b, noises_drift, params=grad_params
        )
        gen_acts = jnp.clip(gen_acts, -1, 1)
        gen_samples = gen_acts.reshape(batch_size, b, action_dim)  # [B, b, D]

        # Q loss on the b fresh samples
        qs_gen = self.network.select("critic")(obs_rep_b, actions=gen_acts)
        q_gen = getattr(jnp, self.config["q_agg"])(qs_gen, axis=0).reshape(
            batch_size, b
        )
        q_loss = -q_gen.mean()
        lam = jax.lax.stop_gradient(1 / (jnp.abs(q_gen).mean()))
        if self.config["normalize_q_loss"]:
            q_loss = lam * q_loss
        q_loss = self.config["q_alpha"] * q_loss

        R_list = tuple(self.config.get("drift_temps", [0.1]))
        plus_only = bool(self.config.get("drift_plus_only", False))
        use_neg_only = bool(self.config.get("drift_use_neg_only", False))

        if mode == "hard":
            drift_loss_pos, drift_info_pos = drift_loss(
                gen=gen_samples,
                fixed_pos=pos_acts,
                fixed_neg=neg_acts,
                R_list=R_list,
                plus_only=plus_only,
                use_neg_only=use_neg_only,
            )
            g_info = {}
        else:  # soft
            # Normalize Q values to [-1, 1]
            q_min = cand_qs.min(axis=-1, keepdims=True)
            q_max = cand_qs.max(axis=-1, keepdims=True)
            norm_qs = 2.0 * (cand_qs - q_min) / jnp.clip(q_max - q_min, a_min=1e-8) - 1.0

            pos_q = jnp.take_along_axis(norm_qs, pos_idx, axis=1)  # [B, a], close to +1
            weight_pos = jnp.clip(pos_q, a_min=0.0)  # [B, a]

            drift_loss_pos, drift_info_pos = drift_loss(
                gen=gen_samples,
                fixed_pos=pos_acts,
                fixed_neg=None,
                weight_pos=weight_pos,
                R_list=R_list,
                plus_only=plus_only,
                use_neg_only=use_neg_only
            )
            g_info = {
                "weight_pos_mean": weight_pos.mean(),
            }

        # BC term reuses Phase 2's gen_samples so we don't pay for an extra
        # actor forward pass; both gen_samples and target_act live on the same
        # observations (`obs`).
        drift_loss_tgt, drift_info_tgt = drift_loss(
            gen=gen_samples,
            fixed_pos=target_act,
            fixed_neg=None,
            R_list=R_list,
            plus_only=plus_only,
        )

        alpha_default = self.config.get("alpha", 1.0)
        alpha_pos = self.config.get("alpha_pos", None)
        alpha_pos = alpha_default if alpha_pos is None else alpha_pos
        alpha_target = self.config.get("alpha_target", None)
        alpha_target = alpha_default if alpha_target is None else alpha_target
        actor_drift_loss_pos = alpha_pos * drift_loss_pos.mean()
        actor_drift_loss_tgt = alpha_target * drift_loss_tgt.mean()
        actor_drift_loss = actor_drift_loss_pos + actor_drift_loss_tgt
        actor_loss = actor_drift_loss + q_loss

        info = dict(
            actor_loss=actor_loss,
            actor_drift_loss=actor_drift_loss,
            actor_drift_loss_pos=actor_drift_loss_pos,
            actor_drift_loss_tgt=actor_drift_loss_tgt,
            drift_scale_pos=drift_info_pos.get("scale", 0.0),
            drift_scale_tgt=drift_info_tgt.get("scale", 0.0),
            q_loss=q_loss,
            online_pos_q_mean=jnp.take_along_axis(cand_qs, pos_idx, axis=1).mean(),
            online_sel_q_mean=cand_qs.mean(),
            online_gen_q_mean=q_gen.mean(),
            **g_info,
        )
        for key, val in drift_info_pos.items():
            if key.startswith("loss_"):
                info[f"drift_pos_{key}"] = val
        for key, val in drift_info_tgt.items():
            if key.startswith("loss_"):
                info[f"drift_tgt_{key}"] = val
        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    def actor_ema_update(self, network):
        """Polyak update for the actor EMA (online-stage only)."""
        tau = self.config.get("actor_ema_tau")
        new_ema_params = jax.tree_util.tree_map(
            lambda p, ep: p * tau + ep * (1 - tau),
            self.network.params["modules_actor_drift"],
            self.network.params["modules_actor_drift_ema"],
        )
        network.params["modules_actor_drift_ema"] = new_ema_params

    def switch_config_to_online(self):
        new_config = self.config.copy({
            "drift_mode": self.config.get("online_drift_mode", "none"),
            "use_actor_ema": True,  # ← 추가
        })
        # (Re-)initialize EMA params from current actor_drift so the EMA tracks
        # only the online-stage trajectory, not pretraining.
        new_params = dict(self.network.params)
        new_params["modules_actor_drift_ema"] = new_params["modules_actor_drift"]
        new_network = self.network.replace(params=new_params)
        return self.replace(config=new_config, network=new_network)

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, "critic")
        if agent.config.get("use_actor_ema", False):  
            agent.actor_ema_update(new_network)      
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    def sample_noises(self, obs, rng):
        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )
        return jax.random.normal(
            rng,
            (
                *obs.shape[: -len(self.config["ob_dims"])],
                full_action_dim,
            ),
        )

    def _score_actions(self, observations, actions):
        qs = self.network.select("critic")(observations, actions)
        if self.config["q_agg"] == "mean":
            return qs.mean(axis=0)
        return qs.min(axis=0)

    def _select_best_bon_action(self, actions, q_values):
        indices = jnp.argmax(q_values, axis=-1)
        bshape = indices.shape
        indices = indices.reshape(-1)
        bsize = len(indices)
        return jnp.reshape(actions, (-1, actions.shape[-2], actions.shape[-1]))[
            jnp.arange(bsize), indices, :
        ].reshape(bshape + (actions.shape[-1],))

    @partial(jax.jit, static_argnames=("use_q_bon",))
    def sample_actions(self, observations, rng=None, use_q_bon=False):
        """Sample actions with either direct actor or best-of-n search."""
        if rng is None:
            rng = jax.random.PRNGKey(0)
        actor_type = self.config.get("actor_type", "distill-ddpg")
        full_action_dim = self.config["action_dim"] * (
            self.config["horizon_length"] if self.config["action_chunking"] else 1
        )

        if actor_type == "best-of-n":
            num_samples = self._get_bon_samples(use_q_bon=use_q_bon)
            rng, init_noise_rng = jax.random.split(rng)
            noises = jax.random.normal(
                init_noise_rng,
                (
                    *observations.shape[: -len(self.config["ob_dims"])],
                    num_samples,
                    full_action_dim,
                ),
            )
            obs_rep = jnp.repeat(observations[..., None, :], num_samples, axis=-2)
            actions = self.network.select("actor_drift")(obs_rep, noises)
            actions = jnp.clip(actions, -1, 1)

            q = self._score_actions(obs_rep, actions)
            return self._select_best_bon_action(actions, q)

        noises = self.sample_noises(observations, rng)
        actions = self.network.select("actor_drift")(observations, noises)
        return jnp.clip(actions, -1, 1)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            encoders["critic"] = encoder_module()
            encoders["actor_drift"] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config.get("num_qs", 2),
            encoder=encoders.get("critic"),
        )
        actor_drift_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_drift"),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_drift=(actor_drift_def, (ex_observations, full_actions)),
            actor_drift_ema=(copy.deepcopy(actor_drift_def), (ex_observations, full_actions)),  # ← 추가
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]
        # EMA mirrors actor_drift; only updated after switch_config_to_online.
        params["modules_actor_drift_ema"] = params["modules_actor_drift"]  

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        if config.get("actor_type") is None:
            config["actor_type"] = "distill-ddpg"
        if config.get("actor_num_samples") is None:
            config["actor_num_samples"] = 32
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="drift",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg="mean",
            num_qs=2,
            alpha=0.01,
            alpha_pos=ml_collections.config_dict.placeholder(float),    # online: weight for top-a pos drift loss (default: alpha)
            alpha_target=ml_collections.config_dict.placeholder(float), # online: weight for dataset target drift loss (default: alpha)
            normalize_q_loss=True,
            q_alpha=1.0,
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=False,
            actor_type="distill-ddpg",
            actor_num_samples=32,
            q_bon=-1,
            eval_bon=-1,
            drift_temps=(0.02, 0.05, 0.2),
            gen_per_label=8,
            # Online drift mode params
            drift_mode="none",              # Active mode: "none" / "hard" / "soft" (set by switch_config_to_online)
            online_drift_mode="none",       # Target mode for online: "none" / "hard" / "soft"
            online_n_samples=32,            # N: samples to generate, score with Q, and use as gen
            online_pos_samples=4,           # a: top-Q samples used as positive
            online_neg_samples=4,           # b: bottom-Q samples used as negative
            drift_plus_only=False,          # If True, only use the positive drifting field in drift_loss
            drift_use_neg_only=False,       # If True, drift_loss negatives use only fixed_neg (no implicit gen-as-neg)
            # Actor EMA (for N-sample pool selection in online stage)
            use_actor_ema=False,            # Set True automatically by switch_config_to_online
            actor_ema_tau=ml_collections.config_dict.placeholder(float),  # If None, falls back to `tau`
        )
    )
    return config
