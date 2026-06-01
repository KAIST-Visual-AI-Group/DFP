"""
MeanFlow Policy with Q-Learning (1-step)

- Using 1-step meanflow loss (mean velocity + IVC)
- No distillation loss
- No guidance weighting (pure meanflow + Q loss)
- Q-learning for value estimation
"""

import copy
from typing import Any
from functools import partial

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from jax.scipy.special import logsumexp

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


class MeanflowAgent(flax.struct.PyTreeNode):
    """MeanFlow Policy agent with meanflow loss."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _get_bon_samples(self, use_q_bon: bool):
        key = "q_bon" if use_q_bon else "eval_bon"
        n = self.config[key]
        if n <= 0:
            n = self.config["actor_num_samples"]
        return n

    def _sample_time_values(self, rng, size):
        sampler = self.config.get("tr_sampler", "uniform")
        if sampler == "uniform":
            return jax.random.uniform(rng, (size,), minval=1e-6, maxval=1.0 - 1e-6)
        if sampler == "lognorm":
            mu = self.config.get("tr_lognorm_mu", -0.4)
            sigma = self.config.get("tr_lognorm_sigma", 1.0)
            z = jax.random.normal(rng, (size,)) * sigma + mu
            x = jnp.exp(z)
            # Map (0, +inf) to (0, 1) for time variables.
            u = x / (1.0 + x)
            return jnp.clip(u, 1e-6, 1.0 - 1e-6)
        raise ValueError(f"Unsupported tr_sampler '{sampler}'")

    def _sample_t_r(self, rng, size):
        rng_t, rng_r = jax.random.split(rng)
        t_raw = self._sample_time_values(rng_t, size)
        r_raw = self._sample_time_values(rng_r, size)
        # Sample independently first, then enforce t > r by swapping.
        t = jnp.maximum(t_raw, r_raw)
        r = jnp.minimum(t_raw, r_raw)
        return t, r

    def critic_loss(self, batch, grad_params, rng):
        """Compute the MeanFlow critic loss."""
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

        # Metrics:
        info = {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "tgt_q_mean": next_q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

        return critic_loss, info

    def meanflow_scalar_loss(self, obs, action, eps, r, t, grad_params):
        """Compute meanflow JVP loss + IVC for single sample."""
        # Straight path: z = (1-t)*a + t*eps
        z = (1.0 - t) * action + t * eps
        v = eps - action

        if self.config["encoder"] is not None:
            encoded_obs = self.network.select("actor_meanflow_encoder")(obs, params=grad_params)

            def mean_u_fn(actions, time_interval):
                return self.network.select("actor_meanflow")(
                    encoded_obs,
                    actions,
                    time_interval,
                    params=grad_params,
                    is_encoded=True,
                )
        else:
            def mean_u_fn(actions, time_interval):
                return self.network.select("actor_meanflow")(
                    obs,
                    actions,
                    time_interval,
                    params=grad_params,
                )

        def u_fn(z_, r_, t_):
            rt = jnp.stack([r_, t_], axis=-1)
            return mean_u_fn(z_, rt)

        # JVP computation
        (u_pred, dudt) = jax.jvp(
            lambda z_, r_, t_: u_fn(z_, r_, t_),
            (z, r, t),
            (v, jnp.zeros_like(r), jnp.ones_like(t)),
        )
        u_tgt = jax.lax.stop_gradient(v - (t - r) * dudt)
        mf_loss = jnp.square(u_pred - u_tgt)

        # IVC: instantaneous velocity constraint at r
        z_r = (1.0 - r) * action + r * eps
        u_rr = u_fn(z_r, r, r)
        ivc_loss = jnp.square(u_rr - v)

        return mf_loss, ivc_loss

    def actor_loss(self, batch, grad_params, rng):
        """Compute actor loss with meanflow and guidance."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        batch_size, action_dim = batch_actions.shape

        if self.config.get("online_top_k", 0) > 0:
            return self._actor_loss_online_top_k(
                batch, batch_actions, grad_params, rng
            )

        rng, mf_rng, q_rng = jax.random.split(rng, 3)

        # Get actor actions for Q loss and guidance
        actor_noises = jax.random.normal(q_rng, (batch_size, action_dim))
        r0 = jnp.zeros((batch_size, 1), dtype=actor_noises.dtype)
        t1 = jnp.ones((batch_size, 1), dtype=actor_noises.dtype)
        rt = jnp.concatenate([r0, t1], axis=-1)
        if self.config["encoder"] is not None:
            encoded_obs = self.network.select("actor_meanflow_encoder")(
                batch["observations"], params=grad_params
            )
            actor_u = self.network.select("actor_meanflow")(
                encoded_obs,
                actor_noises,
                rt,
                params=grad_params,
                is_encoded=True,
            )
        else:
            actor_u = self.network.select("actor_meanflow")(
                batch["observations"],
                actor_noises,
                rt,
                params=grad_params,
            )
        actor_acts = actor_noises - actor_u  # 1-step: a = eps - u(eps, r=0, t=1)
        actor_acts = jnp.clip(actor_acts, -1, 1)

        # Q loss
        if self.config["q_alpha"] > 0:
            qs = self.network.select("critic")(batch["observations"], actions=actor_acts)
            q = getattr(jnp, self.config["q_agg"])(qs, axis=0)
            q_loss = -q.mean()
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            if self.config["normalize_q_loss"]:
                q_loss = lam * q_loss
            q_loss = self.config["q_alpha"] * q_loss
        else:
            q_loss = 0.0
            lam = 1.0

        # Meanflow loss with guidance
        n_samples = self.config.get("n_samples_per_action", 1)
        obs_repeated = jnp.repeat(batch["observations"], n_samples, axis=0)
        act_repeated = jnp.repeat(batch_actions, n_samples, axis=0)

        # Sample eps, r, t
        # eps_samples = jax.random.normal(mf_rng, (batch_size * n_samples, action_dim))
        
        # couple noises
        assert n_samples == 1, "Coupled noises only implemented for n_samples=1"
        eps_samples = actor_noises
        
        rng, tr_rng = jax.random.split(rng)
        t_samples, r_samples = self._sample_t_r(tr_rng, batch_size * n_samples)

        # Vectorized meanflow loss computation
        mf_losses, ivc_losses = jax.vmap(
            lambda obs, act, eps, r, t: self.meanflow_scalar_loss(
                obs, act, eps, r, t, grad_params
            )
        )(obs_repeated, act_repeated, eps_samples, r_samples, t_samples)

        if self.config["action_chunking"]:
            mf_losses = jnp.reshape(
                mf_losses,
                (batch_size, n_samples, self.config["horizon_length"], self.config["action_dim"]),
            )
            ivc_losses = jnp.reshape(
                ivc_losses,
                (batch_size, n_samples, self.config["horizon_length"], self.config["action_dim"]),
            )
            valid = (
                batch["valid"][:, None, :, None]
                if "valid" in batch
                else jnp.ones(
                    (batch_size, 1, self.config["horizon_length"], 1),
                    dtype=mf_losses.dtype,
                )
            )
            mf_losses_per_sample = jnp.mean(mf_losses * valid, axis=(-1, -2))
            ivc_losses_per_sample = jnp.mean(ivc_losses * valid, axis=(-1, -2))
            mf_loss_metric = jnp.mean(mf_losses_per_sample)
            ivc_loss_metric = jnp.mean(ivc_losses_per_sample)
        else:
            mf_losses = jnp.reshape(mf_losses, (batch_size, n_samples, action_dim))
            ivc_losses = jnp.reshape(ivc_losses, (batch_size, n_samples, action_dim))
            mf_losses_per_sample = jnp.mean(mf_losses, axis=-1)
            ivc_losses_per_sample = jnp.mean(ivc_losses, axis=-1)
            mf_loss_metric = jnp.mean(mf_losses_per_sample)
            ivc_loss_metric = jnp.mean(ivc_losses_per_sample)

        # Combine MF + IVC
        ivc_lambda = self.config.get("ivc_lambda", 0.0)
        combined_losses = mf_losses_per_sample + ivc_lambda * ivc_losses_per_sample
        meanflow_loss_per_sample = combined_losses.mean(axis=1)  # [B]

        # Apply guidance weighting if enabled
        if self.config["eta_temperature"] == 0:
            actor_meanflow_loss = meanflow_loss_per_sample.mean()
            g_info = dict()
        else:
            q_ref = self.network.select("target_critic")(
                batch["observations"], actions=jax.lax.stop_gradient(actor_acts)
            )
            q_ref = getattr(jnp, self.config["q_agg"])(q_ref, axis=0)
            q_data = self.network.select("target_critic")(
                batch["observations"], actions=batch_actions
            )
            q_data = getattr(jnp, self.config["q_agg"])(q_data, axis=0)
            scale = lam / self.config["eta_temperature"]

            assert self.config["guidance_fn"] in ["softmax", "advantage"]
            scl_q_data = scale * q_data
            scl_q_ref = scale * q_ref
            if self.config["guidance_fn"] == "softmax":
                log_denominator = logsumexp(
                    jnp.stack([scl_q_data, scl_q_ref], axis=-1), axis=-1
                )
                guidance = jnp.exp(scl_q_data - log_denominator)
            else:
                guidance = jnp.exp(scl_q_data - scl_q_ref)
            actor_meanflow_loss = jnp.mean(guidance * meanflow_loss_per_sample)
            g_info = {
                "q_data": q_data.mean(),
                "q_ref": q_ref.mean(),
                "scaled_q_data": scl_q_data.mean(),
                "scaled_q_ref": scl_q_ref.mean(),
                "guidance": guidance.mean(),
            }

        # Apply alpha to match Q loss magnitude
        alpha = self.config.get("alpha", 1.0)
        actor_meanflow_loss = alpha * actor_meanflow_loss

        # Total loss
        actor_loss = actor_meanflow_loss + q_loss

        info = dict(
            actor_loss=actor_loss,
            actor_meanflow_loss=actor_meanflow_loss,
            mf_loss=mf_loss_metric,
            ivc_loss=ivc_loss_metric,
            q_loss=q_loss,
            **g_info,
        )
        return actor_loss, info

    def _actor_loss_online_top_k(self, batch, batch_actions, grad_params, rng):
        """Online actor loss: best-K-of-N candidates + dataset target action.

        Phase 1: generate N proposal candidates (with the EMA actor when
                 ``use_actor_ema`` is True, otherwise the trainable actor with no
                 grad), score with Q, and select top-K as fixed positive
                 meanflow targets (all stop-gradient).
        Phase 2: if ``q_alpha > 0``, generate fresh samples with the trainable
                 actor for the Q loss only. (This phase is skipped when
                 ``q_alpha == 0`` to avoid an unused forward pass.)
        Targets passed to meanflow loss = top-K (stop-grad) + 1 dataset target.
        Each target is paired with an independent (eps, r, t).
        """
        batch_size, action_dim = batch_actions.shape
        rng, sel_noise_rng, q_noise_rng, mf_rng, tr_rng = jax.random.split(rng, 5)

        N = self.config.get("online_n_samples", 32)
        K = self.config.get("online_top_k", 4)
        obs = batch["observations"]

        # --- Phase 1: N-sample pool for positive selection (stop-gradient) ---
        obs_rep_N = jnp.repeat(obs, N, axis=0)
        sel_noises = jax.random.normal(sel_noise_rng, (batch_size * N, action_dim))
        r0_N = jnp.zeros((batch_size * N, 1), dtype=sel_noises.dtype)
        t1_N = jnp.ones((batch_size * N, 1), dtype=sel_noises.dtype)
        rt_N = jnp.concatenate([r0_N, t1_N], axis=-1)

        use_ema = self.config.get("use_actor_ema", False)
        sel_actor_name = "actor_meanflow_ema" if use_ema else "actor_meanflow"
        if self.config["encoder"] is not None:
            sel_encoder_name = (
                "actor_meanflow_encoder_ema" if use_ema else "actor_meanflow_encoder"
            )
            sel_encoded = self.network.select(sel_encoder_name)(obs_rep_N)
            sel_u = self.network.select(sel_actor_name)(
                sel_encoded, sel_noises, rt_N, is_encoded=True,
            )
        else:
            sel_u = self.network.select(sel_actor_name)(
                obs_rep_N, sel_noises, rt_N,
            )
        sel_acts = jnp.clip(sel_noises - sel_u, -1, 1)
        sel_pool = jax.lax.stop_gradient(
            sel_acts.reshape(batch_size, N, action_dim)
        )  # [B, N, D]

        qs_sel = self.network.select("critic")(obs_rep_N, actions=sel_acts)
        q_sel = getattr(jnp, self.config["q_agg"])(qs_sel, axis=0).reshape(
            batch_size, N
        )
        cand_qs = jax.lax.stop_gradient(q_sel)  # [B, N]
        sorted_idx = jnp.argsort(cand_qs, axis=-1)
        top_k_idx = sorted_idx[:, -K:]  # [B, K]
        top_k_acts = jnp.take_along_axis(
            sel_pool, top_k_idx[..., None], axis=1
        )  # [B, K, D]

        # --- Phase 2: fresh trainable-actor samples for Q loss ---
        if self.config["q_alpha"] > 0:
            q_noises = jax.random.normal(q_noise_rng, (batch_size * N, action_dim))
            if self.config["encoder"] is not None:
                q_encoded = self.network.select("actor_meanflow_encoder")(
                    obs_rep_N, params=grad_params
                )
                q_u = self.network.select("actor_meanflow")(
                    q_encoded, q_noises, rt_N,
                    params=grad_params, is_encoded=True,
                )
            else:
                q_u = self.network.select("actor_meanflow")(
                    obs_rep_N, q_noises, rt_N, params=grad_params,
                )
            q_acts = jnp.clip(q_noises - q_u, -1, 1)
            qs_q = self.network.select("critic")(obs_rep_N, actions=q_acts)
            q_q = getattr(jnp, self.config["q_agg"])(qs_q, axis=0)
            q_loss = -q_q.mean()
            lam = jax.lax.stop_gradient(1 / jnp.abs(q_q).mean())
            if self.config["normalize_q_loss"]:
                q_loss = lam * q_loss
            q_loss = self.config["q_alpha"] * q_loss
        else:
            q_loss = jnp.zeros(())

        # Append the dataset target action as the (K+1)-th target.
        target_act = jnp.expand_dims(batch_actions, axis=1)  # [B, 1, D]
        all_targets = jnp.concatenate([top_k_acts, target_act], axis=1)  # [B, K+1, D]
        M = K + 1

        # Meanflow loss over K+1 targets with independent (eps, r, t) per target.
        flat_targets = all_targets.reshape(batch_size * M, action_dim)
        obs_rep_M = jnp.repeat(obs, M, axis=0)
        eps_samples = jax.random.normal(mf_rng, (batch_size * M, action_dim))
        t_samples, r_samples = self._sample_t_r(tr_rng, batch_size * M)

        mf_losses, ivc_losses = jax.vmap(
            lambda obs, act, eps, r, t: self.meanflow_scalar_loss(
                obs, act, eps, r, t, grad_params
            )
        )(obs_rep_M, flat_targets, eps_samples, r_samples, t_samples)

        if self.config["action_chunking"]:
            mf_losses = jnp.reshape(
                mf_losses,
                (
                    batch_size,
                    M,
                    self.config["horizon_length"],
                    self.config["action_dim"],
                ),
            )
            ivc_losses = jnp.reshape(
                ivc_losses,
                (
                    batch_size,
                    M,
                    self.config["horizon_length"],
                    self.config["action_dim"],
                ),
            )
            mf_losses_per = jnp.mean(mf_losses, axis=(-1, -2))  # [B, M]
            ivc_losses_per = jnp.mean(ivc_losses, axis=(-1, -2))
        else:
            mf_losses = mf_losses.reshape(batch_size, M, action_dim)
            ivc_losses = ivc_losses.reshape(batch_size, M, action_dim)
            mf_losses_per = jnp.mean(mf_losses, axis=-1)  # [B, M]
            ivc_losses_per = jnp.mean(ivc_losses, axis=-1)

        mf_loss_metric = mf_losses_per.mean()
        ivc_loss_metric = ivc_losses_per.mean()

        ivc_lambda = self.config.get("ivc_lambda", 0.0)
        combined = mf_losses_per + ivc_lambda * ivc_losses_per  # [B, M]

        # Split into K generated positives and 1 dataset target.
        combined_pos = combined[:, :K]   # [B, K]
        combined_tgt = combined[:, K:]   # [B, 1]

        alpha_default = self.config.get("alpha", 1.0)
        alpha_pos = self.config.get("alpha_pos", None)
        alpha_pos = alpha_default if alpha_pos is None else alpha_pos
        alpha_target = self.config.get("alpha_target", None)
        alpha_target = alpha_default if alpha_target is None else alpha_target

        actor_meanflow_loss_pos = alpha_pos * combined_pos.mean()
        actor_meanflow_loss_tgt = alpha_target * combined_tgt.mean()
        actor_meanflow_loss = actor_meanflow_loss_pos + actor_meanflow_loss_tgt

        actor_loss = actor_meanflow_loss + q_loss

        info = dict(
            actor_loss=actor_loss,
            actor_meanflow_loss=actor_meanflow_loss,
            actor_meanflow_loss_pos=actor_meanflow_loss_pos,
            actor_meanflow_loss_tgt=actor_meanflow_loss_tgt,
            mf_loss=mf_loss_metric,
            ivc_loss=ivc_loss_metric,
            q_loss=q_loss,
            online_top_k_q_mean=jnp.take_along_axis(cand_qs, top_k_idx, axis=1).mean(),
            online_cand_q_mean=cand_qs.mean(),
        )
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
        """Polyak update for the actor (and encoder) EMA (online-stage only)."""
        tau = self.config.get("actor_ema_tau")
        new_ema = jax.tree_util.tree_map(
            lambda p, ep: p * tau + ep * (1 - tau),
            self.network.params["modules_actor_meanflow"],
            self.network.params["modules_actor_meanflow_ema"],
        )
        network.params["modules_actor_meanflow_ema"] = new_ema
        if self.config["encoder"] is not None:
            new_enc_ema = jax.tree_util.tree_map(
                lambda p, ep: p * tau + ep * (1 - tau),
                self.network.params["modules_actor_meanflow_encoder"],
                self.network.params["modules_actor_meanflow_encoder_ema"],
            )
            network.params["modules_actor_meanflow_encoder_ema"] = new_enc_ema

    def switch_config_to_online(self):
        new_config = self.config.copy(
            {
                "eta_temperature": self.config["eta_temperature_online"],
                "use_actor_ema": True,
            }
        )
        # (Re-)initialize EMA params from current actor so the EMA tracks only
        # the online-stage trajectory, not pretraining.
        new_params = dict(self.network.params)
        new_params["modules_actor_meanflow_ema"] = new_params["modules_actor_meanflow"]
        if self.config["encoder"] is not None:
            new_params["modules_actor_meanflow_encoder_ema"] = (
                new_params["modules_actor_meanflow_encoder"]
            )
        new_network = self.network.replace(params=new_params)
        return self.replace(config=new_config, network=new_network)

    @staticmethod
    def _update(agent, batch):
        """Single gradient update used by both update and batch_update."""
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

    def _predict_u(self, observations, noises, r, t):
        rt = jnp.concatenate([r, t], axis=-1)
        if self.config["encoder"] is not None:
            observations = self.network.select("actor_meanflow_encoder")(observations)
            return self.network.select("actor_meanflow")(
                observations,
                noises,
                rt,
                is_encoded=True,
            )
        return self.network.select("actor_meanflow")(
            observations,
            noises,
            rt,
        )

    def _actions_from_noises(self, observations, noises, r, t):
        u_pred = self._predict_u(observations, noises, r, t)
        actions = noises - u_pred
        return jnp.clip(actions, -1, 1)

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
        """Sample actions with direct one-step meanflow or best-of-n search."""
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

            r0 = jnp.zeros((*noises.shape[:-1], 1), dtype=noises.dtype)
            t1 = jnp.ones((*noises.shape[:-1], 1), dtype=noises.dtype)
            actions = self._actions_from_noises(obs_rep, noises, r0, t1)

            q = self._score_actions(obs_rep, actions)
            return self._select_best_bon_action(actions, q)

        noises = self.sample_noises(observations, rng)
        r0 = jnp.zeros((*noises.shape[:-1], 1), dtype=noises.dtype)
        t1 = jnp.ones((*noises.shape[:-1], 1), dtype=noises.dtype)
        return self._actions_from_noises(observations, noises, r0, t1)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent."""
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
            encoders["actor_meanflow"] = encoder_module()

        # Define networks.
        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config.get("num_qs", 2),
            encoder=encoders.get("critic"),
        )

        ex_times = ex_actions[..., :1]
        ex_time_intervals = jnp.concatenate([jnp.zeros_like(ex_times), jnp.ones_like(ex_times)], axis=-1)

        actor_meanflow_def = ActorVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=full_action_dim,
            layer_norm=config["actor_layer_norm"],
            encoder=encoders.get("actor_meanflow"),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor_meanflow=(actor_meanflow_def, (ex_observations, full_actions, ex_time_intervals)),
            actor_meanflow_ema=(
                copy.deepcopy(actor_meanflow_def),
                (ex_observations, full_actions, ex_time_intervals),
            ),
        )
        if encoders.get("actor_meanflow") is not None:
            network_info["actor_meanflow_encoder"] = (encoders.get("actor_meanflow"), (ex_observations,))
            network_info["actor_meanflow_encoder_ema"] = (
                copy.deepcopy(encoders.get("actor_meanflow")),
                (ex_observations,),
            )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        actor_lr = config.get("actor_lr")
        if actor_lr is None:
            network_tx = optax.adam(learning_rate=config["lr"])
        else:
            def _label_params(params):
                return {
                    k: ("actor" if k.startswith("modules_actor_meanflow") else "other")
                    for k in params.keys()
                }
            network_tx = optax.multi_transform(
                {
                    "actor": optax.adam(learning_rate=actor_lr),
                    "other": optax.adam(learning_rate=config["lr"]),
                },
                _label_params,
            )
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_critic"] = params["modules_critic"]
        # EMA mirrors actor_meanflow; only updated after switch_config_to_online.
        params["modules_actor_meanflow_ema"] = params["modules_actor_meanflow"]
        if encoders.get("actor_meanflow") is not None:
            params["modules_actor_meanflow_encoder_ema"] = (
                params["modules_actor_meanflow_encoder"]
            )

        config["ob_dims"] = ob_dims
        config["action_dim"] = action_dim
        if config.get("actor_type") is None:
            config["actor_type"] = "distill-ddpg"
        if config.get("actor_num_samples") is None:
            config["actor_num_samples"] = 32
        if config.get("eta_temperature_online") is None:
            config["eta_temperature_online"] = config["eta_temperature"]
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="meanflow",
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            actor_lr=ml_collections.config_dict.placeholder(float),
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg="mean",
            num_qs=2,
            alpha=1.0,
            alpha_pos=ml_collections.config_dict.placeholder(float),    # online: weight for top-K generated mf loss (default: alpha)
            alpha_target=ml_collections.config_dict.placeholder(float), # online: weight for dataset target mf loss (default: alpha)
            normalize_q_loss=True,
            q_alpha=1.0,
            encoder=ml_collections.config_dict.placeholder(str),
            horizon_length=ml_collections.config_dict.placeholder(int),
            action_chunking=False,
            actor_type="distill-ddpg",
            actor_num_samples=32,
            q_bon=-1,
            eval_bon=-1,
            eta_temperature=0.0,
            eta_temperature_online=ml_collections.config_dict.placeholder(float),
            guidance_fn="softmax",
            n_samples_per_action=8,
            ivc_lambda=0.0,
            use_fourier_features=False,
            fourier_feature_dim=64,
            tr_sampler="uniform",
            tr_lognorm_mu=-0.4,
            tr_lognorm_sigma=1.0,
            # Online top-K of N params (0 disables the online branch).
            online_top_k=0,             # K: number of top-Q candidates used as extra targets
            online_n_samples=32,        # N: number of candidates generated and ranked by Q
            # Actor EMA (used to propose top-K candidates in the online stage).
            use_actor_ema=False,        # Set True automatically by switch_config_to_online
            actor_ema_tau=ml_collections.config_dict.placeholder(float),  # Polyak rate for the actor EMA
        )
    )
    return config
