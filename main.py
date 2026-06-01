import glob, tqdm, wandb, os, json, random, time, jax, yaml, sys
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.robomimic_utils import is_robomimic_env

from utils.flax_utils import save_agent, restore_agent_with_file
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate
from agents import agents
import numpy as np

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS


def _infer_agent_config_flag_default(argv):
    """Pick a better default config file for --agent.* override type resolution."""
    # If user already passed --agent=..., respect that.
    for i, arg in enumerate(argv[1:], start=1):
        if arg.startswith('--agent='):
            value = arg.split('=', 1)[1].strip()
            if value:
                return value
        if arg == '--agent' and i + 1 < len(argv):
            value = argv[i + 1].strip()
            if value and not value.startswith('--'):
                return value

    # Otherwise infer from --agent_config=<name> when possible.
    agent_cfg_name = None
    for i, arg in enumerate(argv[1:], start=1):
        if arg.startswith('--agent_config='):
            candidate = arg.split('=', 1)[1].strip()
            if candidate:
                agent_cfg_name = candidate
                break
        if arg == '--agent_config' and i + 1 < len(argv):
            candidate = argv[i + 1].strip()
            if candidate and not candidate.startswith('--'):
                agent_cfg_name = candidate
                break

    if agent_cfg_name:
        candidate_path = os.path.join('agents', f'{agent_cfg_name}.py')
        if os.path.exists(candidate_path):
            return candidate_path

    return 'agents/acfql.py'


AGENT_CONFIG_FLAG_DEFAULT = _infer_agent_config_flag_default(sys.argv)


def _ogbench_dataset_name_from_env_name(env_name):
    """Infer the OGBench dataset basename from an env flag value."""
    splits = env_name.split('-')
    if 'singletask' in splits:
        pos = splits.index('singletask')
        # Strip size tags (e.g. '100m') — they appear in directory names, not file names.
        prefix = [s for s in splits[:pos] if not (s.endswith('m') and s[:-1].isdigit())]
        return '-'.join(prefix + splits[-1:])
    if 'oraclerep' in splits:
        return '-'.join(splits[:-2] + splits[-1:])
    return env_name


def _ogbench_task_index_from_env_name(env_name):
    """Extract singletask task index from env name. Returns 0 if not found."""
    splits = env_name.split('-')
    if 'singletask' in splits:
        pos = splits.index('singletask')
        # Look for 'taskN' right after 'singletask'.
        if pos + 1 < len(splits) and splits[pos + 1].startswith('task'):
            return int(splits[pos + 1][len('task'):])
    return 0


def _read_yaml_file(path):
    if path is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML config not found: {path}")
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data or {}


def _extract_agent_config_name(main_cfg):
    if 'agent_config' in main_cfg and main_cfg['agent_config'] is not None:
        return main_cfg['agent_config']
    defaults = main_cfg.get('defaults', [])
    for item in defaults:
        if isinstance(item, dict) and 'agent' in item:
            return item['agent']
    return None


def _collect_cli_overrides(argv):
    flag_names = set()
    agent_keys = set()
    for arg in argv[1:]:
        if not arg.startswith('--'):
            continue
        token = arg[2:]
        if token == '':
            continue
        if token.startswith('no') and '=' not in token:
            name = token[2:]
        else:
            name = token.split('=', 1)[0]
        if name == '':
            continue
        flag_names.add(name)
        if name.startswith('agent.'):
            agent_keys.add(name.split('.', 1)[1])
    return flag_names, agent_keys


def _apply_main_yaml_overrides(main_cfg, agent_cfg, cli_flag_names, cli_agent_keys):
    flag_key_map = {
        'run_group': 'run_group',
        'wandb_project': 'wandb_project',
        'seed': 'seed',
        'env_name': 'env_name',
        'agent_config': 'agent_config',
        'exp_name_flags': 'exp_name_flags',
        'exp_name_tags': 'exp_name_tags',
        'save_dir': 'save_dir',
        'restore_path': 'restore_path',
        'restore_epoch': 'restore_epoch',
        'restore_params_only': 'restore_params_only',
        'offline_steps': 'offline_steps',
        'online_steps': 'online_steps',
        'buffer_size': 'buffer_size',
        'log_interval': 'log_interval',
        'eval_interval': 'eval_interval',
        'save_interval': 'save_interval',
        'start_training': 'start_training',
        'utd_ratio': 'utd_ratio',
        'discount': 'discount',
        'eval_episodes': 'eval_episodes',
        'video_episodes': 'video_episodes',
        'video_frame_skip': 'video_frame_skip',
        'dataset_proportion': 'dataset_proportion',
        'dataset_replace_interval': 'dataset_replace_interval',
        'ogbench_dataset_dir': 'ogbench_dataset_dir',
        'horizon_length': 'horizon_length',
        'sparse': 'sparse',
        'save_all_online_states': 'save_all_online_states',
        'offline_batch_ratio': 'offline_batch_ratio',
        'separate_offline_buffer': 'separate_offline_buffer',
    }
    for yaml_key, flag_key in flag_key_map.items():
        if flag_key in cli_flag_names:
            continue
        if yaml_key in main_cfg and main_cfg[yaml_key] is not None:
            coerced = _coerce_like(main_cfg[yaml_key], getattr(FLAGS, flag_key))
            if coerced is not None:
                setattr(FLAGS, flag_key, coerced)

    # Backward-compatibility for older config naming.
    if (
        'ogbench_dataset_dir' not in cli_flag_names
        and 'dataset_dir' in main_cfg
        and main_cfg['dataset_dir'] is not None
        and main_cfg.get('ogbench_dataset_dir') is None
    ):
        FLAGS.ogbench_dataset_dir = main_cfg['dataset_dir']

    # Action chunking settings are used both by main and by agent config.
    if 'horizon_length' in cli_flag_names:
        agent_cfg['horizon_length'] = int(FLAGS.horizon_length)
    elif 'horizon_length' not in cli_agent_keys and 'horizon_length' in main_cfg and main_cfg['horizon_length'] is not None:
        agent_cfg['horizon_length'] = int(main_cfg['horizon_length'])
    if 'action_chunking' not in cli_agent_keys and 'action_chunking' in main_cfg and main_cfg['action_chunking'] is not None:
        agent_cfg['action_chunking'] = bool(main_cfg['action_chunking'])

    # Merge nested agent section at the end so it has highest priority in YAML.
    if 'agent' in main_cfg and isinstance(main_cfg['agent'], dict):
        for key, value in main_cfg['agent'].items():
            if value is not None and key not in cli_agent_keys:
                if key in agent_cfg:
                    coerced = _coerce_like(value, agent_cfg[key])
                    if coerced is not None:
                        agent_cfg[key] = coerced
                else:
                    agent_cfg[key] = value


def _coerce_like(value, reference_value):
    """Cast YAML values to the same type as existing typed config values."""
    if reference_value is None:
        return value

    if isinstance(value, str) and value.startswith('${'):
        return None

    ref_type = type(reference_value)
    if isinstance(value, ref_type):
        return value

    try:
        if ref_type is bool:
            if isinstance(value, str):
                lowered = value.lower()
                if lowered in ('true', '1', 'yes', 'y', 'on'):
                    return True
                if lowered in ('false', '0', 'no', 'n', 'off'):
                    return False
                return None
            return bool(value)
        if ref_type is int:
            return int(value)
        if ref_type is float:
            return float(value)
        if ref_type is str:
            return str(value)
        if ref_type is tuple and isinstance(value, list):
            return tuple(value)
        if ref_type is list and isinstance(value, tuple):
            return list(value)
        if ref_type is list and not isinstance(value, list):
            return [value]
    except (TypeError, ValueError):
        return None

    return value


def _build_exp_name_tags(exp_name_flags, agent_cfg):
    tags = []
    for raw_key in exp_name_flags:
        key = raw_key.strip()
        if not key:
            continue

        if key.startswith('agent.'):
            agent_key = key.split('.', 1)[1]
            if agent_key not in agent_cfg:
                continue
            value = agent_cfg[agent_key]
            tag_key = key
        else:
            if key not in FLAGS:
                continue
            value = getattr(FLAGS, key)
            tag_key = key

        tags.append(f'{tag_key}={value}')
    return tags

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_string('wandb_project', 'qc', 'Weights & Biases project name.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_list('exp_name_flags', [], 'Additional flag keys to append to exp name (e.g., env_name,horizon_length,agent.alpha).')
flags.DEFINE_list('exp_name_tags', [], 'Literal tags to append to exp name (e.g., ablation1,trialA).')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 10000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 100, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', AGENT_CONFIG_FLAG_DEFAULT, lock_config=False)
flags.DEFINE_string('main_config', 'config/main.yaml', 'Path to a YAML file for main/agent overrides.')
flags.DEFINE_string('agent_config', None, 'Agent YAML name under config/agent (without .yaml). Overrides main_config value when set.')

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', '/home/dbsghd363/2026-spring/ogbench', 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

flags.DEFINE_float('offline_batch_ratio', 0.0,
                   'Fraction of each online training batch sampled from the offline '
                   'train_dataset (0.0 = pure replay_buffer, 0.5 = 50/50 mix like main_online.py).')

flags.DEFINE_bool('separate_offline_buffer', False,
                  'If True, keep the full offline dataset intact (train_dataset) and '
                  'accumulate online transitions in a separate replay_buffer of size '
                  'FLAGS.buffer_size (max total = train_dataset.size + buffer_size). '
                  'Critic / actor-Q batch is mixed from offline + online (when '
                  '--offline_batch_ratio is left at its default 0.0, the mix is auto-set '
                  'to the size-proportional ratio offline_size / (offline_size + online_size); '
                  'otherwise the explicit ratio is used). The actor BC term draws '
                  '(obs, action) pairs only from the offline train_dataset.')

flags.DEFINE_string('restore_path', None, 'Path to the checkpoint .pkl file to restore from.')
flags.DEFINE_integer('restore_epoch', 0, 'Epoch to resume training from.')
flags.DEFINE_bool('restore_params_only', False,
                  'If True, only network params are restored from the checkpoint; '
                  'opt_state/step are reinitialized. Required when the optimizer '
                  'config differs from the saved checkpoint (e.g. setting agent.actor_lr).')

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def main(_):
    config = FLAGS.agent
    if FLAGS.main_config is not None:
        cli_flag_names, cli_agent_keys = _collect_cli_overrides(sys.argv)
        main_cfg = _read_yaml_file(FLAGS.main_config)

        agent_cfg_name = FLAGS.agent_config if FLAGS.agent_config is not None else _extract_agent_config_name(main_cfg)
        if agent_cfg_name is not None:
            agent_yaml_path = os.path.join('config', 'agent', f'{agent_cfg_name}.yaml')
            if os.path.exists(agent_yaml_path):
                agent_yaml = _read_yaml_file(agent_yaml_path)
                for key, value in agent_yaml.items():
                    if key not in cli_agent_keys:
                        if key in config:
                            coerced = _coerce_like(value, config[key])
                            if coerced is not None:
                                config[key] = coerced
                        else:
                            config[key] = value
            else:
                raise FileNotFoundError(
                    f"Agent YAML config '{agent_yaml_path}' not found. "
                    f"Set 'agent_config' in main YAML to an existing file in config/agent/."
                )

        _apply_main_yaml_overrides(main_cfg, config, cli_flag_names, cli_agent_keys)

    exp_name_tags = _build_exp_name_tags(FLAGS.exp_name_flags, config)
    exp_name_tags.extend([tag.strip() for tag in FLAGS.exp_name_tags if tag.strip()])
    exp_name = get_exp_name(FLAGS.seed, exp_name_tags)
    # exp_name = f'{FLAGS.env_name}/{exp_name}'
    run = setup_wandb(project=FLAGS.wandb_project, group=FLAGS.run_group, name=f'{FLAGS.env_name}/{exp_name}')
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f, indent=2, sort_keys=True)
        f.write('\n')

    # data loading
    dataset_idx = 0  # Initialize dataset_idx for all cases
    dataset_paths = []  # Initialize dataset_paths for all cases

    # Check if this is a robomimic environment
    from envs.robomimic_utils import is_robomimic_env

    if is_robomimic_env(FLAGS.env_name):
        # Robomimic environment - use robomimic loader
        print(f"Loading robomimic environment: {FLAGS.env_name}")
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)
    elif FLAGS.ogbench_dataset_dir is not None:
        # OGBench with custom dataset directory
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0

        dataset_name = _ogbench_dataset_name_from_env_name(FLAGS.env_name)
        dataset_paths = [
            file
            for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/{dataset_name}*.npz"))
            if '-val.npz' not in file
        ]
        if len(dataset_paths) == 0:
            raise FileNotFoundError(
                f"No OGBench train dataset found for env '{FLAGS.env_name}' (expected prefix '{dataset_name}') "
                f"under '{FLAGS.ogbench_dataset_dir}'."
            )

        if len(dataset_paths) == 1:
            print(f"Only 1 dataset found: '{dataset_paths[0]}'. Disabling dataset replacement.")
            FLAGS.dataset_replace_interval = 0
            dataset_idx = 0
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
        # Default OGBench environment (no custom dataset dir)
        print(f"Loading default OGBench environment: {FLAGS.env_name}")
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = FLAGS.restore_epoch
    
    discount = FLAGS.discount
    # config["horizon_length"] = FLAGS.horizon_length

    # handle dataset
    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if is_robomimic_env(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds
    
    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    
    agent_class = agents[config['agent_name']]

    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    if FLAGS.restore_path is not None:
        agent = restore_agent_with_file(
            agent, FLAGS.restore_path, params_only=FLAGS.restore_params_only,
        )

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes},
        wandb_logger=wandb,
    )

    offline_init_time = time.time()
    # Offline RL
    start_offline_step = min(FLAGS.restore_epoch + 1, FLAGS.offline_steps + 1)
    for i in tqdm.tqdm(range(start_offline_step, FLAGS.offline_steps + 1)):
        log_step += 1

        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and len(dataset_paths) > 0 and i % FLAGS.dataset_replace_interval == 0:
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            # print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
            )
            train_dataset = process_train_dataset(train_dataset)

        batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)

        agent, offline_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            logger.log(offline_info, "offline_agent", step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

        # eval
        if i == FLAGS.offline_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            # during eval, the action chunk is executed fully
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                seed=FLAGS.seed + 1000,
            )
            logger.log(eval_info, "eval", step=log_step)

    if FLAGS.offline_steps > 0:
        save_agent(agent, FLAGS.save_dir, "offline_final")

    # transition from offline to online
    if hasattr(agent, 'switch_config_to_online'):
        agent = agent.switch_config_to_online()

    if FLAGS.separate_offline_buffer:
        # Keep the offline dataset (train_dataset) intact and accumulate online
        # transitions in a separate buffer of size FLAGS.buffer_size. The buffer
        # is initialised empty; transitions are added one by one in the online
        # loop. We seed it with an example transition built from the offline
        # dataset so dtypes/shapes match the live env transitions.
        example_transition = dict(
            observations=np.asarray(train_dataset['observations'][0]),
            actions=np.asarray(train_dataset['actions'][0]),
            rewards=np.asarray(train_dataset['rewards'][0]).astype(np.float32),
            terminals=np.asarray(train_dataset['terminals'][0]).astype(np.float32),
            masks=np.asarray(train_dataset['masks'][0]).astype(np.float32),
            next_observations=np.asarray(train_dataset['next_observations'][0]),
        )
        replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
    else:
        replay_buffer = ReplayBuffer.create_from_initial_dataset(
            dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
        )

    # Seed the training env once; subsequent reset() calls continue from the
    # seeded internal RNG. Offset 3000 keeps it disjoint from eval seeds (+1000 / +2000).
    ob, _ = env.reset(seed=FLAGS.seed + 3000)

    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    # Online RL
    update_info = {}

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()
    
    start_online_step = max(1, FLAGS.restore_epoch - FLAGS.offline_steps + 1)
    for i in tqdm.tqdm(range(start_online_step, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)
        
        # during online rl, the action chunk is executed fully
        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)

            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))
        
        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            # Adjust reward for D4RL antmaze.
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            # Adjust online (0, 1) reward for robomimic
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)
        
        # done
        if done:
            ob, _ = env.reset()
            action_queue = []  # reset the action queue
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            # In separate-offline-buffer mode, if the user didn't override
            # offline_batch_ratio, the critic / actor-Q batch is sampled
            # uniformly from the union (offline + online) pool — i.e. the mix
            # is proportional to the current sizes of the two buffers.
            if FLAGS.separate_offline_buffer and FLAGS.offline_batch_ratio <= 0.0:
                offline_size = train_dataset.size
                online_size = max(replay_buffer.size, 1)
                effective_offline_ratio = offline_size / float(offline_size + online_size)
            else:
                effective_offline_ratio = FLAGS.offline_batch_ratio

            n_offline = int(round(config['batch_size'] * effective_offline_ratio))
            n_online = config['batch_size'] - n_offline

            if n_offline > 0 and n_online > 0:
                dataset_batch = train_dataset.sample_sequence(
                    n_offline * FLAGS.utd_ratio,
                    sequence_length=FLAGS.horizon_length, discount=discount)
                replay_batch = replay_buffer.sample_sequence(
                    n_online * FLAGS.utd_ratio,
                    sequence_length=FLAGS.horizon_length, discount=discount)
                batch = {k: np.concatenate([
                    dataset_batch[k].reshape((FLAGS.utd_ratio, n_offline) + dataset_batch[k].shape[1:]),
                    replay_batch[k].reshape((FLAGS.utd_ratio, n_online) + replay_batch[k].shape[1:])],
                    axis=1) for k in dataset_batch}
            elif n_offline > 0:
                batch = train_dataset.sample_sequence(
                    config['batch_size'] * FLAGS.utd_ratio,
                    sequence_length=FLAGS.horizon_length, discount=discount)
                batch = jax.tree.map(lambda x: x.reshape((
                    FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)
            else:
                batch = replay_buffer.sample_sequence(
                    config['batch_size'] * FLAGS.utd_ratio,
                    sequence_length=FLAGS.horizon_length, discount=discount)
                batch = jax.tree.map(lambda x: x.reshape((
                    FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

            # When using separate offline buffer, also sample an offline-only batch
            # for the actor BC term. The agent picks these up via `bc_observations`
            # and `bc_actions`; observations / actions for critic + actor-Q stay
            # on the (mixed) `batch` above.
            if FLAGS.separate_offline_buffer:
                bc_batch = train_dataset.sample_sequence(
                    config['batch_size'] * FLAGS.utd_ratio,
                    sequence_length=FLAGS.horizon_length, discount=discount)
                batch["bc_observations"] = bc_batch["observations"].reshape(
                    (FLAGS.utd_ratio, config["batch_size"]) + bc_batch["observations"].shape[1:]
                )
                batch["bc_actions"] = bc_batch["actions"].reshape(
                    (FLAGS.utd_ratio, config["batch_size"]) + bc_batch["actions"].shape[1:]
                )

            agent, update_info["online_agent"] = agent.batch_update(batch)
            
        if i % FLAGS.log_interval == 0:
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, _ = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
                seed=FLAGS.seed + 2000,
            )
            logger.log(eval_info, "eval", step=log_step)

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, log_step)

    if FLAGS.online_steps > 0:
        save_agent(agent, FLAGS.save_dir, "online_final")

    end_time = time.time()

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                 "qpos": np.stack(data["qpos"], axis=0), 
                 "qvel": np.stack(data["qvel"], axis=0), 
                 "obs": np.stack(data["obs"], axis=0), 
                 "offline_time": online_init_time - offline_init_time,
                 "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

if __name__ == '__main__':
    app.run(main)
