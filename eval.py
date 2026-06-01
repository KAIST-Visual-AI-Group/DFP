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

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', AGENT_CONFIG_FLAG_DEFAULT, lock_config=False)
flags.DEFINE_string('main_config', 'config/eval.yaml', 'Path to a YAML file for main/agent overrides.')
flags.DEFINE_string('agent_config', None, 'Agent YAML name under config/agent (without .yaml). Overrides main_config value when set.')

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', '/home/dbsghd363/2026-spring/ogbench', 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

flags.DEFINE_string('restore_path', None, 'Path to the checkpoint .pkl file to restore from.')
flags.DEFINE_integer('restore_epoch', 0, 'Epoch to resume training from.')
flags.DEFINE_bool('use_wandb', False, 'Use WandB logging during evaluation.')

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        if self.wandb_logger is not None:
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

    if FLAGS.use_wandb:
        run = setup_wandb(project=FLAGS.wandb_project, group=FLAGS.run_group, name=exp_name)
    else:
        run = None
    
    project_name = run.project if run is not None else 'qc'
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, project_name, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f, indent=2, sort_keys=True)
        f.write('\n')

    if run is not None:
        with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
            f.write(run.url)

    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_replace_interval != 0
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = _ogbench_task_index_from_env_name(FLAGS.env_name)
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
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
    else:
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
        agent = restore_agent_with_file(agent, FLAGS.restore_path)

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv")) 
                    for prefix in prefixes},
        wandb_logger=wandb if FLAGS.use_wandb else None,
    )

    if FLAGS.restore_path is None:
        print("\nWarning: Evaluating agent WITHOUT a --restore_path specified. It will be randomly initialized!")

    print(f"\n[{FLAGS.env_name}] Starting Standalone Evaluation with seed={FLAGS.seed}...", flush=True)
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
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for k, v in eval_info.items():
        print(f"{k}: {v:.4f}")
    print("="*50)

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

if __name__ == '__main__':
    app.run(main)




# # QC
# MUJOCO_GL=egl python main.py --exp_name_tags=QC --agent_config=acfql --run_group=reproduce --agent.actor_type=best-of-n --agent.q_bon=32 --agent.eval_bon=32 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5

# # BFN-n
# MUJOCO_GL=egl python main.py --exp_name_tags=BFN-n --agent_config=acfql --run_group=reproduce --agent.actor_type=best-of-n --agent.q_bon=4 --agent.eval_bon=4 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=False

# # BFN
# MUJOCO_GL=egl python main.py --exp_name_tags=BFN --agent_config=acfql --run_group=reproduce --agent.actor_type=best-of-n --agent.q_bon=4 --agent.eval_bon=4 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=1

# # QC-FQL
# MUJOCO_GL=egl python main.py --exp_name_tags=QC-FQL --agent_config=acfql --run_group=reproduce --agent.alpha=100 --agent.q_alpha=1 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5

# # FQL-n
# MUJOCO_GL=egl python main.py --exp_name_tags=FQL-n --agent_config=acfql --run_group=reproduce --agent.alpha=100 --agent.q_alpha=1 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=False

# # FQL
# MUJOCO_GL=egl python main.py --exp_name_tags=FQL --agent_config=acfql --run_group=reproduce --agent.alpha=100 --agent.q_alpha=1 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=1


# Drifting
# MUJOCO_GL=egl python main.py --exp_name_tags=Drifting --agent_config=drift --run_group=reproduce --agent.alpha=0.1 --agent.q_alpha=1 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=1

# Drifting-QC
# MUJOCO_GL=egl python main.py --exp_name_tags=Drifting-QC --agent_config=drift --run_group=reproduce --agent.alpha=0.1 --agent.q_alpha=1 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True

# Drifting-QC-BoN
# MUJOCO_GL=egl python main.py --exp_name_tags=Drifting-QC-BoN --agent_config=drift --run_group=reproduce --agent.actor_type=best-of-n --agent.q_bon=32 --agent.eval_bon=32 --agent.alpha=1 --agent.q_alpha=0 --agent.gen_per_label=8 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True

# Drifting-BoN
# MUJOCO_GL=egl python main.py --exp_name_tags=Drifting-BoN --agent_config=drift --run_group=reproduce --agent.actor_type=best-of-n --agent.q_bon=32 --agent.eval_bon=32 --agent.alpha=1 --agent.q_alpha=1 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=1 --agent.action_chunking=False

# Drifting-QC-FQL
# MUJOCO_GL=egl python main.py --exp_name_tags=Drifting-QC-FQL --agent_config=drift --run_group=reproduce --agent.alpha=0.1 --agent.q_alpha=1 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True

# Drifting-QC-BoN-T0_2
# MUJOCO_GL=egl python main.py --exp_name_tags=Drifting-QC-BoN-T0_2 --agent_config=drift --run_group=reproduce --agent.actor_type=best-of-n --agent.q_bon=32 --agent.eval_bon=32 --agent.alpha=1 --agent.q_alpha=0 --agent.gen_per_label=8 --agent.drift_temps='[0.2]' --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True

# Drifting-QC-FQL-T0_2
# MUJOCO_GL=egl python main.py --exp_name_tags=Drifting-QC-FQL-T0_2 --agent_config=drift --run_group=reproduce --agent.alpha=0.1 --agent.q_alpha=1 --agent.gen_per_label=8 --agent.drift_temps='[0.1]' --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True



# MeanFlow
# MUJOCO_GL=egl python main.py --exp_name_tags=MeanFlow --agent_config=meanflow --run_group=reproduce --agent.alpha=1 --agent.q_alpha=0 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5

# MeanFlow-QC
# MUJOCO_GL=egl python main.py --exp_name_tags=MeanFlow-QC --agent_config=meanflow --run_group=reproduce --agent.alpha=1 --agent.q_alpha=0 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True

# MeanFlow-QC-BoN
# MUJOCO_GL=egl python main.py --exp_name_tags=MeanFlow-QC-BoN --agent_config=meanflow --run_group=reproduce --agent.actor_type=best-of-n --agent.q_bon=32 --agent.eval_bon=32 --agent.alpha=1 --agent.q_alpha=0 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True

# MeanFlow-BoN
# MUJOCO_GL=egl python main.py --exp_name_tags=MeanFlow-BoN --agent_config=meanflow --run_group=reproduce --agent.actor_type=best-of-n --agent.q_bon=32 --agent.eval_bon=32 --agent.alpha=1 --agent.q_alpha=0 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=1 --agent.action_chunking=False

# MeanFlow-QC-FQL
# MUJOCO_GL=egl python main.py --exp_name_tags=MeanFlow-QC-FQL --agent_config=meanflow --run_group=reproduce --agent.alpha=0.1 --agent.q_alpha=1 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=5 --agent.action_chunking=True

# MeanFlow-FQL
# MUJOCO_GL=egl python main.py --exp_name_tags=MeanFlow-FQL --agent_config=meanflow --run_group=reproduce --agent.alpha=100 --agent.q_alpha=1 --env_name=cube-double-play-singletask-task2-v0 --sparse=False --horizon_length=1 --agent.action_chunking=False





# antmaze-large-navigate-singletask-task3-v0
