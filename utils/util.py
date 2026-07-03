import yaml
import torch
import os
import glob
from datetime import timedelta
from torch.distributed import init_process_group

def get_cuda_devices() -> list[str]:
    """Get list of cuda devices available for training."""
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    n_gpus = torch.cuda.device_count()
    if cuda_visible_devices is not None:
        gpu_ids = list(map(int, cuda_visible_devices.split(",")))
    else:
        gpu_ids = list(range(n_gpus))

    device_info = []
    for idx, i in zip(gpu_ids, range(n_gpus)):
        name = torch.cuda.get_device_name(i)
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mem_free, mem_total = mem_free / 1024 ** 3, mem_total / 1024 ** 3
        mem_usage = mem_total - mem_free
        percent = mem_usage / mem_total
        info = f"{name} [gpu:{idx} | cuda:{i} | utilization: {percent:7.2%} ({mem_usage:4.1f}GB/{mem_total:4.1f}GB)]"
        device_info.append(info)

    return device_info

def print_gpu_info(cfg):
    n_gpus = torch.cuda.device_count()

    if n_gpus > 0:
        devices = get_cuda_devices()
        devices = "\n".join(devices)
        print(f"Starting training on\n{devices}\n")
        print("Batch size per GPU:", int(cfg["training_cfg"]["batch_size"] / n_gpus))

def _deep_merge(base, override):
    """Recursively merge ``override`` into ``base`` (override wins); returns base."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path):
    """Load a YAML config, resolving an optional ``base:`` parent by deep-merge.

    A recipe may set ``base: <relative/path.yaml>`` (resolved relative to the
    recipe) to inherit a shared config; the recipe's own keys override the base,
    dicts merged recursively. Nesting is supported (a base may declare its own
    base). Configs without a ``base:`` key load exactly as before.
    """
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    base_ref = cfg.pop("base", None)
    if base_ref is not None:
        base_path = os.path.join(os.path.dirname(config_path), base_ref)
        return _deep_merge(load_config(base_path), cfg)
    return cfg

def initialize_seed(seed):
    """Initialize the random seed for both CPU and GPU."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def initialize_process_group(cfg, rank):
    """Initialize the process group for distributed training.

    The collective timeout is bumped well above the NCCL default (10 min) because
    validation runs on rank 0 only: while it scores the full val set, the other
    ranks block at the next training collective, and a long pass on a slow machine
    would otherwise trip the watchdog. Override via
    ``env_setting.dist_cfg.timeout_seconds``.
    """
    timeout_s = cfg['env_setting']['dist_cfg'].get('timeout_seconds', 3600)
    init_process_group(
        backend=cfg['env_setting']['dist_cfg']['dist_backend'],
        init_method=cfg['env_setting']['dist_cfg']['dist_url'],
        world_size=cfg['env_setting']['dist_cfg']['world_size'] * cfg['env_setting']['num_gpus'],
        rank=rank,
        timeout=timedelta(seconds=timeout_s),
    )

def log_model_info(rank, model, exp_path):
    """Log model information and create necessary directories."""
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print("Generator Parameters :", num_params)
    os.makedirs(exp_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'logs'), exist_ok=True)
    print("checkpoints directory :", exp_path)

def load_ckpts(args, device):
    """Load checkpoints if available."""
    if os.path.isdir(args.exp_path):
        cp_g = scan_checkpoint(args.exp_path, 'g_')
        cp_do = scan_checkpoint(args.exp_path, 'do_')
        if cp_g is None or cp_do is None:
            return None, None, 0, -1
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        return state_dict_g, state_dict_do, state_dict_do['steps'] + 1, state_dict_do['epoch']
    return None, None, 0, -1

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????' + '.pth')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def build_env(config, config_name, exp_path):
    """Persist the fully-resolved config next to the run.

    Dumps the merged (base + overrides) config rather than copying the source
    file, so each run's ``config.yaml`` is a self-contained snapshot that
    ``evaluate.py`` can read back without the repo's shared ``base.yaml``.
    """
    os.makedirs(exp_path, exist_ok=True)
    t_path = os.path.join(exp_path, config_name)
    resolved = load_config(config)
    with open(t_path, 'w') as file:
        yaml.safe_dump(resolved, file, sort_keys=False)

def load_optimizer_states(optimizers, state_dict_do):
    """Load optimizer states from checkpoint."""
    if state_dict_do is not None:
        optim_g, optim_d = optimizers
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
