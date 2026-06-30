import warnings

from utils.viz import log_audio_and_spectrograms

warnings.simplefilter(action="ignore", category=FutureWarning)
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import wandb
from dataloaders.dataloader_vctk import VCTKDemandDataset
from models.discriminator import MetricDiscriminator, batch_pesq
from models.generator import SEMamba
from models.linoss.linoss import LinOSS
from models.selective_lru import SelectiveLRU, SelectiveLRUMIMO
from models.loss import phase_losses
from models.s4d.s4d import S4DKernel
from models.s5.s5 import S5SSM
from models.stfts import mag_phase_istft, mag_phase_stft
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import Evaluator
from utils.util import (
    build_env,
    initialize_process_group,
    initialize_seed,
    load_ckpts,
    load_config,
    load_optimizer_states,
    log_model_info,
    print_gpu_info,
    save_checkpoint,
)

torch.backends.cudnn.benchmark = True
# Enable TF32 for float32 matmuls/convs on Ampere+ (faster, negligible quality impact).
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def validate(generator, evaluator, validation_loader, cfg, device, sw, steps, stft_params, best):
    """Run the (sharded) validation pass, log averaged metrics, and return updated bests.

    Called on every rank: each rank scores its disjoint shard of the validation set
    (``validation_loader`` is already strided per rank) one full-length utterance at
    a time, then the per-rank metric sums and counts are all-reduced so every rank
    derives the same averages. Spreading the work across ranks keeps them all busy
    instead of leaving ranks>0 idling at the next training collective while rank 0
    validates -- that idle wait is what tripped the NCCL watchdog.

    Only the original metric set (magnitude / phase / complex losses + PESQ / MR-STFT
    / UTMOS) is computed; the heavier neural / CPU metrics (DistillMOS, SI-SDR, LSD,
    NISQA, eSTOI) are left to ``evaluate.py``. Only rank 0 (the one with ``sw``) logs
    scalars / example spectrograms and prints. ``best`` is the running
    ``(pesq, pesq_step, utmos, utmos_step)`` tuple, returned updated (identically on
    every rank, since it is derived from the reduced totals).
    """
    n_fft, hop_size, win_size, compress_factor = stft_params
    sr = cfg["stft_cfg"]["sampling_rate"]
    num_viz_samples = cfg["env_setting"].get("num_viz_samples", 5)
    viz_max_seconds = cfg["env_setting"].get("viz_max_seconds", 5.0)

    # Use the unwrapped module: shards differ in length (e.g. 211/211/210 over 632),
    # so going through the DDP wrapper would fire its per-forward buffer broadcast a
    # different number of times per rank and deadlock. The only collective here is
    # the single all-reduce of the metric sums below, which every rank reaches once.
    model = generator.module if isinstance(generator, DistributedDataParallel) else generator
    model.eval()
    torch.cuda.empty_cache()

    val_metrics = dict.fromkeys(
        (
            "Magnitude Loss",
            "Phase Loss",
            "Complex Loss",
            "PESQ Score",
            "MultiResSTFT Loss",
            "UTMOS Score",
        ),
        0.0,
    )
    n_utts = 0

    with torch.no_grad():
        for j, batch in enumerate(validation_loader):
            print(f"BATCH{j}")
            print("loading data...")
            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            clean_mag = clean_mag.to(device, non_blocking=True)
            clean_pha = clean_pha.to(device, non_blocking=True)
            clean_com = clean_com.to(device, non_blocking=True)
            noisy_mag = noisy_mag.to(device, non_blocking=True)
            noisy_pha = noisy_pha.to(device, non_blocking=True)
            print("generating...")
            mag_g, pha_g, com_g = model(noisy_mag, noisy_pha)

            print("ISTFT")
            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            min_len = min(clean_audio.size(-1), audio_g.size(-1))

            print("computing metrics")
            m = evaluator.compute(
                clean_audio[..., :min_len],
                audio_g[..., :min_len],
                exclude=("nisqa", "estoi", "distillmos", "sisdr", "lsd"),
            )

            print("visualize samples...")
            # Log a handful of example utterances as waveforms / spectrograms
            # (rank 0 only -- it is the only rank with a SummaryWriter).
            if sw is not None and j < num_viz_samples:
                noisy_audio = mag_phase_istft(
                    noisy_mag, noisy_pha, n_fft, hop_size, win_size, compress_factor
                )
                log_audio_and_spectrograms(
                    sw,
                    j,
                    steps,
                    sr,
                    hop_size,
                    compress_factor,
                    clean_audio,
                    noisy_audio,
                    audio_g,
                    clean_mag,
                    noisy_mag,
                    mag_g,
                    max_seconds=viz_max_seconds,
                )

            print("compute losses")
            ip, gd, iaf = phase_losses(clean_pha, pha_g, cfg)
            val_metrics["Phase Loss"] += (ip + gd + iaf).item()
            val_metrics["Magnitude Loss"] += F.mse_loss(clean_mag, mag_g).item()
            val_metrics["Complex Loss"] += F.mse_loss(clean_com, com_g).item()
            val_metrics["PESQ Score"] += m.pesq
            val_metrics["MultiResSTFT Loss"] += m.mrstft
            val_metrics["UTMOS Score"] += m.utmos
            n_utts += 1

    # Reduce per-rank sums + counts into global totals so every rank computes the
    # same averages over the whole validation set. dict order is identical across
    # ranks (same code), so the packed tensor layout matches for the all-reduce.
    keys = list(val_metrics)
    totals = torch.tensor([val_metrics[k] for k in keys] + [float(n_utts)], device=device)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    n_total = max(totals[-1].item(), 1.0)
    averaged = {k: (totals[i] / n_total).item() for i, k in enumerate(keys)}
    best_pesq, best_pesq_step, best_utmos, best_utmos_step = best
    log_str = "VALIDATION"
    for metric in keys:
        score = averaged[metric]
        if sw is not None:
            sw.add_scalar(f"Validation/{metric}", score, steps)
        log_str += f" | {metric}: {score:.4f}"

        if metric == "PESQ Score":
            if score >= best_pesq:
                best_pesq, best_pesq_step = score, steps
            log_str += f" (max. {best_pesq:.4f} @ {best_pesq_step} steps)"
        elif metric == "UTMOS Score":
            if score >= best_utmos:
                best_utmos, best_utmos_step = score, steps
            log_str += f" (max. {best_utmos:.4f} @ {best_utmos_step} steps)"

    if sw is not None:  # rank 0 only
        print(log_str)
    model.train()
    return best_pesq, best_pesq_step, best_utmos, best_utmos_step


def create_partitioned_optimizer(
    model, base_lr=1e-3, ssm_lr_factor=0.01, betas=(0.9, 0.999), weight_decay=1e-2
):
    ssm_params = []
    rest_params = []
    ssm_param_ids = set()

    for module in model.modules():
        if isinstance(module, LinOSS):
            for attr in ["steps", "A_diag", "G_diag"]:
                param = getattr(module, attr, None)
                if param is not None and param.requires_grad:
                    ssm_params.append(param)
                    ssm_param_ids.add(id(param))
        elif isinstance(module, S5SSM):
            # S5's continuous-time dynamics (eigenvalues Lambda_re/Lambda_im and the
            # per-mode timescales log_step) are the analog of LinOSS's A_diag/G_diag/steps:
            # the original S5 trains them at a reduced "ssm_lr" with no weight decay.
            for attr in ["Lambda_re", "Lambda_im", "log_step"]:
                param = getattr(module, attr, None)
                if param is not None and param.requires_grad:
                    ssm_params.append(param)
                    ssm_param_ids.add(id(param))
        elif isinstance(module, S4DKernel):
            # S4D's continuous-time diagonal dynamics (per-mode timescales log_dt and
            # the diagonal A eigenvalues log_A_real/A_imag) are the analog of LinOSS's
            # A_diag/G_diag/steps: the original S4D trains them at a reduced lr with no
            # weight decay. The C readout and the D skip (S4DCore.D) stay at the base LR.
            for attr in ["log_dt", "log_A_real", "A_imag"]:
                param = getattr(module, attr, None)
                if param is not None and param.requires_grad:
                    ssm_params.append(param)
                    ssm_param_ids.add(id(param))

    for param in model.parameters():
        if param.requires_grad and id(param) not in ssm_param_ids:
            rest_params.append(param)

    param_groups = [
        {"params": rest_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": ssm_params, "lr": base_lr * ssm_lr_factor, "weight_decay": 0.0},
    ]

    # Apply the betas globally to the optimizer
    return optim.AdamW(param_groups, betas=betas)


def setup_optimizers(models, cfg):
    """Set up optimizers for the models."""
    generator, discriminator = models
    learning_rate = cfg["training_cfg"]["learning_rate"]
    betas = (cfg["training_cfg"]["adam_b1"], cfg["training_cfg"]["adam_b2"])

    # Extract weight decay if it exists in your config, otherwise default to 1e-2
    weight_decay = cfg["training_cfg"].get("weight_decay", 1e-2)

    optim_g = create_partitioned_optimizer(
        generator,
        base_lr=learning_rate,
        ssm_lr_factor=0.01,
        betas=betas,
        weight_decay=weight_decay,
    )
    optim_d = optim.AdamW(
        discriminator.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay
    )

    return optim_g, optim_d


def setup_schedulers(optimizers, cfg, last_epoch):
    """Set up learning rate schedulers."""
    optim_g, optim_d = optimizers
    lr_decay = cfg["training_cfg"]["lr_decay"]

    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay, last_epoch=last_epoch)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay, last_epoch=last_epoch)

    return scheduler_g, scheduler_d


def create_dataset(cfg, train=True, split=True, device="cuda:0"):
    """Create dataset based on cfguration."""
    clean_json = (
        cfg["data_cfg"]["train_clean_json"] if train else cfg["data_cfg"]["valid_clean_json"]
    )
    noisy_json = (
        cfg["data_cfg"]["train_noisy_json"] if train else cfg["data_cfg"]["valid_noisy_json"]
    )
    shuffle = (cfg["env_setting"]["num_gpus"] <= 1) if train else False
    pcs = cfg["training_cfg"]["use_PCS400"] if train else False

    # Rebase wav paths onto staged data (e.g. SLURM $TMPDIR) when requested. The
    # DATA_ROOT env var wins over the optional data_cfg.data_root config key, so a
    # job script can point training at node-local storage without editing recipes.
    data_root = os.environ.get("DATA_ROOT") or cfg["data_cfg"].get("data_root")
    orig_data_root = os.environ.get("DATA_ROOT_ORIG") or cfg["data_cfg"].get("orig_data_root")

    return VCTKDemandDataset(
        clean_json=clean_json,
        noisy_json=noisy_json,
        data_root=data_root,
        orig_data_root=orig_data_root,
        sampling_rate=cfg["stft_cfg"]["sampling_rate"],
        segment_size=cfg["training_cfg"]["segment_size"],
        n_fft=cfg["stft_cfg"]["n_fft"],
        hop_size=cfg["stft_cfg"]["hop_size"],
        win_size=cfg["stft_cfg"]["win_size"],
        compress_factor=cfg["model_cfg"]["compress_factor"],
        split=split,
        n_cache_reuse=0,
        shuffle=shuffle,
        device=device,
        pcs=pcs,
    )


def create_dataloader(dataset, cfg, train=True, rank=0, world_size=1):
    """Create dataloader based on dataset and configuration."""
    if not train:
        # Shard the (full-length) validation set across ranks: this rank scores the
        # strided slice [rank, rank+world_size, ...], one utterance per step
        # (batch_size=1, the dataset is built with split=False). A plain index list
        # is used rather than DistributedSampler because the latter pads short shards
        # with duplicate samples, which would bias the averaged validation metrics;
        # the strided list is exact and lets workers load only this rank's utterances.
        shard = list(range(rank, len(dataset), world_size))
        return DataLoader(
            dataset,
            num_workers=1,
            shuffle=False,
            sampler=shard,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
        )

    if cfg["env_setting"]["num_gpus"] > 1:
        sampler = DistributedSampler(dataset)
        # set_epoch is called per-epoch in the training loop so each epoch
        # reshuffles; setting it once here would freeze the shuffle order.
        batch_size = cfg["training_cfg"]["batch_size"] // cfg["env_setting"]["num_gpus"]
    else:
        sampler = None
        batch_size = cfg["training_cfg"]["batch_size"]
    num_workers = cfg["env_setting"]["num_workers"]

    return DataLoader(
        dataset,
        num_workers=num_workers,
        persistent_workers=True,
        shuffle=(sampler is None),
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
    )


def train(rank, args, cfg):
    num_gpus = cfg["env_setting"]["num_gpus"]
    n_fft, hop_size, win_size = (
        cfg["stft_cfg"]["n_fft"],
        cfg["stft_cfg"]["hop_size"],
        cfg["stft_cfg"]["win_size"],
    )
    compress_factor = cfg["model_cfg"]["compress_factor"]
    batch_size = cfg["training_cfg"]["batch_size"] // cfg["env_setting"]["num_gpus"]
    if num_gpus >= 1:
        initialize_process_group(cfg, rank)
        device = torch.device("cuda:{:d}".format(rank))
        torch.cuda.set_device(device)
    else:
        raise RuntimeError("Mamba needs GPU acceleration")

    generator = SEMamba(cfg).to(device)
    discriminator = MetricDiscriminator().to(device)

    # Every rank validates a shard of the val set (see create_dataloader), so every
    # rank needs an evaluator. Validation scores one utterance at a time, so PESQ is
    # fed batch_size=1: keep n_processes=1, since forking a multiprocessing pool per
    # single utterance is ~60x slower than the sequential path (see Evaluator).
    evaluator = Evaluator(sr=cfg["stft_cfg"]["sampling_rate"], pesq_n_processes=1).to(device)
    if rank == 0:
        log_model_info(rank, generator, args.exp_path)

    state_dict_g, state_dict_do, steps, last_epoch = load_ckpts(args, device)
    if state_dict_g is not None:
        generator.load_state_dict(state_dict_g["generator"], strict=False)
        discriminator.load_state_dict(state_dict_do["discriminator"], strict=False)

    if num_gpus > 1 and torch.cuda.is_available():
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    if cfg["training_cfg"].get("use_pretrainedD", False):
        discriminator.load_state_dict(torch.load("ckpts/pretrained_discriminator.pth"))
        print("Loaded pretrained weight from ckpts/pretrained_discriminator.pth.")

    # Create optimizer and schedulers
    optimizers = setup_optimizers((generator, discriminator), cfg)
    load_optimizer_states(optimizers, state_dict_do)
    optim_g, optim_d = optimizers
    scheduler_g, scheduler_d = setup_schedulers(optimizers, cfg, last_epoch)

    # Create trainset and train_loader
    trainset = create_dataset(cfg, train=True, split=True, device=device)
    train_loader = create_dataloader(trainset, cfg, train=True)

    # Create the validation set on every rank and shard it across ranks: each rank
    # scores a disjoint slice of utterances (see create_dataloader), so all ranks
    # stay busy during validation instead of ranks>0 idling at the next training
    # collective while rank 0 validates -- that idle wait is what tripped the NCCL
    # watchdog. The per-rank metric sums are all-reduced in validate().
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    validset = create_dataset(cfg, train=False, split=False, device=device)
    validation_loader = create_dataloader(
        validset, cfg, train=False, rank=rank, world_size=world_size
    )
    # Only rank 0 owns the TensorBoard / wandb writer.
    sw = None
    if rank == 0:
        wandb.init(
            project="SEMambaXLinOSS",
            name=args.exp_name,
            dir=args.exp_path,
            config=cfg,
            sync_tensorboard=True,
        )
        sw = SummaryWriter(os.path.join(args.exp_path, "logs"))

    generator.train()
    discriminator.train()

    best_pesq, best_pesq_step = 0.0, 0
    best_utmos, best_utmos_step = 0.0, 0
    for epoch in range(max(0, last_epoch), cfg["training_cfg"]["training_epochs"]):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        # Reshuffle each epoch and keep the per-rank split in sync across ranks.
        if train_loader.sampler is not None and isinstance(
            train_loader.sampler, DistributedSampler
        ):
            train_loader.sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = (
                batch  # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
            )
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
            clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
            clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))
            noisy_mag = torch.autograd.Variable(noisy_mag.to(device, non_blocking=True))
            noisy_pha = torch.autograd.Variable(noisy_pha.to(device, non_blocking=True))
            one_labels = torch.ones(batch_size).to(device, non_blocking=True)

            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            audio_list_r, audio_list_g = (
                list(clean_audio.cpu().numpy()),
                list(audio_g.detach().cpu().numpy()),
            )
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g, cfg)

            # Discriminator
            # ------------------------------------------------------- #
            optim_d.zero_grad()
            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_g.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())

            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
            else:
                loss_disc_g = 0

            loss_disc_all = loss_disc_r + loss_disc_g

            loss_disc_all.backward()
            optim_d.step()
            # ------------------------------------------------------- #

            # Generator
            # ------------------------------------------------------- #
            optim_g.zero_grad()

            # Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py
            # L2 Magnitude Loss
            loss_mag = F.mse_loss(clean_mag, mag_g)
            # Anti-wrapping Phase Loss
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, cfg)
            loss_pha = loss_ip + loss_gd + loss_iaf
            # L2 Complex Loss
            loss_com = F.mse_loss(clean_com, com_g) * 2
            # Time Loss
            loss_time = F.l1_loss(clean_audio, audio_g)
            # Metric Loss
            metric_g = discriminator(clean_mag, mag_g)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
            # Consistancy Loss
            _, _, rec_com = mag_phase_stft(
                audio_g, n_fft, hop_size, win_size, compress_factor, addeps=True
            )
            loss_con = F.mse_loss(com_g, rec_com) * 2

            loss_gen_all = (
                loss_metric * cfg["training_cfg"]["loss"]["metric"]
                + loss_mag * cfg["training_cfg"]["loss"]["magnitude"]
                + loss_pha * cfg["training_cfg"]["loss"]["phase"]
                + loss_com * cfg["training_cfg"]["loss"]["complex"]
                + loss_time * cfg["training_cfg"]["loss"]["time"]
                + loss_con * cfg["training_cfg"]["loss"]["consistancy"]
            )

            loss_gen_all.backward()
            optim_g.step()
            # ------------------------------------------------------- #

            if rank == 0:
                # STDOUT logging
                if steps % cfg["env_setting"]["stdout_interval"] == 0:
                    with torch.no_grad():
                        metric_error = F.mse_loss(metric_g.flatten(), one_labels).item()
                        mag_error = F.mse_loss(clean_mag, mag_g).item()
                        pha_error = (loss_ip + loss_gd + loss_iaf).item()
                        com_error = F.mse_loss(clean_com, com_g).item()
                        time_error = F.l1_loss(clean_audio, audio_g).item()
                        con_error = F.mse_loss(com_g, rec_com).item()

                        print(
                            "Steps : {:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, Metric Loss: {:4.3f}, "
                            "Mag Loss: {:4.3f}, Pha Loss: {:4.3f}, Com Loss: {:4.3f}, Time Loss: {:4.3f}, Cons Loss: {:4.3f}, s/b : {:4.3f}".format(
                                steps,
                                loss_gen_all,
                                loss_disc_all,
                                metric_error,
                                mag_error,
                                pha_error,
                                com_error,
                                time_error,
                                con_error,
                                time.time() - start_b,
                            )
                        )

                # Checkpointing
                if steps % cfg["env_setting"]["checkpoint_interval"] == 0 and steps != 0:
                    exp_name = f"{args.exp_path}/g_{steps:08d}.pth"
                    save_checkpoint(
                        exp_name,
                        {
                            "generator": (
                                generator.module if num_gpus > 1 else generator
                            ).state_dict()
                        },
                    )
                    exp_name = f"{args.exp_path}/do_{steps:08d}.pth"
                    save_checkpoint(
                        exp_name,
                        {
                            "discriminator": (
                                discriminator.module if num_gpus > 1 else discriminator
                            ).state_dict(),
                            "optim_g": optim_g.state_dict(),
                            "optim_d": optim_d.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )

                # Tensorboard summary logging
                if steps % cfg["env_setting"]["summary_interval"] == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all, steps)
                    sw.add_scalar("Training/Discriminator Loss", loss_disc_all, steps)
                    sw.add_scalar("Training/Metric Loss", metric_error, steps)
                    sw.add_scalar("Training/Magnitude Loss", mag_error, steps)
                    sw.add_scalar("Training/Phase Loss", pha_error, steps)
                    sw.add_scalar("Training/Complex Loss", com_error, steps)
                    sw.add_scalar("Training/Time Loss", time_error, steps)
                    sw.add_scalar("Training/Consistancy Loss", con_error, steps)

                # If NaN happend in training period, RaiseError
                if torch.isnan(loss_gen_all).any():
                    raise ValueError("NaN values found in loss_gen_all")

            # Validation runs on EVERY rank (each scores a shard and the metric sums
            # are all-reduced), so it sits outside the rank-0 block -- all ranks must
            # hit the same collectives in lockstep. steps advances identically on
            # every rank (DistributedSampler + drop_last), so this fires in sync.
            if steps % cfg["env_setting"]["validation_interval"] == 0 and steps != 0:
                best_pesq, best_pesq_step, best_utmos, best_utmos_step = validate(
                    generator,
                    evaluator,
                    validation_loader,
                    cfg,
                    device,
                    sw,
                    steps,
                    (n_fft, hop_size, win_size, compress_factor),
                    (best_pesq, best_pesq_step, best_utmos, best_utmos_step),
                )

            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print(
                "Time taken for epoch {} is {} sec\n".format(epoch + 1, int(time.time() - start))
            )

    if rank == 0:
        sw.close()
        wandb.finish()


# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_folder", default="exp")
    parser.add_argument("--exp_name", default="MambOSS_MBank_EARS")
    parser.add_argument(
        "--config", default="/data5/baur/SEMambaXLinOSS/recipes/selective/MambOSS.yaml"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["env_setting"]["seed"]
    num_gpus = cfg["env_setting"]["num_gpus"]
    available_gpus = torch.cuda.device_count()

    if num_gpus > available_gpus:
        warnings.warn(
            f"Warning: The actual number of available GPUs ({available_gpus}) is less than the .yaml config ({num_gpus}). Auto reset to num_gpu = {available_gpus}",
            UserWarning,
        )
        cfg["env_setting"]["num_gpus"] = available_gpus
        num_gpus = available_gpus
        time.sleep(5)

    initialize_seed(seed)
    args.exp_path = os.path.join(args.exp_folder, args.exp_name)
    build_env(args.config, "config.yaml", args.exp_path)

    if torch.cuda.is_available():
        print_gpu_info(cfg)
    else:
        print("CUDA is not available.")

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(args, cfg))
    else:
        train(0, args, cfg)


if __name__ == "__main__":
    main()
