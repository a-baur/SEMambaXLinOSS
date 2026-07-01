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
from functools import partial
from tqdm import tqdm

from dataloaders.dataloader_vctk import VCTKDemandDataset, crop_collate_valid
from models.discriminator import MetricDiscriminator, batch_pesq
from models.generator import SEMamba
from models.linoss.linoss import LinOSS
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
    # different number of times per rank and deadlock.
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

    # Progress bar on rank 0 only (the rank that owns the SummaryWriter); other ranks
    # score their shard silently so the terminal isn't spammed with one bar per GPU.
    pbar = tqdm(
        validation_loader,
        desc=f"Validation @ {steps} steps",
        unit="batch",
        disable=False,  # sw is None,
        dynamic_ncols=True,
        leave=False,
    )

    with torch.no_grad():
        for batch in pbar:
            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = (
                t.to(device, non_blocking=True) for t in batch
            )
            b = clean_audio.size(0)

            mag_g, pha_g, com_g = model(noisy_mag, noisy_pha)
            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            min_len = min(clean_audio.size(-1), audio_g.size(-1))

            # Batched PESQ (CPU pool across the batch) and UTMOS (GPU) run concurrently.
            pesq, mrstft, utmos = evaluator.compute_val(
                clean_audio[..., :min_len], audio_g[..., :min_len]
            )

            # Log a handful of example utterances as waveforms / spectrograms
            if sw is not None and n_utts < num_viz_samples:
                noisy_audio = mag_phase_istft(
                    noisy_mag, noisy_pha, n_fft, hop_size, win_size, compress_factor
                )
                for i in range(min(b, num_viz_samples - n_utts)):
                    log_audio_and_spectrograms(
                        sw,
                        n_utts + i,
                        steps,
                        sr,
                        hop_size,
                        compress_factor,
                        clean_audio[i : i + 1],
                        noisy_audio[i : i + 1],
                        audio_g[i : i + 1],
                        clean_mag[i : i + 1],
                        noisy_mag[i : i + 1],
                        mag_g[i : i + 1],
                        max_seconds=viz_max_seconds,
                    )

            ip, gd, iaf = phase_losses(clean_pha, pha_g, cfg)
            val_metrics["Phase Loss"] += (ip + gd + iaf).item() * b
            val_metrics["Magnitude Loss"] += F.mse_loss(clean_mag, mag_g).item() * b
            val_metrics["Complex Loss"] += F.mse_loss(clean_com, com_g).item() * b
            val_metrics["PESQ Score"] += pesq * b
            val_metrics["MultiResSTFT Loss"] += mrstft * b
            val_metrics["UTMOS Score"] += utmos * b
            n_utts += b

            # Live running means on the rank-0 bar (shard-local until the all-reduce).
            if sw is not None:
                pbar.set_postfix(
                    utts=n_utts,
                    pesq=val_metrics["PESQ Score"] / n_utts,
                    utmos=val_metrics["UTMOS Score"] / n_utts,
                )
    pbar.close()

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
        n = len(dataset)
        max_valid = cfg["env_setting"].get("max_valid_samples")
        if max_valid is not None and max_valid > 0:
            n = min(n, int(max_valid))

        shard = list(range(rank, n, world_size))
        val_batch_size = cfg["training_cfg"].get("val_batch_size", 1)
        val_num_workers = cfg["env_setting"].get("val_num_workers", 1)
        crop_seconds = cfg["training_cfg"].get("valid_crop_seconds", 4.0)
        sr = cfg["stft_cfg"]["sampling_rate"]
        hop = cfg["stft_cfg"]["hop_size"]
        crop_samples = int(round(crop_seconds * sr))
        crop_frames = crop_samples // hop + 1  # center=True STFT frame count for that window
        collate = partial(
            crop_collate_valid, crop_samples=crop_samples, crop_frames=crop_frames
        )
        return DataLoader(
            dataset,
            num_workers=val_num_workers,
            shuffle=False,
            sampler=shard,
            batch_size=val_batch_size,
            collate_fn=collate,
            pin_memory=True,
            drop_last=False,
            persistent_workers=val_num_workers > 0,
            prefetch_factor=cfg["env_setting"].get("val_prefetch_factor", 2)
            if val_num_workers > 0
            else None,
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

    val_batch_size = cfg["training_cfg"].get("val_batch_size", 1)
    pesq_n_processes = max(1, min(val_batch_size, os.cpu_count() or 1))
    evaluator = Evaluator(
        sr=cfg["stft_cfg"]["sampling_rate"], pesq_n_processes=pesq_n_processes
    ).to(device)
    if rank == 0:
        log_model_info(rank, generator, args.exp_path)

    state_dict_g, state_dict_do, steps, last_epoch = load_ckpts(args, device)
    if state_dict_g is not None:
        generator.load_state_dict(state_dict_g["generator"], strict=False)
        discriminator.load_state_dict(state_dict_do["discriminator"], strict=False)

    if num_gpus > 1 and torch.cuda.is_available():
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    def unwrap(m):
        # DDP hides the real module under `.module`; no-op when single-GPU / unwrapped.
        return m.module if isinstance(m, DistributedDataParallel) else m

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

    # Create the validation set on every rank and shard it across ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    validset = create_dataset(cfg, train=False, split=False, device=device)
    validation_loader = create_dataloader(
        validset, cfg, train=False, rank=rank, world_size=world_size
    )
    # Only rank 0 owns the TensorBoard / wandb writer.
    sw = None
    if rank == 0:
        wandb.init(
            project="SEMambaBackbones",
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
            # [B, F, T]; F = n_fft//2 + 1, T = n_frames
            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = (
                t.to(device, non_blocking=True) for t in batch
            )
            one_labels = torch.ones(batch_size, device=device)

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
            # Unweighted base losses (weights + the complex/consistency x2 applied below).
            loss_mag = F.mse_loss(clean_mag, mag_g)  # L2 magnitude
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, cfg)
            loss_pha = loss_ip + loss_gd + loss_iaf  # anti-wrapping phase
            loss_com = F.mse_loss(clean_com, com_g)  # L2 complex
            loss_time = F.l1_loss(clean_audio, audio_g)  # time-domain L1
            metric_g = discriminator(clean_mag, mag_g)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)  # PESQ-metric
            _, _, rec_com = mag_phase_stft(
                audio_g, n_fft, hop_size, win_size, compress_factor, addeps=True
            )
            loss_con = F.mse_loss(com_g, rec_com)  # STFT-consistency

            loss_w = cfg["training_cfg"]["loss"]
            loss_gen_all = (
                loss_metric * loss_w["metric"]
                + loss_mag * loss_w["magnitude"]
                + loss_pha * loss_w["phase"]
                + loss_com * 2 * loss_w["complex"]
                + loss_time * loss_w["time"]
                + loss_con * 2 * loss_w["consistancy"]
            )

            loss_gen_all.backward()
            optim_g.step()
            # ------------------------------------------------------- #

            if rank == 0:
                # STDOUT logging. Reuse the losses already computed for backprop
                # (unweighted; .item() detaches and forces the one sync we want here).
                if steps % cfg["env_setting"]["stdout_interval"] == 0:
                    metric_error = loss_metric.item()
                    mag_error = loss_mag.item()
                    pha_error = loss_pha.item()
                    com_error = loss_com.item()
                    time_error = loss_time.item()
                    con_error = loss_con.item()

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
                    save_checkpoint(
                        f"{args.exp_path}/g_{steps:08d}.pth",
                        {"generator": unwrap(generator).state_dict()},
                    )
                    save_checkpoint(
                        f"{args.exp_path}/do_{steps:08d}.pth",
                        {
                            "discriminator": unwrap(discriminator).state_dict(),
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
