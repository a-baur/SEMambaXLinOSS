import warnings

from utils.viz import log_audio_and_spectrograms

warnings.simplefilter(action="ignore", category=FutureWarning)
import argparse
import os
import time

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import wandb
from dataloaders.dataloader_vctk import VCTKDemandDataset
from models.discriminator import MetricDiscriminator, batch_pesq
from models.generator import SEMamba
from models.linoss.linoss import LinOSS
from models.linoss.mamboss6 import MambOSS6
from models.linoss.selective_linoss import MambOSS
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


def _fixed_window(audio, lens, size):
    """Crop/zero-pad ``(B, T)`` waveforms to exactly ``size`` samples.

    Every sample past each utterance's true ``lens`` is zeroed, so the dataloader's
    right-padding -- and any model output produced over it -- never leaks into a
    metric. Utterances longer than ``size`` are cropped to their first ``size``
    samples. The result is a homogeneous batch every metric can score in one call.
    """
    T = audio.size(-1)
    audio = audio[:, :size] if T >= size else F.pad(audio, (0, size - T))
    keep = torch.arange(size, device=audio.device)[None, :] < lens.clamp(max=size)[:, None]
    return audio * keep


def _fixed_frames(spec, lens, n_frames):
    """Crop/zero-pad spectral features to ``n_frames`` frames (see ``_fixed_window``).

    Handles both ``(B, F, T)`` magnitude/phase and ``(B, F, T, 2)`` complex tensors;
    ``lens`` is the true per-utterance frame count.
    """
    T = spec.size(2)
    if T < n_frames:
        pad = (0, 0, 0, n_frames - T) if spec.dim() == 4 else (0, n_frames - T)
        spec = F.pad(spec, pad)
    else:
        spec = spec[:, :, :n_frames]
    keep = torch.arange(n_frames, device=spec.device)[None, :] < lens.clamp(max=n_frames)[:, None]
    keep = keep[:, None, :, None] if spec.dim() == 4 else keep[:, None, :]
    return spec * keep


def validate(generator, evaluator, validation_loader, cfg, device, sw, steps, stft_params, best):
    """Run the rank-0 validation pass, log averaged metrics, and return updated bests.

    Every utterance is scored on a fixed ``segment_size`` window (cropped if longer,
    zero-padded if shorter) so all metrics -- including PESQ, which parallelizes
    across the batch via ``n_processes`` -- run as a single batched call instead of
    a per-utterance Python loop. ``best`` is the running
    ``(pesq, pesq_step, utmos, utmos_step)`` tuple, returned updated.
    """
    n_fft, hop_size, win_size, compress_factor = stft_params
    sr = cfg["stft_cfg"]["sampling_rate"]
    seg = cfg["training_cfg"]["segment_size"]
    n_frames = seg // hop_size + 1
    num_viz_samples = cfg["env_setting"].get("num_viz_samples", 5)
    viz_max_seconds = cfg["env_setting"].get("viz_max_seconds", 5.0)

    generator.eval()
    torch.cuda.empty_cache()

    # GPU metrics accumulate as 0-dim tensors so we sync (.item()) only once, at the end.
    val_sums = {
        k: torch.zeros((), device=device)
        for k in (
            "Magnitude Loss",
            "Phase Loss",
            "Complex Loss",
            "MultiResSTFT Loss",
            "UTMOS Score",
            "DistillMOS Score",
            "SI-SDR",
            "LSD",
        )
    }
    # PESQ is the only CPU metric; collect its (fixed-length) windows on the GPU
    # and score them in a single batched call after the loop, so the per-batch
    # path never touches the host.
    pesq_clean, pesq_pred = [], []
    n_utts, viz_done = 0, 0

    with torch.no_grad():
        for batch in validation_loader:
            (
                clean_audio,
                clean_mag,
                clean_pha,
                clean_com,
                noisy_mag,
                noisy_pha,
                frame_lens,
                audio_lens,
            ) = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            clean_mag = clean_mag.to(device, non_blocking=True)
            clean_pha = clean_pha.to(device, non_blocking=True)
            clean_com = clean_com.to(device, non_blocking=True)
            noisy_mag = noisy_mag.to(device, non_blocking=True)
            noisy_pha = noisy_pha.to(device, non_blocking=True)

            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)
            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)

            B = clean_audio.size(0)
            alen = torch.as_tensor(audio_lens, device=device)
            flen = torch.as_tensor(frame_lens, device=device)

            # Fixed-size windows make every utterance equal length so the metrics
            # (and batched PESQ) run as a single call. Right-padding is masked out.
            cw, gw = _fixed_window(clean_audio, alen, seg), _fixed_window(audio_g, alen, seg)
            cm, mg = _fixed_frames(clean_mag, flen, n_frames), _fixed_frames(mag_g, flen, n_frames)
            cp, pg = _fixed_frames(clean_pha, flen, n_frames), _fixed_frames(pha_g, flen, n_frames)
            cc, cg = _fixed_frames(clean_com, flen, n_frames), _fixed_frames(com_g, flen, n_frames)

            ip, gd, iaf = phase_losses(cp, pg, cfg)
            m = evaluator.compute(cw, gw, exclude=("nisqa", "estoi", "pesq"), as_tensor=True)
            pesq_clean.append(cw)
            pesq_pred.append(gw)

            # Sum over batches (weighted by batch size) and average once at the end.
            val_sums["Phase Loss"] += (ip + gd + iaf) * B
            val_sums["Magnitude Loss"] += F.mse_loss(cm, mg) * B
            val_sums["Complex Loss"] += F.mse_loss(cc, cg) * B
            val_sums["MultiResSTFT Loss"] += m.mrstft * B
            val_sums["UTMOS Score"] += m.utmos * B
            val_sums["DistillMOS Score"] += m.distillmos * B
            val_sums["SI-SDR"] += m.sisdr * B
            val_sums["LSD"] += m.lsd * B
            n_utts += B

            # Log a handful of example waveforms / spectrograms at full length.
            for b in range(B):
                if viz_done >= num_viz_samples:
                    break
                fl, al = frame_lens[b], audio_lens[b]
                noisy_audio = mag_phase_istft(
                    noisy_mag[b : b + 1, :, :fl],
                    noisy_pha[b : b + 1, :, :fl],
                    n_fft,
                    hop_size,
                    win_size,
                    compress_factor,
                )
                log_audio_and_spectrograms(
                    sw,
                    viz_done,
                    steps,
                    sr,
                    hop_size,
                    compress_factor,
                    clean_audio[b : b + 1, :al],
                    noisy_audio,
                    audio_g[b : b + 1, :al],
                    clean_mag[b : b + 1, :, :fl],
                    noisy_mag[b : b + 1, :, :fl],
                    mag_g[b : b + 1, :, :fl],
                    max_seconds=viz_max_seconds,
                )
                viz_done += 1

    # One CPU round-trip + one n_processes pool for all of PESQ
    pesq_score = (
        evaluator.compute(
            torch.cat(pesq_clean),
            torch.cat(pesq_pred),
            exclude=("nisqa", "estoi", "mrstft", "utmos", "distillmos", "sisdr", "lsd"),
        ).pesq
        if pesq_pred
        else 0.0
    )

    # Average every GPU metric with a single host<->device sync (one .tolist()).
    keys = list(val_sums)
    averaged = dict(zip(keys, (torch.stack([val_sums[k] for k in keys]) / max(n_utts, 1)).tolist()))

    # PESQ logged after Complex Loss to preserve the original ordering.
    val_metrics = {
        "Magnitude Loss": averaged["Magnitude Loss"],
        "Phase Loss": averaged["Phase Loss"],
        "Complex Loss": averaged["Complex Loss"],
        "PESQ Score": pesq_score,
        "MultiResSTFT Loss": averaged["MultiResSTFT Loss"],
        "UTMOS Score": averaged["UTMOS Score"],
        "DistillMOS Score": averaged["DistillMOS Score"],
        "SI-SDR": averaged["SI-SDR"],
        "LSD": averaged["LSD"],
    }

    best_pesq, best_pesq_step, best_utmos, best_utmos_step = best
    log_str = "VALIDATION"
    for metric, score in val_metrics.items():
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

    print(log_str)
    generator.train()
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
        elif isinstance(module, MambOSS):
            # Only the baseline-bank biases (c_nu, c_theta) — the S-LinOSS analog of the
            # LTI dynamics scalars steps/A_diag/G_diag — get the low SSM LR. The selective
            # weight matrices W_nu/W_theta *are* the input-dependence; they start tiny and
            # must train at the base LR to learn, so they fall through to rest_params.
            for proj in [module.nu_proj, module.theta_proj]:
                if proj.bias is not None and proj.bias.requires_grad:
                    ssm_params.append(proj.bias)
                    ssm_param_ids.add(id(proj.bias))
        elif isinstance(module, MambOSS6):
            # Static per-(channel, mode) dynamics A_log/omega and the two time-step
            # baselines (dt_*_up.bias) are the LTI analog of LinOSS's A_diag/G_diag/steps
            # — low SSM LR, no weight decay. The low-rank selective projection weights
            # *are* the input-dependence and stay at the base LR (like MambOSS's W_nu/W_theta).
            for attr in ["A_log", "omega"]:
                param = getattr(module, attr, None)
                if param is not None and param.requires_grad:
                    ssm_params.append(param)
                    ssm_param_ids.add(id(param))
            for proj in [module.dt_nu_up, module.dt_theta_up]:
                if proj.bias is not None and proj.bias.requires_grad:
                    ssm_params.append(proj.bias)
                    ssm_param_ids.add(id(proj.bias))
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

    return VCTKDemandDataset(
        clean_json=clean_json,
        noisy_json=noisy_json,
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


def collate_pad(batch):
    """Pad a list of variable-length validation utterances into a batch.

    __getitem__ (split=False) returns squeezed, full-length tensors:
        clean_audio [T], mag/pha [F, Tf], com [F, Tf, 2], noisy mag/pha [F, Tf].
    We right-pad audio to the batch's max sample count and the spectral tensors
    to the max frame count, and return the true per-item lengths so the
    validation loop can crop back before computing any metric (padding must not
    leak into the scores).
    """
    audio_lens = [b[0].size(-1) for b in batch]
    frame_lens = [b[1].size(-1) for b in batch]
    t_max, f_max = max(audio_lens), max(frame_lens)

    pad_time = lambda x: F.pad(x, (0, f_max - x.size(-1)))  # [F, Tf] -> [F, f_max]
    pad_com = lambda x: F.pad(x, (0, 0, 0, f_max - x.size(-2)))  # [F, Tf, 2] (pad Tf)

    clean_audio = torch.stack([F.pad(b[0], (0, t_max - b[0].size(-1))) for b in batch])
    clean_mag = torch.stack([pad_time(b[1]) for b in batch])
    clean_pha = torch.stack([pad_time(b[2]) for b in batch])
    clean_com = torch.stack([pad_com(b[3]) for b in batch])
    noisy_mag = torch.stack([pad_time(b[4]) for b in batch])
    noisy_pha = torch.stack([pad_time(b[5]) for b in batch])
    return (
        clean_audio,
        clean_mag,
        clean_pha,
        clean_com,
        noisy_mag,
        noisy_pha,
        frame_lens,
        audio_lens,
    )


def create_dataloader(dataset, cfg, train=True):
    """Create dataloader based on dataset and configuration."""
    if not train:
        # Validation runs on rank 0 only, so no DistributedSampler (that would
        # silently shard the val set across ranks that never run). Batch the
        # utterances (padded) to keep the GPU busy -- at batch_size=1 the
        # per-op launch/sync latency dominates on high-latency GPUs (e.g. A40).
        return DataLoader(
            dataset,
            num_workers=cfg["env_setting"].get("val_num_workers", 4),
            persistent_workers=True,
            shuffle=False,
            sampler=None,
            batch_size=cfg["training_cfg"].get("val_batch_size", 4),
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_pad,
        )

    if cfg["env_setting"]["num_gpus"] > 1:
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(cfg["training_cfg"]["training_epochs"])
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
    else:
        raise RuntimeError("Mamba needs GPU acceleration")

    generator = SEMamba(cfg).to(device)
    discriminator = MetricDiscriminator().to(device)

    if rank == 0:
        evaluator = Evaluator(sr=cfg["stft_cfg"]["sampling_rate"]).to(device)
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

    # Create validset and validation_loader if rank is 0
    if rank == 0:
        validset = create_dataset(cfg, train=False, split=False, device=device)
        validation_loader = create_dataloader(validset, cfg, train=False)
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

                # Validation
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
