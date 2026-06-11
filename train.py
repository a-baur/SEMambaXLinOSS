import warnings

from utils.viz import log_audio_and_spectrograms

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from dataloaders.dataloader_vctk import VCTKDemandDataset
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import SEMamba
from models.loss import phase_losses
from models.discriminator import MetricDiscriminator, batch_pesq
from models.linoss.linoss import LinOSS
from models.linoss.selective_linoss import MambOSS
from models.linoss.mamboss6 import MambOSS6
from models.s5.s5 import S5SSM
from utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed,
    print_gpu_info, log_model_info, initialize_process_group,
)
from utils.metrics import Evaluator

torch.backends.cudnn.benchmark = True


def create_partitioned_optimizer(model, base_lr=1e-3, ssm_lr_factor=0.01, betas=(0.9, 0.999), weight_decay=1e-2):
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

    for param in model.parameters():
        if param.requires_grad and id(param) not in ssm_param_ids:
            rest_params.append(param)

    param_groups = [
        {"params": rest_params, "lr": base_lr, "weight_decay": weight_decay},
        {"params": ssm_params, "lr": base_lr * ssm_lr_factor, "weight_decay": 0.0}
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
    lr_decay = cfg['training_cfg']['lr_decay']

    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay, last_epoch=last_epoch)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay, last_epoch=last_epoch)

    return scheduler_g, scheduler_d

def create_dataset(cfg, train=True, split=True, device='cuda:0'):
    """Create dataset based on cfguration."""
    clean_json = cfg['data_cfg']['train_clean_json'] if train else cfg['data_cfg']['valid_clean_json']
    noisy_json = cfg['data_cfg']['train_noisy_json'] if train else cfg['data_cfg']['valid_noisy_json']
    shuffle = (cfg['env_setting']['num_gpus'] <= 1) if train else False
    pcs = cfg['training_cfg']['use_PCS400'] if train else False

    return VCTKDemandDataset(
        clean_json=clean_json,
        noisy_json=noisy_json,
        sampling_rate=cfg['stft_cfg']['sampling_rate'],
        segment_size=cfg['training_cfg']['segment_size'],
        n_fft=cfg['stft_cfg']['n_fft'],
        hop_size=cfg['stft_cfg']['hop_size'],
        win_size=cfg['stft_cfg']['win_size'],
        compress_factor=cfg['model_cfg']['compress_factor'],
        split=split,
        n_cache_reuse=0,
        shuffle=shuffle,
        device=device,
        pcs=pcs
    )

def create_dataloader(dataset, cfg, train=True):
    """Create dataloader based on dataset and configuration."""
    if cfg['env_setting']['num_gpus'] > 1:
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(cfg['training_cfg']['training_epochs'])
        batch_size = (cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']) if train else 1
    else:
        sampler = None
        batch_size = cfg['training_cfg']['batch_size'] if train else 1
    num_workers = cfg['env_setting']['num_workers'] if train else 1

    return DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=(sampler is None) and train,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True if train else False
    )


def train(rank, args, cfg):
    num_gpus = cfg['env_setting']['num_gpus']
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    batch_size = cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']
    if num_gpus >= 1:
        initialize_process_group(cfg, rank)
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        raise RuntimeError("Mamba needs GPU acceleration")

    generator = SEMamba(cfg).to(device)
    discriminator = MetricDiscriminator().to(device)

    if rank == 0:
        evaluator = Evaluator(sr=cfg['stft_cfg']['sampling_rate']).to(device)
        log_model_info(rank, generator, args.exp_path)

    state_dict_g, state_dict_do, steps, last_epoch = load_ckpts(args, device)
    if state_dict_g is not None:
        generator.load_state_dict(state_dict_g['generator'], strict=False)
        discriminator.load_state_dict(state_dict_do['discriminator'], strict=False)

    if num_gpus > 1 and torch.cuda.is_available():
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    if cfg['training_cfg'].get('use_pretrainedD', False):
        discriminator.load_state_dict( torch.load('ckpts/pretrained_discriminator.pth') )
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
        sw = SummaryWriter(os.path.join(args.exp_path, 'logs'))

    generator.train()
    discriminator.train()

    best_pesq, best_pesq_step = 0.0, 0
    best_utmos, best_utmos_step = 0.0, 0
    for epoch in range(max(0, last_epoch), cfg['training_cfg']['training_epochs']):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
            clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
            clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))
            noisy_mag = torch.autograd.Variable(noisy_mag.to(device, non_blocking=True))
            noisy_pha = torch.autograd.Variable(noisy_pha.to(device, non_blocking=True))
            one_labels = torch.ones(batch_size).to(device, non_blocking=True)

            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            audio_list_r, audio_list_g = list(clean_audio.cpu().numpy()), list(audio_g.detach().cpu().numpy())
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
            _, _, rec_com = mag_phase_stft(audio_g, n_fft, hop_size, win_size, compress_factor, addeps=True)
            loss_con = F.mse_loss(com_g, rec_com) * 2

            loss_gen_all = (
                loss_metric * cfg['training_cfg']['loss']['metric'] +
                loss_mag * cfg['training_cfg']['loss']['magnitude'] +
                loss_pha * cfg['training_cfg']['loss']['phase'] +
                loss_com * cfg['training_cfg']['loss']['complex'] +
                loss_time * cfg['training_cfg']['loss']['time'] +
                loss_con * cfg['training_cfg']['loss']['consistancy']
            )

            loss_gen_all.backward()
            optim_g.step()
            # ------------------------------------------------------- #

            if rank == 0:
                # STDOUT logging
                if steps % cfg['env_setting']['stdout_interval'] == 0:
                    with torch.no_grad():
                        metric_error = F.mse_loss(metric_g.flatten(), one_labels).item()
                        mag_error = F.mse_loss(clean_mag, mag_g).item()
                        pha_error = (loss_ip + loss_gd + loss_iaf).item()
                        com_error = F.mse_loss(clean_com, com_g).item()
                        time_error = F.l1_loss(clean_audio, audio_g).item()
                        con_error = F.mse_loss( com_g, rec_com ).item()

                        print(
                            'Steps : {:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, Metric Loss: {:4.3f}, '
                            'Mag Loss: {:4.3f}, Pha Loss: {:4.3f}, Com Loss: {:4.3f}, Time Loss: {:4.3f}, Cons Loss: {:4.3f}, s/b : {:4.3f}'.format(
                                steps, loss_gen_all, loss_disc_all, metric_error, mag_error, pha_error, com_error, time_error, con_error, time.time() - start_b
                            )
                        )

                # Checkpointing
                if steps % cfg['env_setting']['checkpoint_interval'] == 0 and steps != 0:
                    exp_name = f"{args.exp_path}/g_{steps:08d}.pth"
                    save_checkpoint(
                        exp_name,
                        {
                            'generator': (generator.module if num_gpus > 1 else generator).state_dict()
                        }
                    )
                    exp_name = f"{args.exp_path}/do_{steps:08d}.pth"
                    save_checkpoint(
                        exp_name,
                        {
                            'discriminator': (discriminator.module if num_gpus > 1 else discriminator).state_dict(),
                            'optim_g': optim_g.state_dict(),
                            'optim_d': optim_d.state_dict(),
                            'steps': steps,
                            'epoch': epoch
                        }
                    )

                # Tensorboard summary logging
                if steps % cfg['env_setting']['summary_interval'] == 0:
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
                if steps % cfg['env_setting']['validation_interval'] == 0 and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()

                    val_metrics = {
                        "Magnitude Loss": 0,
                        "Phase Loss": 0,
                        "Complex Loss": 0,
                        "PESQ Score": 0,
                        "MultiResSTFT Loss": 0,
                        "UTMOS Score": 0,
                    }

                    num_viz_samples = cfg['env_setting'].get('num_viz_samples', 5)
                    viz_max_seconds = cfg['env_setting'].get('viz_max_seconds', 5.0)

                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
                            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
                            clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
                            clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
                            clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))
                            noisy_mag = noisy_mag.to(device, non_blocking=True)
                            noisy_pha = noisy_pha.to(device, non_blocking=True)

                            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

                            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)

                            min_len = min(clean_audio.size(-1), audio_g.size(-1))
                            metrics = evaluator.compute(clean_audio[..., :min_len], audio_g[..., :min_len])

                            # Log a handful of example waveforms / spectrograms to TensorBoard
                            if j < num_viz_samples:
                                noisy_audio = mag_phase_istft(noisy_mag, noisy_pha, n_fft, hop_size, win_size, compress_factor)
                                log_audio_and_spectrograms(
                                    sw, j, steps, cfg['stft_cfg']['sampling_rate'], hop_size, compress_factor,
                                    clean_audio, noisy_audio, audio_g,
                                    clean_mag, noisy_mag, mag_g,
                                    max_seconds=viz_max_seconds,
                                )

                            val_ip_err, val_gd_err, val_iaf_err = phase_losses(clean_pha, pha_g, cfg)
                            val_metrics["Phase Loss"] += (val_ip_err + val_gd_err + val_iaf_err).item()
                            val_metrics["Magnitude Loss"] += F.mse_loss(clean_mag, mag_g).item()
                            val_metrics["Complex Loss"] += F.mse_loss(clean_com, com_g).item()
                            val_metrics["PESQ Score"] += metrics.pesq
                            val_metrics["MultiResSTFT Loss"] += metrics.mrstft
                            val_metrics["UTMOS Score"] += metrics.utmos

                        log_str = f"VALIDATION"
                        for metric, scores in val_metrics.items():
                            score = scores / (j+1)  # average across batches
                            sw.add_scalar(f"Validation/{metric}", score, steps)
                            log_str += f" | {metric}: {score:.4f}"

                            if metric == "PESQ Score":
                                if score >= best_pesq:
                                    best_pesq = score
                                    best_pesq_step = steps
                                log_str += f" (max. {best_pesq:.4f} @ {best_pesq_step} steps)"
                            elif metric == "UTMOS Score":
                                if score >= best_utmos:
                                    best_utmos = score
                                    best_utmos_step = steps
                                log_str += f" (max. {best_utmos:.4f} @ {best_utmos_step} steps)"

                        print(log_str)
                    generator.train()


            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='exp')
    parser.add_argument('--exp_name', default='MambOSS6_EARS')
    parser.add_argument('--config', default='/data5/baur/SEMambaXLinOSS/recipes/latest.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg['env_setting']['seed']
    num_gpus = cfg['env_setting']['num_gpus']
    available_gpus = torch.cuda.device_count()

    if num_gpus > available_gpus:
        warnings.warn(
            f"Warning: The actual number of available GPUs ({available_gpus}) is less than the .yaml config ({num_gpus}). Auto reset to num_gpu = {available_gpus}",
            UserWarning
        )
        cfg['env_setting']['num_gpus'] = available_gpus
        num_gpus = available_gpus
        time.sleep(5)


    initialize_seed(seed)
    args.exp_path = os.path.join(args.exp_folder, args.exp_name)
    build_env(args.config, 'config.yaml', args.exp_path)

    if torch.cuda.is_available():
        print_gpu_info(cfg)
    else:
        print("CUDA is not available.")

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(args, cfg))
    else:
        train(0, args, cfg)

if __name__ == '__main__':
    main()
