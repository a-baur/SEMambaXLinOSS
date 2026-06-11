import torch
import matplotlib
matplotlib.use('Agg')  # headless backend, no display required on training nodes
from matplotlib import pyplot as plt


def log_audio_and_spectrograms(sw, idx, steps, sr, hop_size, compress_factor,
                               clean_audio, noisy_audio, enhanced_audio,
                               clean_mag, noisy_mag, enhanced_mag,
                               log_reference=True, max_seconds=5.0, top_db=80.0):
    """Log one validation example's waveforms and spectrograms to TensorBoard.

    The clean/noisy reference signals are static across training, so they only
    need to be logged once (``log_reference=True``); subsequent validation runs
    can pass ``log_reference=False`` to log only the (changing) enhanced output.
    ``max_seconds`` clips the rendered spectrograms to that many seconds to keep
    the figures readable for long utterances. ``top_db`` fixes the displayed
    dynamic range (dB below each spectrogram's peak) so the magnitude colour
    scale is bounded and comparable across the clean/noisy/enhanced panels.
    """
    max_frames = int(max_seconds * sr / hop_size) if max_seconds is not None else None

    if log_reference:
        sw.add_audio(f"Audio/{idx}_clean", _peak_normalize(clean_audio), steps, sr)
        sw.add_audio(f"Audio/{idx}_noisy", _peak_normalize(noisy_audio), steps, sr)
        clean_fig = _spectrogram_figure(clean_mag.squeeze(), compress_factor, sr, hop_size, "Clean", max_frames, top_db)
        noisy_fig = _spectrogram_figure(noisy_mag.squeeze(), compress_factor, sr, hop_size, "Noisy", max_frames, top_db)
        sw.add_figure(f"Spectrogram/{idx}_clean", clean_fig, steps)
        sw.add_figure(f"Spectrogram/{idx}_noisy", noisy_fig, steps)

    sw.add_audio(f"Audio/{idx}_enhanced", _peak_normalize(enhanced_audio), steps, sr)
    enhanced_fig = _spectrogram_figure(enhanced_mag.squeeze(), compress_factor, sr, hop_size, "Enhanced", max_frames, top_db)
    sw.add_figure(f"Spectrogram/{idx}_enhanced", enhanced_fig, steps)
    plt.close('all')


def _peak_normalize(audio):
    """Scale a waveform to [-1, 1] for clip-free TensorBoard playback.

    The dataloader's RMS normalization can push peaks beyond 1, which TensorBoard
    audio would clip; peak-normalizing each clip preserves the waveform shape.
    """
    audio = audio.detach().float().cpu().reshape(-1)
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak
    return audio


def _spectrogram_figure(mag, compress_factor, sr, hop_size, title="", max_frames=None, top_db=80.0):
    """Render a dB-magnitude spectrogram figure from a compressed-magnitude tensor.

    `mag` is the (RMS-normalized) power-compressed magnitude [F, T] produced by
    mag_phase_stft. We undo the power compression before converting to dB so the
    plot matches a conventional spectrogram. The shared RMS norm_factor applied in
    the dataloader cancels for relative comparisons across clean/noisy/enhanced.
    `max_frames`, if given, clips the time axis to that many STFT frames. The dB
    values are referenced to each clip's peak (0 dB = peak) and clamped to a fixed
    `top_db`-wide window, giving a normalized, bounded colour scale instead of a
    per-figure auto-range dominated by the noise floor.
    """
    mag = mag.detach().float().cpu()
    if max_frames is not None:
        mag = mag[..., :max_frames]
    linear = mag.clamp_min(0).pow(1.0 / compress_factor)
    db = (20.0 * torch.log10(linear + 1e-5)).numpy()
    db = db - db.max()  # reference to peak: 0 dB = loudest bin
    n_frames = db.shape[-1]

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(
        db, origin='lower', aspect='auto', cmap='magma',
        vmin=-top_db, vmax=0.0,
        extent=[0, n_frames * hop_size / sr, 0, sr / 2.0 / 1000.0],
    )
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (kHz)')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, format='%+2.0f dB')
    fig.tight_layout()
    return fig
