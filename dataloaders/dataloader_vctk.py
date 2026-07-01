import os
import re
import json
import random
import torch
import torch.utils.data
import librosa
from models.stfts import mag_phase_stft, mag_phase_istft
from models.pcs400 import cal_pcs

# Matches a trailing SNR suffix such as "_9.5dB" or "_-1.3dB" (EARS-WHAM noisy files).
_SNR_SUFFIX_RE = re.compile(r'_-?\d+(?:\.\d+)?dB$')

def list_files_in_directory(directory_path):
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith('.wav'):   # only add .wav files
                files.append(os.path.join(root, filename))
    return files

def load_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def remap_to_data_root(paths, data_root, orig_root):
    """Rebase absolute wav paths from ``orig_root`` onto ``data_root``.

    The dataset JSONs store absolute wav paths (clean and noisy under different
    subdirs of a shared parent). When the data has been staged to fast node-local
    storage (e.g. SLURM ``$TMPDIR``), set ``data_root`` to the staged base; each
    path is re-pointed there while preserving its layout relative to ``orig_root``.
    ``orig_root`` must be the prefix shared by *both* clean and noisy paths so the
    distinguishing subdir (clean_.../noisy_...) survives the rebase.
    """
    return [os.path.join(data_root, os.path.relpath(p, orig_root)) for p in paths]

def _common_root(file_paths):
    """Deepest directory shared by all paths (used to build a dataset-relative key)."""
    if len(file_paths) == 1:
        return os.path.dirname(file_paths[0])
    return os.path.commonpath(file_paths)

def extract_identifier(file_path, root):
    """
    Build a matching key from the path relative to its dataset root, with the
    extension and any trailing SNR suffix removed.

    This pairs clean/noisy files for both:
      - VCTK-DEMAND (flat dirs, identical basenames): "p231_272"
      - EARS-WHAM   (per-speaker dirs, "_<snr>dB" suffix): "p001/00000"
    """
    rel = os.path.relpath(file_path, root)
    stem = os.path.splitext(rel)[0]
    return _SNR_SUFFIX_RE.sub('', stem)

def get_clean_path_for_noisy(noisy_file_path, noisy_root, clean_path_dict):
    identifier = extract_identifier(noisy_file_path, noisy_root)
    return clean_path_dict.get(identifier, None)

class VCTKDemandDataset(torch.utils.data.Dataset):
    """
    Dataset for loading clean and noisy audio files.

    Args:
        clean_wavs_json (str): Directory containing clean audio files.
        noisy_wavs_json (str): Directory containing noisy audio files.
        audio_index_file (str): File containing audio indexes.
        sampling_rate (int, optional): Sampling rate of the audio files. Defaults to 16000.
        segment_size (int, optional): Size of the audio segments. Defaults to 32000.
        n_fft (int, optional): FFT size. Defaults to 400.
        hop_size (int, optional): Hop size. Defaults to 100.
        win_size (int, optional): Window size. Defaults to 400.
        compress_factor (float, optional): Magnitude compression factor. Defaults to 1.0.
        split (bool, optional): Whether to split the audio into segments. Defaults to True.
        n_cache_reuse (int, optional): Number of times to reuse cached audio. Defaults to 1.
        device (torch.device, optional): Target device. Defaults to None
        pcs (bool, optional): Use PCS in training period. Defaults to False
    """
    def __init__(
        self, 
        clean_json, 
        noisy_json, 
        sampling_rate=16000, 
        segment_size=32000,
        n_fft=400, 
        hop_size=100, 
        win_size=400, 
        compress_factor=1.0, 
        split=True,
        n_cache_reuse=1,
        shuffle=True,
        device=None,
        pcs=False,
        data_root=None,
        orig_data_root=None
    ):

        self.clean_wavs_path = load_json_file( clean_json )
        self.noisy_wavs_path = load_json_file( noisy_json )

        # Optionally rebase the stored absolute wav paths onto a staged copy of
        # the data (e.g. node-local $TMPDIR). orig_data_root defaults to the
        # prefix shared by every clean+noisy path so the clean/noisy subdirs are
        # preserved under data_root.
        if data_root:
            orig_root = orig_data_root or os.path.commonpath(
                self.clean_wavs_path + self.noisy_wavs_path
            )
            self.clean_wavs_path = remap_to_data_root(self.clean_wavs_path, data_root, orig_root)
            self.noisy_wavs_path = remap_to_data_root(self.noisy_wavs_path, data_root, orig_root)

        random.seed(1234)

        if shuffle:
            random.shuffle(self.noisy_wavs_path)

        # Roots used to derive dataset-relative matching keys for clean/noisy pairing.
        self.clean_root = _common_root(self.clean_wavs_path)
        self.noisy_root = _common_root(self.noisy_wavs_path)
        self.clean_path_dict = {
            extract_identifier(clean_path, self.clean_root): clean_path
            for clean_path in self.clean_wavs_path
        }

        self.sampling_rate = sampling_rate
        self.segment_size = segment_size
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.compress_factor = compress_factor
        self.split = split
        self.n_cache_reuse = n_cache_reuse

        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self._cache_ref_count = 0
        self.device = device
        self.pcs = pcs

    def __getitem__(self, index):
        """
        Get an audio sample by index.

        Args:
            index (int): Index of the audio sample.

        Returns:
            tuple: clean audio, clean magnitude, clean phase, clean complex, noisy magnitude, noisy phase
        """
        if self._cache_ref_count == 0:
            noisy_path = self.noisy_wavs_path[index]
            clean_path = get_clean_path_for_noisy(noisy_path, self.noisy_root, self.clean_path_dict)
            if clean_path is None:
                raise FileNotFoundError(
                    f"No matching clean file for noisy file {noisy_path!r} "
                    f"(matching key {extract_identifier(noisy_path, self.noisy_root)!r})"
                )
            noisy_audio, _ = librosa.load( noisy_path, sr=self.sampling_rate)
            clean_audio, _ = librosa.load( clean_path, sr=self.sampling_rate)
            if self.pcs == True:
                clean_audio = cal_pcs(clean_audio)
            self.cached_noisy_wav = noisy_audio
            self.cached_clean_wav = clean_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1

        clean_audio, noisy_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(noisy_audio)
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        assert clean_audio.size(1) == noisy_audio.size(1)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[:, audio_start:audio_start + self.segment_size]
                noisy_audio = noisy_audio[:, audio_start:audio_start + self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)), 'constant')
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_size - noisy_audio.size(1)), 'constant')

        clean_mag, clean_pha, clean_com = mag_phase_stft(clean_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)
        noisy_mag, noisy_pha, noisy_com = mag_phase_stft(noisy_audio, self.n_fft, self.hop_size, self.win_size, self.compress_factor)

        return (clean_audio.squeeze(), clean_mag.squeeze(), clean_pha.squeeze(), clean_com.squeeze(), noisy_mag.squeeze(), noisy_pha.squeeze())

    def __len__(self):
        return len(self.noisy_wavs_path)


def crop_collate_valid(batch, crop_samples, crop_frames):
    """Crop each full-length validation utterance to a fixed window and stack.

    ``VCTKDemandDataset`` with ``split=False`` yields full-length utterances, so
    items in a batch differ in duration. Cropping every item to the same window
    -- ``crop_samples`` in the time domain, ``crop_frames`` in the STFT domain --
    yields a uniform, pad-free batch that can be pushed through the model and the
    metrics in one shot (no padded frames wasting compute or biasing losses, and
    PESQ/UTMOS can score the whole batch at once). The leading window is taken so
    the crop is deterministic across runs. Utterances shorter than the window
    (none in the standard 16 kHz val sets, which are all >= 4 s) are right-padded
    with zeros as a safety net.

    Returns clean_audio [B, crop_samples], clean_mag/pha/noisy_mag/pha
    [B, F, crop_frames], clean_com [B, F, crop_frames, 2].
    """
    pad = torch.nn.functional.pad

    def fit_last(x, n):  # crop or right-pad the last axis to length n
        return x[..., :n] if x.size(-1) >= n else pad(x, (0, n - x.size(-1)))

    def fit_com(x, n):  # com is [F, T, 2]; the time axis is dim=1
        return x[:, :n] if x.size(1) >= n else pad(x, (0, 0, 0, n - x.size(1)))

    clean_audio = torch.stack([fit_last(b[0], crop_samples) for b in batch])
    clean_mag = torch.stack([fit_last(b[1], crop_frames) for b in batch])
    clean_pha = torch.stack([fit_last(b[2], crop_frames) for b in batch])
    clean_com = torch.stack([fit_com(b[3], crop_frames) for b in batch])
    noisy_mag = torch.stack([fit_last(b[4], crop_frames) for b in batch])
    noisy_pha = torch.stack([fit_last(b[5], crop_frames) for b in batch])

    return clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha
