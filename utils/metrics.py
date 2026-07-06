import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import distillmos
import librosa
import numpy as np
import torch
from auraloss.freq import MultiResolutionSTFTLoss
from torchmetrics.audio import (
    NonIntrusiveSpeechQualityAssessment,
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalDistortionRatio,
    ShortTimeObjectiveIntelligibility,
)

# STFT used for the log-spectral distance (independent of the model's analysis STFT).
_LSD_N_FFT = 1024
_LSD_HOP = 256

HPC_ALEX_SPEECHMOS_PATH = "/home/hpc/f102ac/f102ac13/dev/SEMambaXLinOSS/SpeechMOS"

# DNSMOS Pro (https://github.com/fcumlin/DNSMOSPro): non-intrusive MOS predictor.
# TorchScript checkpoint (NISQA-trained, ~340 KB) auto-downloaded to the torch.hub
# cache; set DNSMOSPRO_MODEL_PATH to a local .pt on offline/HPC nodes.
DNSMOSPRO_MODEL_URL = (
    "https://raw.githubusercontent.com/fcumlin/DNSMOSPro/main/runs/NISQA/model_best.pt"
)


@dataclass
class EvalMetrics:
    # All optional so excluded metrics (see Evaluator.compute) come back as None.
    mrstft: float | None = None
    pesq: float | None = None
    utmos: float | None = None
    distillmos: float | None = None
    nisqa: float | None = None
    sisdr: float | None = None
    lsd: float | None = None
    estoi: float | None = None
    dnsmospro: float | None = None


class Evaluator:
    def __init__(self, sr, pesq_n_processes=1):
        self._mrstft = MultiResolutionSTFTLoss(sample_rate=sr)
        self._pesq = PerceptualEvaluationSpeechQuality(
            fs=sr, mode="wb", n_processes=pesq_n_processes
        )
        if os.path.exists(HPC_ALEX_SPEECHMOS_PATH):
            self._utmos = torch.hub.load(
                repo_or_dir=HPC_ALEX_SPEECHMOS_PATH,
                model="utmos22_strong",
                source="local",
            ).eval()
        else:
            self._utmos = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            ).eval()
        self._nisqa = NonIntrusiveSpeechQualityAssessment(sr)
        self._sisdr = ScaleInvariantSignalDistortionRatio()
        self._estoi = ShortTimeObjectiveIntelligibility(sr, extended=True)
        self._distillmos = distillmos.ConvTransformerSQAModel().eval()
        self._dnsmospro_model = self._load_dnsmospro()

        self._pesq_executor = ThreadPoolExecutor(max_workers=1)

        self._device = torch.device("cpu")
        self._sr = sr

    def _pesq_safe(self, pred, clean):
        """Batched PESQ mean as a float; -1.0 on the degenerate-utterance error."""
        try:
            return self._pesq(pred.cpu(), clean.cpu()).item()
        except Exception as e:
            # PESQ raises on silent/degenerate utterances (common early in training).
            print(f"Error computing PESQ score: {e}")
            return -1.0

    @staticmethod
    def _load_dnsmospro():
        """Load the DNSMOS Pro (NISQA-trained) TorchScript predictor.

        Uses a local checkpoint from ``DNSMOSPRO_MODEL_PATH`` when set (for
        offline/HPC nodes); otherwise downloads it once into the torch.hub cache.
        """
        path = os.environ.get("DNSMOSPRO_MODEL_PATH")
        if not path:
            cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
            os.makedirs(cache_dir, exist_ok=True)
            path = os.path.join(cache_dir, "dnsmospro_nisqa_model_best.pt")
            if not os.path.exists(path):
                torch.hub.download_url_to_file(DNSMOSPRO_MODEL_URL, path)
        return torch.jit.load(path, map_location="cpu").eval()

    def _dnsmospro_score(self, pred):
        """Mean DNSMOS Pro MOS over the batch as a 0-dim tensor.

        Non-intrusive (scores ``pred`` only). Reproduces the repo's log-magnitude
        STFT front end (n_fft=320 / hop=160, |STFT| clipped to [1e-7, 1e7], log10,
        time-major) so the predictor sees its training-time input distribution.
        """
        scores = []
        for wav in pred:
            samples = wav.detach().cpu().numpy()
            spec = librosa.stft(y=samples, win_length=320, hop_length=160, n_fft=320)
            spec = np.log10(np.clip(np.abs(spec.T), 1e-7, 1e7))
            spec = torch.from_numpy(spec).float().to(self._device)
            scores.append(self._dnsmospro_model(spec[None, None, ...])[0, 0])
        return torch.stack(scores).mean()

    def compute_val(self, clean, pred):
        """Fast batched metrics for the in-training validation loop.

        Computes PESQ, MR-STFT, UTMOS, SI-SDR and LSD for a whole (fixed-length,
        uniform) batch and returns their batch means as Python floats. PESQ runs on
        a CPU process pool launched on a background thread, so it overlaps with the
        GPU metrics instead of running after them. Returns
        ``(pesq, mrstft, utmos, sisdr, lsd)``.
        """
        # Launch PESQ (CPU) first so its pool spins up while the GPU metrics run.
        pesq_future = self._pesq_executor.submit(self._pesq_safe, pred, clean)
        mrstft = self._mrstft(pred.unsqueeze(1), clean.unsqueeze(1)).item()
        utmos = self._utmos(pred, self._sr).mean().item()
        sisdr = self._sisdr(pred, clean).item()
        lsd = self._lsd(clean, pred).item()
        pesq = pesq_future.result()
        return pesq, mrstft, utmos, sisdr, lsd

    def _lsd(self, clean, pred):
        """Mean log-spectral distance (dB-free, log10 power) as a 0-dim tensor.

        Returned as a tensor (not ``.item()``) so callers can accumulate on the
        GPU and sync once; see ``compute(as_tensor=True)``.
        """
        win = torch.hann_window(_LSD_N_FFT, device=clean.device)
        spec = lambda x: torch.stft(  # noqa: E731
            x, _LSD_N_FFT, _LSD_HOP, window=win, return_complex=True
        ).abs().pow(2)
        log_c = torch.log10(spec(clean) + 1e-10)
        log_p = torch.log10(spec(pred) + 1e-10)
        # L2 across frequency per frame, then averaged over frames (and batch).
        return torch.sqrt(torch.mean((log_c - log_p) ** 2, dim=-2)).mean()

    def compute(self, clean, pred, exclude=(), as_tensor=False) -> EvalMetrics:
        """Compute metrics; names in ``exclude`` are skipped and returned as None.

        Excluding the CPU-bound metrics (e.g. ``("nisqa", "estoi")``) keeps the
        in-training validation loop fast while ``evaluate.py`` still gets the
        full suite.

        ``as_tensor=True`` returns the GPU metrics as 0-dim tensors instead of
        Python floats, so the caller can sum them on-device and call ``.item()``
        once per validation pass. Each ``.item()`` forces a host<->device sync,
        which is very expensive at batch_size=1 on high-latency GPUs (e.g. A40);
        deferring them is the main reason in-training validation was slow. PESQ
        runs on CPU regardless, so it is always returned as a float.
        """
        exclude = set(exclude)
        keep = lambda name: name not in exclude  # noqa: E731

        mrstft_loss = (
            self._mrstft(pred.unsqueeze(1), clean.unsqueeze(1)) if keep("mrstft") else None
        )

        if not keep("pesq"):
            # Caller computes PESQ separately (e.g. once over the whole pass).
            pesq_score = None
        else:
            pesq_score = self._pesq_safe(pred, clean)

        utmos_score = self._utmos(pred, self._sr).mean() if keep("utmos") else None
        nisqa_score = self._nisqa(pred)[..., 0].mean() if keep("nisqa") else None
        sisdr_score = self._sisdr(pred, clean) if keep("sisdr") else None
        estoi_score = self._estoi(pred, clean) if keep("estoi") else None
        lsd_score = self._lsd(clean, pred) if keep("lsd") else None
        distillmos_score = None
        if keep("distillmos"):
            with torch.no_grad():
                distillmos_score = self._distillmos(pred).mean()
        dnsmospro_score = None
        if keep("dnsmospro"):
            with torch.no_grad():
                dnsmospro_score = self._dnsmospro_score(pred)

        if not as_tensor:
            t = lambda x: None if x is None else x.item()  # noqa: E731
            mrstft_loss = t(mrstft_loss)
            utmos_score, sisdr_score = t(utmos_score), t(sisdr_score)
            lsd_score, distillmos_score = t(lsd_score), t(distillmos_score)
            nisqa_score, estoi_score = t(nisqa_score), t(estoi_score)
            dnsmospro_score = t(dnsmospro_score)

        return EvalMetrics(
            mrstft_loss,
            pesq_score,
            utmos_score,
            distillmos_score,
            nisqa_score,
            sisdr_score,
            lsd_score,
            estoi_score,
            dnsmospro_score,
        )

    def to(self, device) -> "Evaluator":
        self._device = torch.device(device)
        self._mrstft.to(device)
        # self._pesq stays on CPU (numpy algorithm); see __init__.
        self._utmos.to(device)
        self._nisqa.to(device)
        self._sisdr.to(device)
        self._estoi.to(device)
        self._distillmos.to(device)
        self._dnsmospro_model.to(device)
        return self
