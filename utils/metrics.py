import os
from dataclasses import dataclass

import distillmos
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


class Evaluator:
    def __init__(self, sr):

        self._mrstft = MultiResolutionSTFTLoss(sample_rate=sr)
        self._pesq = PerceptualEvaluationSpeechQuality(
            fs=sr, mode="wb", n_processes=os.cpu_count() or 1
        )
        self._utmos = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        ).eval()
        self._nisqa = NonIntrusiveSpeechQualityAssessment(sr)
        self._sisdr = ScaleInvariantSignalDistortionRatio()
        self._estoi = ShortTimeObjectiveIntelligibility(sr, extended=True)
        self._distillmos = distillmos.ConvTransformerSQAModel().eval()

        self._device = torch.device("cpu")
        self._sr = sr

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
        mrstft_loss = self._mrstft(pred.unsqueeze(1), clean.unsqueeze(1))

        if "pesq" in exclude:
            # Caller computes PESQ separately (e.g. in parallel across a batch).
            pesq_score = None
        else:
            try:
                pesq_score = self._pesq(pred, clean).item()
            except Exception as e:
                # PESQ raises on silent/degenerate utterances (common early in training)
                print(f"Error computing PESQ score: {e}")
                pesq_score = -1.0

        utmos_score = self._utmos(pred, self._sr).mean()  # mean over batch
        nisqa_score = None if "nisqa" in exclude else self._nisqa(pred)[..., 0].mean()
        sisdr_score = self._sisdr(pred, clean)
        estoi_score = None if "estoi" in exclude else self._estoi(pred, clean)
        lsd_score = self._lsd(clean, pred)
        with torch.no_grad():
            distillmos_score = self._distillmos(pred).mean()

        if not as_tensor:
            t = lambda x: None if x is None else x.item()  # noqa: E731
            mrstft_loss, utmos_score, sisdr_score, lsd_score, distillmos_score = (
                mrstft_loss.item(),
                utmos_score.item(),
                sisdr_score.item(),
                lsd_score.item(),
                distillmos_score.item(),
            )
            nisqa_score, estoi_score = t(nisqa_score), t(estoi_score)

        return EvalMetrics(
            mrstft_loss,
            pesq_score,
            utmos_score,
            distillmos_score,
            nisqa_score,
            sisdr_score,
            lsd_score,
            estoi_score,
        )

    def to(self, device) -> "Evaluator":
        self._device = torch.device(device)
        self._mrstft.to(device)
        self._pesq.to(device)
        self._utmos.to(device)
        self._nisqa.to(device)
        self._sisdr.to(device)
        self._estoi.to(device)
        self._distillmos.to(device)
        return self
