from dataclasses import dataclass

import torch
from auraloss.freq import MultiResolutionSTFTLoss
from torchmetrics.audio import (
    NonIntrusiveSpeechQualityAssessment,
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalDistortionRatio,
    ShortTimeObjectiveIntelligibility,
)
import distillmos


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
        self._pesq = PerceptualEvaluationSpeechQuality(fs=sr, mode="wb")
        self._utmos = (
            torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            .eval()
        )
        self._nisqa = NonIntrusiveSpeechQualityAssessment(sr)
        self._sisdr = ScaleInvariantSignalDistortionRatio()
        self._estoi = ShortTimeObjectiveIntelligibility(sr, extended=True)
        self._distillmos = distillmos.ConvTransformerSQAModel().eval()

        self._device = torch.device("cpu")
        self._sr = sr

    def _lsd(self, clean, pred) -> float:
        """Mean log-spectral distance (dB-free, log10 power) between clean and pred."""
        win = torch.hann_window(_LSD_N_FFT, device=clean.device)
        spec = lambda x: torch.stft(  # noqa: E731
            x, _LSD_N_FFT, _LSD_HOP, window=win, return_complex=True
        ).abs().pow(2)
        log_c = torch.log10(spec(clean) + 1e-10)
        log_p = torch.log10(spec(pred) + 1e-10)
        # L2 across frequency per frame, then averaged over frames (and batch).
        return torch.sqrt(torch.mean((log_c - log_p) ** 2, dim=-2)).mean().item()

    def compute(self, clean, pred, exclude=()) -> EvalMetrics:
        """Compute metrics; names in ``exclude`` are skipped and returned as None.

        Excluding the CPU-bound metrics (e.g. ``("nisqa", "estoi")``) keeps the
        in-training validation loop fast while ``evaluate.py`` still gets the
        full suite.
        """
        exclude = set(exclude)
        mrstft_loss = self._mrstft(pred.unsqueeze(1), clean.unsqueeze(1))

        try:
            pesq_score = self._pesq(pred, clean).item()
        except Exception as e:
            # PESQ raises on silent/degenerate utterances (common early in training)
            print(f"Error computing PESQ score: {e}")
            pesq_score = -1.0

        utmos_score = self._utmos(pred, self._sr).item()
        nisqa_score = None if "nisqa" in exclude else self._nisqa(pred)[0].item()  # overall MOS only
        sisdr_score = self._sisdr(pred, clean).item()
        estoi_score = None if "estoi" in exclude else self._estoi(pred, clean).item()
        lsd_score = self._lsd(clean, pred)
        with torch.no_grad():
            distillmos_score = self._distillmos(pred).mean().item()

        return EvalMetrics(
            mrstft_loss.item(),
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
