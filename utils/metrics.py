from dataclasses import dataclass

import torch
from auraloss.freq import MultiResolutionSTFTLoss
from torchmetrics.audio import PerceptualEvaluationSpeechQuality


@dataclass
class EvalMetrics:
    mrstft: float
    pesq: float
    utmos: float


class Evaluator:
    def __init__(self, sr):

        self._mrstft = MultiResolutionSTFTLoss(sample_rate=sr)
        self._pesq = PerceptualEvaluationSpeechQuality(fs=sr, mode="wb")
        self._utmos = (
            torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
            .eval()
        )
        self._sr = sr

    def compute(self, clean, pred) -> EvalMetrics:
        mrstft_loss = self._mrstft(pred.unsqueeze(1), clean.unsqueeze(1))

        try:
            pesq_score = self._pesq(pred, clean).item()
        except Exception as e:
            # PESQ raises on silent/degenerate utterances (common early in training)
            print(f"Error computing PESQ score: {e}")
            pesq_score = -1.0

        utmos_score = self._utmos(pred, self._sr)

        return EvalMetrics(
            mrstft_loss.item(),
            pesq_score,
            utmos_score.item(),
        )

    def to(self, device) -> "Evaluator":
        self._mrstft.to(device)
        self._pesq.to(device)
        self._utmos.to(device)
        return self
