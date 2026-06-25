"""Standalone PESQ helper, deliberately dependency-light.

Kept separate from ``utils.metrics`` so that ``joblib.Parallel`` worker
processes only import numpy + pesq (not torch / distillmos / torchmetrics) when
they unpickle ``pesq_wb``. That keeps process-pool spawn cheap during the
in-training validation pass.
"""
import numpy as np
from pesq import pesq as _pesq_backend


def pesq_wb(sr, ref, deg) -> float:
    """Wideband PESQ for a single utterance (numpy in), -1.0 on failure.

    PESQ is CPU-bound, serial per utterance, and (being a Cython/C backend that
    holds the GIL) only parallelises across *processes*. The -1.0 sentinel
    matches ``Evaluator.compute`` for degenerate/silent utterances.
    """
    try:
        return _pesq_backend(sr, np.asarray(ref, dtype=np.float64),
                             np.asarray(deg, dtype=np.float64), "wb")
    except Exception:
        return -1.0
