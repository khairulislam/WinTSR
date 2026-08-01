"""tslens: attribution methods for time series deep learning models.

A drop-in, Captum-compatible interpretability toolkit for any PyTorch model
that consumes ``(batch, seq_len, n_features)`` tensors. Includes WinTSR and
other natively implemented methods, plus a consistent interface over Captum
and Time Interpret baselines.

    >>> from tslens import WinTSR
    >>> attr = WinTSR(model).attribute(inputs, baselines=torch.zeros_like(inputs))

WinTSR paper: https://arxiv.org/abs/2412.04532
"""

from .attr import WinTSR
from .functional import get_baseline, normalize_scale

__version__ = "0.1.0"
__all__ = ["WinTSR", "get_baseline", "normalize_scale", "__version__"]
