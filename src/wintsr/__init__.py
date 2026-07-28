"""WinTSR: Windowed Temporal Saliency Rescaling for time series models.

A drop-in, Captum-compatible attribution method for any PyTorch model that
consumes ``(batch, seq_len, n_features)`` tensors.

    >>> from wintsr import WinTSR
    >>> attr = WinTSR(model).attribute(inputs, baselines=torch.zeros_like(inputs))

Paper: https://arxiv.org/abs/2412.04532
"""

from .attr import WinTSR
from .functional import get_baseline, normalize_scale

__version__ = "0.1.0"
__all__ = ["WinTSR", "get_baseline", "normalize_scale", "__version__"]
