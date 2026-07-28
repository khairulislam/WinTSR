"""Small tensor helpers used by the attribution methods.

These are vendored here (rather than imported from ``utils.tools``) so that the
``wintsr`` package is installable and usable without the rest of the research
repository.
"""

import torch

__all__ = ["normalize_scale", "get_baseline"]

_EPS = torch.finfo(torch.float32).eps


def normalize_scale(
    data: torch.Tensor,
    dim: int = 1,
    norm_type: str = "standard",
    legacy: bool = False,
) -> torch.Tensor:
    """Normalize ``data`` along ``dim``.

    Args:
        data: tensor to normalize.
        dim: dimension to reduce over.
        norm_type: one of ``standard``, ``minmax`` or ``l1``.
        legacy: reproduce the exact ``minmax`` behaviour of the original WinTSR
            research code, which normalizes the whole tensor by the *first*
            slice's min/max instead of per-slice. See :class:`wintsr.WinTSR` for
            why this switch exists. Ignored for other ``norm_type`` values.
    """
    if norm_type == "standard":
        mean = data.mean(dim=dim, keepdim=True)
        std = data.std(dim=dim, keepdim=True)
        return (data - mean) / (std + _EPS)

    if norm_type == "minmax":
        max_val = torch.amax(data, dim=dim, keepdim=True)
        min_val = torch.amin(data, dim=dim, keepdim=True)
        if legacy:
            # The original code indexed the result of ``amax``/``amin`` with
            # ``[0]``, a leftover from ``torch.max`` returning a (values,
            # indices) tuple. With ``amax`` this selects the first slice, so
            # every sample gets normalized by sample 0's range.
            max_val, min_val = max_val[0], min_val[0]
        return (data - min_val) / (max_val - min_val + _EPS)

    if norm_type == "l1":
        sum_val = data.abs().sum(dim=dim, keepdim=True)
        # this converts neg to absolute values
        return data.abs() / (sum_val + _EPS)

    raise NameError(f'Normalize method "{norm_type}" not implemented')


def get_baseline(inputs, mode: str = "zero"):
    """Build a baseline tensor (or tuple of them) matching ``inputs``.

    Args:
        inputs: tensor or tuple of tensors shaped ``(batch, seq_len, features)``.
        mode: ``zero``, ``random``, ``normal`` or ``mean``.
    """
    if isinstance(inputs, tuple):
        return tuple(get_baseline(x, mode) for x in inputs)

    if mode == "zero":
        return torch.zeros_like(inputs).float()
    if mode == "random":
        return torch.randn_like(inputs).float()
    if mode == "normal":
        mean = inputs.mean(dim=(0, 1), keepdim=True)
        std = inputs.std(dim=(0, 1), keepdim=True)
        return torch.normal(
            mean.expand_as(inputs), std.expand_as(inputs).clamp_min(_EPS)
        ).float()
    if mode == "mean":
        return inputs.mean(dim=(0, 1), keepdim=True).expand_as(inputs).float()

    raise NameError(f'Baseline mode "{mode}" not implemented')
