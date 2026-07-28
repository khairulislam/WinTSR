"""Attribution methods for time series models.

:class:`WinTSR` is the method introduced in the paper. :class:`TSR`,
:class:`WinIT` and :class:`GateMask` are the baselines used in its evaluation.
All four work from a plain ``pip install wintsr`` -- none of them require the
research harness under ``research/``.

The baselines are imported lazily so that ``import wintsr`` stays cheap and does
not drag in ``pytorch_lightning`` or ``sklearn`` unless a method that needs them
is actually requested.
"""

from typing import TYPE_CHECKING

from .wintsr import WinTSR

if TYPE_CHECKING:  # pragma: no cover
    from .gate_mask import GateMask
    from .tsr import TSR
    from .winit import WinIT

_LAZY = {"TSR": ".tsr", "WinIT": ".winit", "GateMask": ".gate_mask"}

__all__ = ["WinTSR", "TSR", "WinIT", "GateMask"]


def __getattr__(name: str):
    """PEP 562 lazy import for the baseline methods."""
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    try:
        module = import_module(_LAZY[name], __name__)
    except ImportError as exc:
        raise ImportError(
            f"{name} could not be imported. It is a baseline method whose "
            "dependencies normally arrive with `time-interpret`; a broken or "
            "partial install is the usual cause. Try reinstalling with "
            f"`pip install --force-reinstall wintsr`. Original error: {exc}"
        ) from exc

    value = getattr(module, name)
    globals()[name] = value  # cache so __getattr__ runs once per name
    return value


def __dir__():
    return sorted(__all__)
