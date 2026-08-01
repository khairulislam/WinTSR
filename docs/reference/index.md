# API reference

The API reference is generated from WinTSR's source docstrings, so signatures and
defaults stay synchronized with the installed package.

## Attribution methods

| API | Best starting point for |
| --- | --- |
| [`WinTSR`](wintsr.md) | Window-aware, joint time–feature attribution. |
| [`TSR`](tsr.md) | Gradient-based temporal saliency rescaling. |
| [`WinIT`](winit.md) | Delayed and distributional feature importance. |
| [`GateMask`](gate_mask.md) | Sparse explanations learned for each input. |

## Utilities

[`get_baseline`](functional.md) creates zero, mean, random, or feature-wise normal
reference tensors. [`normalize_scale`](functional.md) rescales attribution tensors.

!!! tip "Looking for an end-to-end example?"

    Start with the [integration cookbook](../integration.md). Return here when you need
    the complete parameter list or exact output-shape behavior.
