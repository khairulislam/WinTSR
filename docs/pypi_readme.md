# tslens

**Which time steps and which features did your time series model actually use?**

tslens is a Captum-compatible interpretability toolkit for deep time series models,
giving you a consistent PyTorch interface to WinTSR and other established interpretation
methods. WinTSR (Windowed Temporal Saliency Rescaling), the native flagship method,
accounts for the temporal dependency between neighbouring time steps and scores the
time and feature dimensions jointly rather than separately — unlike interpretation
methods borrowed from vision and NLP.

Paper: [arXiv:2412.04532](https://arxiv.org/abs/2412.04532) ·
Code: [github.com/khairulislam/tslens](https://github.com/khairulislam/tslens)

## Install

```bash
pip install tslens
```

## Use

Works with any PyTorch model that maps `(batch, seq_len, n_features)` to predictions.
No training framework to adopt, no dataset format to conform to.

```python
import torch
from tslens import WinTSR

inputs = torch.randn(16, 96, 7)          # (batch, seq_len, n_features)
attr = WinTSR(model).attribute(
    inputs,
    baselines=torch.zeros_like(inputs),
    threshold=0.5,                        # skip the least relevant time steps
)

attr.shape  # (16, n_output, 96, 7) -- (batch, n_output, seq_len, n_features)
```

Plot it as a heatmap over `(seq_len, n_features)` and you can read off what the model used.

### Options that matter

| Argument | Effect |
| --- | --- |
| `threshold` | Quantile of time-relevance below which steps are skipped in stage two. Higher is faster and sparser; `0.0` keeps every step. |
| `sliding_window_shapes` | Window over `(time, features)`. Defaults to `(1, 1)`. Widen the first entry to attribute over multi-step windows. |
| `baselines` | Replacement values for occluded regions. Defaults to zeros; `tslens.get_baseline(inputs, "normal")` gives other options. |
| `unflatten` | `True` (default) returns `(batch, n_output, seq_len, n_features)`. `False` returns the flat `(batch * n_output, ...)` layout used internally. |
| `legacy_normalize` | Constructor flag. Restores the exact normalization used to produce the published numbers — see below. |

### Multi-input models

Pass a tuple, get a tuple back. This is how you explain a
[TSlib](https://github.com/thuml/Time-Series-Library) model (DLinear, iTransformer,
TimesNet, ...) — its four forward arguments split into two attributed inputs and two
context tensors, with no wrapper class:

```python
attr_enc, attr_mark = WinTSR(model).attribute(
    inputs=(x_enc, x_mark_enc),
    baselines=(torch.zeros_like(x_enc), torch.zeros_like(x_mark_enc)),
    additional_forward_args=(x_dec, x_mark_dec),
)
```

### More recipes

The [integration cookbook](https://github.com/khairulislam/tslens/blob/main/docs/integration.md)
has copy-paste snippets for dict/tuple model outputs, classification models, baseline
choice, explaining a single forecast horizon, speed tuning, and troubleshooting. Two
runnable notebooks:
[quickstart](https://colab.research.google.com/github/khairulislam/tslens/blob/main/notebooks/quickstart.ipynb)
and
[TSlib models](https://colab.research.google.com/github/khairulislam/tslens/blob/main/notebooks/tslib_models.ipynb).

## Requirements

`torch`, `numpy`, `captum`, and [`time-interpret`](https://github.com/josephenguehard/time_interpret).

## Citation

```bibtex
@article{islam2024wintsr,
  title={WinTSR: A Windowed Temporal Saliency Rescaling Method for Interpreting Time Series Deep Learning Models},
  author={Islam, Md Khairul and Fox, Judy},
  journal={arXiv preprint arXiv:2412.04532},
  year={2024}
}
```
