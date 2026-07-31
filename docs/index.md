# WinTSR

**Which time steps and which features did your time series model actually use?**

WinTSR (Windowed Temporal Saliency Rescaling) is a local attribution method for deep
time series models. Unlike attribution methods borrowed from vision and NLP, it accounts
for the temporal dependency between neighbouring time steps and scores the time and
feature dimensions jointly rather than separately.

Paper: [arXiv:2412.04532](https://arxiv.org/abs/2412.04532) ·
Code: [github.com/khairulislam/WinTSR](https://github.com/khairulislam/WinTSR)

## Install

```bash
pip install wintsr
```

Requires Python 3.9+ and PyTorch 1.13+.

## Use

Works with any PyTorch model that maps `(batch, seq_len, n_features)` to predictions.
No training framework to adopt, no dataset format to conform to.

```python
import torch
from wintsr import WinTSR

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
| `baselines` | Replacement values for occluded regions. Defaults to zeros; [`wintsr.get_baseline`](reference/functional.md) gives other options. |
| `unflatten` | `True` (default) returns `(batch, n_output, seq_len, n_features)`. `False` returns the flat `(batch * n_output, ...)` layout used internally. |
| `legacy_normalize` | Constructor flag. Restores the exact normalization used to produce the published numbers — see [Reproducing the paper](#reproducing-the-paper). |

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

### Where to next

- **[Tutorials](integration.md)** — copy-paste recipes for dict/tuple model outputs,
  classification, baseline choice, single horizons, speed tuning, and troubleshooting.
- **[Interpretation methods](methods.md)** — what WinTSR, TSR, WinIT and GateMask each
  do differently, and when to reach for which one.
- **[API reference](reference/wintsr.md)** — every argument, generated from the
  docstrings.
- Runnable notebooks:
  [quickstart](https://colab.research.google.com/github/khairulislam/WinTSR/blob/main/notebooks/quickstart.ipynb)
  and
  [TSlib models](https://colab.research.google.com/github/khairulislam/WinTSR/blob/main/notebooks/tslib_models.ipynb).

## Reproducing the paper

The published results were produced with a min-max normalization that scaled the whole
batch by the first sample's range. The packaged default normalizes each sample
independently. Thresholding is unaffected either way (it is a per-sample quantile,
invariant to a shared affine transform), but final attribution magnitudes differ. To
reproduce the paper exactly:

```python
WinTSR(model, legacy_normalize=True)
```

Training the models and running the full benchmark lives in a separate repo,
[WinTSR-research](https://github.com/khairulislam/WinTSR-research), which depends on
this package the same way any other user would.

## Citation

```bibtex
@article{islam2024wintsr,
  title={WinTSR: A Windowed Temporal Saliency Rescaling Method for Interpreting Time Series Deep Learning Models},
  author={Islam, Md Khairul and Fox, Judy},
  journal={arXiv preprint arXiv:2412.04532},
  year={2024}
}
```
