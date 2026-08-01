# Integration cookbook

Every snippet here is executed as part of the test suite or a notebook. Copy, paste, adapt.

The contract is always the same:

```python
attr = WinTSR(model).attribute(inputs, baselines=..., additional_forward_args=...)
```

- **`inputs`** — tensors you want explained. They get perturbed, and you get one
  attribution per input.
- **`additional_forward_args`** — everything else the model needs. Passed through
  untouched, never attributed.

Attributions come back shaped `(batch, n_output, seq_len, n_features)`.

---

## Your own model

If it maps `(batch, seq_len, n_features)` to predictions, there is nothing to configure.

```python
import torch
from wintsr import WinTSR

inputs = torch.randn(16, 96, 7)
attr = WinTSR(model).attribute(inputs, baselines=torch.zeros_like(inputs))
# (16, n_output, 96, 7)
```

Full walkthrough: [quickstart notebook](https://colab.research.google.com/github/khairulislam/WinTSR/blob/main/notebooks/quickstart.ipynb).

## TSlib models (DLinear, iTransformer, TimesNet, Autoformer, ...)

TSlib models take four tensors. Split them: the two encoder inputs are attributed, the
two decoder inputs are context. **No wrapper class needed.**

```python
attr_enc, attr_mark = WinTSR(model).attribute(
    inputs=(x_enc, x_mark_enc),
    baselines=(torch.zeros_like(x_enc), torch.zeros_like(x_mark_enc)),
    additional_forward_args=(x_dec, x_mark_dec),
    threshold=0.5,
)
```

`attr_enc` is the one you usually want. Because the model returns
`(batch, pred_len, c_out)`, `n_output` is `pred_len * c_out` — one saliency map per
predicted value.

Full walkthrough: [TSlib models notebook](https://colab.research.google.com/github/khairulislam/WinTSR/blob/main/notebooks/tslib_models.ipynb).

## Single-input foundation models (CALF, OFA/GPT4TS)

These consume only the series. Drop the tuples.

```python
attr = WinTSR(model).attribute(
    inputs=x_enc,
    baselines=torch.zeros_like(x_enc),
)
```

## Classification models

Pass the padding mask (and anything else) as context. `n_output` becomes the class count.

```python
attr = WinTSR(model).attribute(
    inputs=batch_x,
    baselines=torch.zeros_like(batch_x),
    additional_forward_args=(padding_mask,),
)
# (batch, n_classes, seq_len, n_features)
```

Full walkthrough: [classification notebook](https://colab.research.google.com/github/khairulislam/WinTSR/blob/main/notebooks/classification.ipynb).

## Models that return a dict or a tuple

Attribution needs a tensor. Wrap the model in a callable that picks the right field —
`WinTSR` accepts any callable, not just `nn.Module`.

```python
# model returns {"outputs_time": ..., "outputs_text": ...}
attr = WinTSR(lambda x: model(x)["outputs_time"]).attribute(inputs, baselines=...)

# model returns (predictions, attention_weights)
attr = WinTSR(lambda x: model(x)[0]).attribute(inputs, baselines=...)
```

Full walkthrough: [custom outputs notebook](https://colab.research.google.com/github/khairulislam/WinTSR/blob/main/notebooks/custom_outputs.ipynb).

## Choosing a baseline

The baseline is what an occluded region is replaced with. Zeros are the default and are
fine for standardized data.

```python
from wintsr import get_baseline

get_baseline(inputs, "zero")     # zeros (default)
get_baseline(inputs, "random")   # standard normal noise
get_baseline(inputs, "normal")   # sampled from each feature's own mean/std
get_baseline(inputs, "mean")     # each feature's mean, broadcast
```

Use `"normal"` or `"mean"` when zero is a meaningful value in your data and would itself
look like a signal.

Full walkthrough: [baselines notebook](https://colab.research.google.com/github/khairulislam/WinTSR/blob/main/notebooks/baselines.ipynb).

## Explaining one forecast horizon

`n_output` indexes the predictions. Index it instead of averaging.

```python
attr = WinTSR(model).attribute(inputs, baselines=zeros)

attr[:, 0]                       # first horizon only
attr.abs().mean(dim=1)           # averaged over all horizons
```

## Speed

`threshold` is the main dial. It is the quantile of time-relevance below which time steps
are skipped in stage two, so higher means fewer model calls and a sparser map.

```python
WinTSR(model).attribute(inputs, baselines=zeros, threshold=0.0)   # every time step
WinTSR(model).attribute(inputs, baselines=zeros, threshold=0.5)   # skip the bottom half
WinTSR(model).attribute(inputs, baselines=zeros, threshold=0.9)   # fastest, sparsest
```

Widening the temporal window also reduces the number of positions to evaluate:

```python
WinTSR(model).attribute(inputs, baselines=zeros, sliding_window_shapes=(6, 1))
```

> `perturbations_per_eval > 1` only works for **single-output** models. Multi-output
> models must leave it at `1`. This is an upstream limitation of tint's FeatureAblation —
> plain `tint.attr.Occlusion` fails the same way — and WinTSR raises a clear `ValueError`
> explaining it rather than letting the raw assertion through.

## Comparing against other methods

WinTSR is Captum-compatible, so the baselines are one-liners. The other methods from the
paper ship in the package too.

```python
from captum.attr import IntegratedGradients
from tint.attr import FeatureAblation, Occlusion
from wintsr.attr import WinTSR, WinIT, GateMask, TSR

Occlusion(model).attribute(inputs, sliding_window_shapes=(1, 1), baselines=zeros)
IntegratedGradients(model).attribute(inputs, baselines=zeros)
```

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `AssertionError: All inputs must have the same time dimension` | Tensors in `inputs` disagree on `shape[1]`. Move the odd one to `additional_forward_args`. |
| `ValueError: perturbations_per_eval > 1 only works for single-output models` | Leave `perturbations_per_eval` at `1`. |
| Attribution is entirely zero | `threshold` is too high; try `0.0`. |
| Shape is `(batch * n_output, seq_len, n_features)` | You passed `unflatten=False`. |
| Attribution looks like noise | Check the model actually learned — an untrained model has nothing to explain. |
