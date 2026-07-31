# Supported models

WinTSR attributes **any callable** that maps `(batch, seq_len, n_features)` — or a tuple
of tensors — to predictions. There is no model registry and nothing to subclass: if the
forward pass is a differentiable-or-not PyTorch computation, it can be explained.

That said, real forecasting/classification models rarely take a single clean tensor.
The tables below are the models this package (and its baselines) have actually been
run against, and exactly how to wire each calling convention into `.attribute(...)`.

## Single-input models

Pass the series straight through — nothing to split.

| Model | Family |
| --- | --- |
| DLinear | Linear |
| LightTS | Linear/MLP |
| TiDE | Linear/MLP |
| FiLM | Linear/MLP |
| TSMixer | MLP-Mixer |
| FreTS | Frequency-domain MLP |
| MICN | Convolutional |
| Crossformer | Transformer |
| PatchTST | Transformer |
| Pyraformer | Transformer |
| SegRNN | Recurrent |
| Koopa | Koopman operator |
| LSTM / TCN | Recurrent / convolutional |
| CALF | LLM-backed foundation model |
| OFA (GPT4TS) | LLM-backed foundation model |
| TimeLLM | LLM-backed foundation model |

```python
attr = WinTSR(model).attribute(inputs=x_enc, baselines=torch.zeros_like(x_enc))
```

## Dual-input (TSlib) models

These [TSlib](https://github.com/thuml/Time-Series-Library) models consume every
forward argument themselves (`x_enc, x_mark_enc, x_dec, x_mark_dec`), so the calendar
features get attributed alongside the series. `wintsr.attr.tsr.DUAL_INPUT_USERS` is the
canonical list — pass `dual_input_users=[...]` to `TSR` if you're explaining a model not
on it.

| Model | Family |
| --- | --- |
| Transformer | Transformer |
| Informer | Transformer |
| Autoformer | Transformer |
| FEDformer | Transformer |
| ETSformer | Transformer |
| Nonstationary Transformer | Transformer |
| Reformer | Transformer |
| iTransformer | Transformer |
| TimeXer | Transformer |
| TimeMixer | MLP-Mixer |
| TimesNet | Convolutional |
| RNN | Recurrent |

```python
attr_enc, attr_mark = WinTSR(model).attribute(
    inputs=(x_enc, x_mark_enc),
    baselines=(torch.zeros_like(x_enc), torch.zeros_like(x_mark_enc)),
    additional_forward_args=(x_dec, x_mark_dec),
)
```

See the [integration cookbook](integration.md) for the full walkthrough, or the
[TSlib models notebook](https://colab.research.google.com/github/khairulislam/WinTSR/blob/main/notebooks/tslib_models.ipynb)
to run it.

## Model zoo and training harness

Trained checkpoints, dataset loaders, and experiment scripts for all of the above live
in [WinTSR-research](https://github.com/khairulislam/WinTSR-research) — the paper's
training/interpretation harness, which depends on this package the same way any user
would (`pip install wintsr`). This repository ships the attribution methods only; it
does not vendor model code.
