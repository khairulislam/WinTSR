# Supported models

WinTSR attributes **any callable** that maps `(batch, seq_len, n_features)` — or a tuple
of tensors — to predictions. There is no model registry and nothing to subclass: if the
forward pass is a differentiable-or-not PyTorch computation, it can be explained.

That said, real forecasting/classification models rarely take a single clean tensor.
The tables below are the models this package (and its baselines) have actually been
run against, and exactly how to wire each calling convention into `.attribute(...)`.

## Single-input models

Pass the series straight through — nothing to split.

| Model | Family | Paper |
| --- | --- | --- |
| DLinear | Linear | [Zeng et al., AAAI 2023](https://arxiv.org/abs/2205.13504) |
| LightTS | Linear/MLP | [Zhang et al., arXiv:2207.01186](https://arxiv.org/abs/2207.01186) |
| TiDE | Linear/MLP | [Das et al., TMLR 2023](https://arxiv.org/abs/2304.08424) |
| FiLM | Linear/MLP | [Zhou et al., NeurIPS 2022](https://arxiv.org/abs/2205.08897) |
| TSMixer | MLP-Mixer | [Chen et al., TMLR 2023](https://arxiv.org/abs/2303.06053) |
| FreTS | Frequency-domain MLP | [Yi et al., NeurIPS 2023](https://arxiv.org/abs/2311.06184) |
| MICN | Convolutional | [Wang et al., ICLR 2023](https://openreview.net/pdf?id=zt53IDUR1U) |
| Crossformer | Transformer | [Zhang & Yan, ICLR 2023](https://openreview.net/pdf?id=vSVLM2j9eie) |
| PatchTST | Transformer | [Nie et al., ICLR 2023](https://arxiv.org/abs/2211.14730) |
| Pyraformer | Transformer | [Liu et al., ICLR 2022](https://openreview.net/pdf?id=0EXmFzUn5I) |
| SegRNN | Recurrent | [Lin et al., arXiv:2308.11200](https://arxiv.org/abs/2308.11200) |
| Koopa | Koopman operator | [Liu et al., NeurIPS 2023](https://arxiv.org/abs/2305.18803) |
| LSTM | Recurrent | [Hochreiter & Schmidhuber, Neural Computation 1997](https://www.bioinf.jku.at/publications/older/2604.pdf) |
| TCN | Convolutional | [Bai et al., arXiv:1803.01271](https://arxiv.org/abs/1803.01271) |
| CALF | LLM-backed foundation model | [Liu et al., arXiv:2403.07300](https://arxiv.org/abs/2403.07300) |
| OFA (GPT4TS) | LLM-backed foundation model | [Zhou et al., NeurIPS 2023](https://arxiv.org/abs/2302.11939) |
| TimeLLM | LLM-backed foundation model | [Jin et al., ICLR 2024](https://arxiv.org/abs/2310.01728) |

```python
attr = WinTSR(model).attribute(inputs=x_enc, baselines=torch.zeros_like(x_enc))
```

## Dual-input (TSlib) models

These [TSlib](https://github.com/thuml/Time-Series-Library) models consume every
forward argument themselves (`x_enc, x_mark_enc, x_dec, x_mark_dec`), so the calendar
features get attributed alongside the series. `wintsr.attr.tsr.DUAL_INPUT_USERS` is the
canonical list — pass `dual_input_users=[...]` to `TSR` if you're explaining a model not
on it.

| Model | Family | Paper |
| --- | --- | --- |
| Transformer | Transformer | [Vaswani et al., NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) |
| Informer | Transformer | [Zhou et al., AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17325) |
| Autoformer | Transformer | [Wu et al., NeurIPS 2021](https://openreview.net/pdf?id=I55UqU-M11y) |
| FEDformer | Transformer | [Zhou et al., ICML 2022](https://proceedings.mlr.press/v162/zhou22g.html) |
| ETSformer | Transformer | [Woo et al., arXiv:2202.01381](https://arxiv.org/abs/2202.01381) |
| Nonstationary Transformer | Transformer | [Liu et al., NeurIPS 2022](https://openreview.net/pdf?id=ucNDIDRNjjv) |
| Reformer | Transformer | [Kitaev et al., ICLR 2020](https://openreview.net/forum?id=rkgNKkHtvB) |
| iTransformer | Transformer | [Liu et al., ICLR 2024](https://arxiv.org/abs/2310.06625) |
| TimeXer | Transformer | [Wang et al., NeurIPS 2024](https://arxiv.org/abs/2402.19072) |
| TimeMixer | MLP-Mixer | [Wang et al., ICLR 2024](https://arxiv.org/abs/2405.14616) |
| TimesNet | Convolutional | [Wu et al., ICLR 2023](https://openreview.net/pdf?id=ju_Uqw384Oq) |
| RNN | Recurrent | [Hochreiter & Schmidhuber, Neural Computation 1997](https://www.bioinf.jku.at/publications/older/2604.pdf) |

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
