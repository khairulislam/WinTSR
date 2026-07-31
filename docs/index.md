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

Plot it as a heatmap over `(seq_len, n_features)` and you can read off what the model used:

![WinTSR attribution heatmap recovering a planted ground-truth signal at feature 0, steps 20-25, on a synthetic AR(1) series](assets/wintsr_heatmap.png)

### Options that matter

| Argument | Effect |
| --- | --- |
| `threshold` | Quantile of time-relevance below which steps are skipped in stage two. Higher is faster and sparser; `0.0` keeps every step. |
| `sliding_window_shapes` | Window over `(time, features)`. Defaults to `(1, 1)`. Widen the first entry to attribute over multi-step windows. |
| `baselines` | Replacement values for occluded regions. Defaults to zeros; [`wintsr.get_baseline`](reference/functional.md) gives other options. |
| `unflatten` | `True` (default) returns `(batch, n_output, seq_len, n_features)`. `False` returns the flat `(batch * n_output, ...)` layout used internally. |
| `legacy_normalize` | Constructor flag. Restores the exact normalization used to produce the published numbers. |

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

??? note "All 14 supported interpretation methods (click to expand)"

    Four — WinTSR, TSR, WinIT, GateMask — are implemented natively; the other ten call
    [Captum](https://captum.ai/docs/introduction) and
    [tint](https://josephenguehard.github.io/time_interpret/build/html/index.html)
    directly. Full comparison matrix (model requirement, baseline, API doc) on the
    [Interpretation methods](methods.md) page.

    | Method | Type | Paper |
    | --- | --- | --- |
    | [WinTSR](reference/wintsr.md) | Perturbation | [Islam & Fox, arXiv:2412.04532](https://arxiv.org/abs/2412.04532) |
    | [TSR](reference/tsr.md) | Gradient | [Ismail et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/47a3893cc405396a5c30d91320572d6d-Paper.pdf) |
    | [WinIT](reference/winit.md) | Perturbation | [Leung et al., ICLR 2023](https://arxiv.org/abs/2107.14317) |
    | [GateMask](reference/gate_mask.md) | Learned mask | [Liu et al., ICLR 2024](https://arxiv.org/abs/2401.08552) |
    | Occlusion | Perturbation | [Zeiler & Fergus, ECCV 2014](https://arxiv.org/abs/1311.2901) |
    | Feature Ablation | Perturbation | [Kokhlikyan et al., arXiv:2009.07896](https://arxiv.org/abs/2009.07896) |
    | Feature Permutation | Perturbation | [Kokhlikyan et al., arXiv:2009.07896](https://arxiv.org/abs/2009.07896) |
    | Augmented Occlusion | Perturbation | [Enguehard, ICML 2023](https://proceedings.mlr.press/v202/enguehard23a.html) |
    | Integrated Gradients | Gradient | [Sundararajan et al., ICML 2017](https://arxiv.org/abs/1703.01365) |
    | Gradient SHAP | Gradient | [Lundberg & Lee, NeurIPS 2017](https://arxiv.org/abs/1705.07874) |
    | FIT | Perturbation | [Tonekaboni et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/08fa43588c2571ade19bc0fa5936e028-Abstract.html) |
    | Dyna Mask | Learned mask | [Crabbé & van der Schaar, ICML 2021](https://proceedings.mlr.press/v139/crabbe21a.html) |
    | Extremal Mask | Learned mask | [Enguehard, ICML 2023](https://proceedings.mlr.press/v202/enguehard23a.html) |
    | Lime | Surrogate | [Ribeiro et al., KDD 2016](https://arxiv.org/abs/1602.04938) |

??? note "All 29 supported model architectures (click to expand)"

    WinTSR attributes any callable that maps `(batch, seq_len, n_features)` — or a
    tuple of tensors — to predictions, so nothing here is hard-coded. This is what the
    package and its [research harness](https://github.com/khairulislam/WinTSR-research)
    have actually been run against. Calling convention (single vs. dual-input) on the
    [Supported models](models.md) page.

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

### Where to next

- **[Tutorials](integration.md)** — copy-paste recipes for dict/tuple model outputs,
  classification, baseline choice, single horizons, speed tuning, and troubleshooting.
- **[Interpretation methods](methods.md)** — what WinTSR, TSR, WinIT and GateMask each
  do differently, and when to reach for which one.
- **[Supported models](models.md)** — which model architectures and calling
  conventions this works with out of the box.
- **[API reference](reference/wintsr.md)** — every argument, generated from the
  docstrings.
- Runnable notebooks:
  [quickstart](https://colab.research.google.com/github/khairulislam/WinTSR/blob/main/notebooks/quickstart.ipynb)
  and
  [TSlib models](https://colab.research.google.com/github/khairulislam/WinTSR/blob/main/notebooks/tslib_models.ipynb).

## Citation

```bibtex
@article{islam2024wintsr,
  title={WinTSR: A Windowed Temporal Saliency Rescaling Method for Interpreting Time Series Deep Learning Models},
  author={Islam, Md Khairul and Fox, Judy},
  journal={arXiv preprint arXiv:2412.04532},
  year={2024}
}
```
