# tslens

### A PyTorch framework for interpreting time-series deep learning models

[![arXiv](https://img.shields.io/badge/arXiv-2412.04532-b31b1b.svg)](https://arxiv.org/abs/2412.04532)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khairulislam/tslens/blob/main/notebooks/quickstart.ipynb)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://khairulislam.github.io/tslens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Which time steps and features did your time-series model actually use?**

tslens is a drop-in, Captum-compatible interpretability toolkit for PyTorch
time-series models. Every method produces a saliency map over the input window so you
can inspect *when* and *where* a model found evidence for its prediction.

Explaining time series models is hard for two reasons that interpretation methods borrowed
from vision and NLP do not handle: subsequent time steps are strongly dependent, and
feature importance varies over time. Existing studies (1) do not consider the temporal
dependencies among the feature vectors in the input window, and (2) consider the time
dimension separately from the feature dimension when calculating importance scores.
**Windowed Temporal Saliency Rescaling (WinTSR)**, the toolkit's native flagship
method, addresses both.

![WinTSR attribution heatmap](docs/assets/wintsr_heatmap.png)

## Why tslens?

- **Model-agnostic:** explain any callable PyTorch model with the expected input shape.
- **Time-aware:** preserve relationships between neighbouring observations, via WinTSR.
- **Joint attribution:** identify important time–feature regions, not just global features.
- **Practical:** support multi-input models, custom baselines, classification, and forecasting.
- **One interface:** compare established interpretation methods from Captum, Time Interpret,
  and this package's own native implementations side by side.

## Quickstart

```bash
pip install tslens
```

Requires Python 3.9+ and PyTorch 1.13+. Works with any PyTorch model mapping
`(batch, seq_len, n_features)` to predictions — no training framework to adopt, no
dataset format to conform to:

```python
import torch
from tslens import WinTSR

inputs = torch.randn(16, 96, 7)             # (batch, seq_len, n_features)
attr = WinTSR(model).attribute(
    inputs,
    baselines=torch.zeros_like(inputs),
    threshold=0.5,
)
attr.shape   # (16, n_output, 96, 7)
```

For local development, install the package with its test and documentation tools:

```bash
pip install -e ".[dev,docs]"
pytest
```

Plot `attr` as a heatmap over `(seq_len, n_features)` to read off what the model used.

### Getting started

| | |
| --- | --- |
| **[Quickstart notebook](https://colab.research.google.com/github/khairulislam/tslens/blob/main/notebooks/quickstart.ipynb)** | 60 seconds, no dataset download. Plants a known signal, trains a small GRU, checks WinTSR recovers it. |
| **[TSlib models notebook](https://colab.research.google.com/github/khairulislam/tslens/blob/main/notebooks/tslib_models.ipynb)** | Explaining DLinear, iTransformer, TimesNet and friends. No wrapper class needed. |
| **[Classification notebook](https://colab.research.google.com/github/khairulislam/tslens/blob/main/notebooks/classification.ipynb)** | `n_output` becomes the class count; a padding mask goes through as context, not as an attributed input. |
| **[Baselines notebook](https://colab.research.google.com/github/khairulislam/tslens/blob/main/notebooks/baselines.ipynb)** | Why the baseline matters, `get_baseline`'s four modes compared by attribution quality, and how to use a custom one. |
| **[Custom outputs notebook](https://colab.research.google.com/github/khairulislam/tslens/blob/main/notebooks/custom_outputs.ipynb)** | Wrapping models that return a dict or tuple (TimeLLM-style) with a one-line lambda. |
| **[Integration cookbook](https://khairulislam.github.io/tslens/integration/)** | Copy-paste recipes: dict/tuple outputs, classification, baselines, single horizons, speed, troubleshooting. |
| **[Interpretation methods](https://khairulislam.github.io/tslens/methods/)** | What WinTSR, TSR, WinIT and GateMask each do differently, and when to reach for which one. |
| **[API reference](https://khairulislam.github.io/tslens/reference/wintsr/)** | Every argument, generated from the docstrings, including `legacy_normalize=True` to reproduce the published numbers. |

Already have a TSlib model? It takes four tensors, so split them into what you want
attributed and what is just context:

```python
attr_enc, attr_mark = WinTSR(model).attribute(
    inputs=(x_enc, x_mark_enc),
    baselines=(torch.zeros_like(x_enc), torch.zeros_like(x_mark_enc)),
    additional_forward_args=(x_dec, x_mark_dec),
)
```

## Interpretation methods

tslens gives you a broad set of interpretation methods through one interface, and works
with any PyTorch time series model. WinTSR, TSR, WinIT, and GateMask are implemented
natively here; the rest call [Captum](https://captum.ai/docs/introduction) and
[tint](https://josephenguehard.github.io/time_interpret/build/html/index.html) directly.
See [Interpretation methods](https://khairulislam.github.io/tslens/methods/) for what
each one does differently and when to use it.

<details>
<summary><b>Supported methods</b> (click to expand)</summary>

| Method | Type | Paper |
| --- | --- | --- |
| WinTSR | Perturbation | [Islam & Fox, arXiv:2412.04532](https://arxiv.org/abs/2412.04532) |
| TSR | Gradient | [Ismail et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/47a3893cc405396a5c30d91320572d6d-Paper.pdf) |
| WinIT | Perturbation | [Leung et al., ICLR 2023](https://arxiv.org/abs/2107.14317) |
| GateMask | Learned mask | [Liu et al., ICLR 2024](https://arxiv.org/abs/2401.08552) |
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

Full comparison matrix (model requirement, baseline, API doc):
[Interpretation methods](https://khairulislam.github.io/tslens/methods/).
</details>

## Supported models

tslens attributes any callable that maps `(batch, seq_len, n_features)` — or a tuple of
tensors — to predictions, so nothing here is hard-coded to a specific architecture. The
list below is what this package and its [research harness](https://github.com/khairulislam/WinTSR-research)
have actually been run against.

<details>
<summary><b>Supported model architectures</b> (click to expand)</summary>

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

Calling convention (single vs. dual-input) and trained checkpoints:
[Supported models](https://khairulislam.github.io/tslens/models/).
</details>

## Repository layout

| Path | What it is |
| --- | --- |
| [src/tslens/](/src/tslens/) | The installable library. `WinTSR` plus the paper's baseline methods (`TSR`, `WinIT`, `GateMask`). |
| [notebooks/](/notebooks/) | Runnable quickstart and TSlib walkthrough. |
| [tests/](/tests/) | Test suite, including a numerical-equivalence check against the pre-refactor implementation. |
| [docs/](/docs/) | Source for the [docs site](https://khairulislam.github.io/tslens/): tutorials, method explainer, API reference, and the PyPI-page library reference. |

This repo is the library only. The training/interpretation harness that produced the
paper's results — model zoo, experiment scripts, saved results — lives in
[WinTSR-research](https://github.com/khairulislam/WinTSR-research) and depends on this
package the same way any user would (`pip install tslens`).

## Reproducing the paper

Training the models, running the full benchmark, and the paper's saved results live in
[WinTSR-research](https://github.com/khairulislam/WinTSR-research), which installs this
package as a regular dependency. That repo also has the model zoo — DLinear,
iTransformer, TimesNet, CALF, TimeLLM and 25 others from
[TSlib](https://github.com/thuml/Time-Series-Library) — dataset download instructions,
and Docker/Singularity definitions.

## Citation

Find our paper on [arXiv](https://arxiv.org/pdf/2412.04532). Please cite the following if you use our work
(also available as [CITATION.cff](CITATION.cff), used by GitHub's "Cite this repository" button).

```bibtex
@article{islam2024wintsr,
  title={WinTSR: A Windowed Temporal Saliency Rescaling Method for Interpreting Time Series Deep Learning Models},
  author={Islam, Md Khairul and Fox, Judy},
  journal={arXiv preprint arXiv:2412.04532},
  year={2024}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Core libraries

tslens builds on these open-source projects:

- **[Captum](https://captum.ai/docs/introduction)** — model interpretability library for PyTorch.
- **[Time Interpret (tint)](https://josephenguehard.github.io/time_interpret/build/html/index.html)** — extends Captum with methods designed for time series.
- **[Time-Series-Library (TSlib)](https://github.com/thuml/Time-Series-Library)** — deep time series analysis models used in the benchmark.

<!-- add here https://github.com/thuml/OpenLTM, https://github.com/thuml/Large-Time-Series-Model, https://github.com/fzi-forschungszentrum-informatik/TSInterpret.git -->