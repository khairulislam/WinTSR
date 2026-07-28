# WinTSR: Interpreting Multi-Horizon Time Series Deep Learning Models

[![arXiv](https://img.shields.io/badge/arXiv-2412.04532-b31b1b.svg)](https://arxiv.org/abs/2412.04532)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/khairulislam/WinTSR/blob/main/notebooks/quickstart.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Which time steps and which features did your time series model actually use?**

Explaining time series models is hard for two reasons that attribution methods borrowed
from vision and NLP do not handle: subsequent time steps are strongly dependent, and
feature importance varies over time. Existing studies (1) do not consider the temporal
dependencies among the feature vectors in the input window, and (2) consider the time
dimension separately from the feature dimension when calculating importance scores.
**Windowed Temporal Saliency Rescaling (WinTSR)** addresses both.

## Quickstart

```bash
pip install wintsr
```

Requires Python 3.9+ and PyTorch 1.13+. Works with any PyTorch model mapping
`(batch, seq_len, n_features)` to predictions — no training framework to adopt, no
dataset format to conform to:

```python
import torch
from wintsr import WinTSR

inputs = torch.randn(16, 96, 7)             # (batch, seq_len, n_features)
attr = WinTSR(model).attribute(
    inputs,
    baselines=torch.zeros_like(inputs),
    threshold=0.5,
)
attr.shape   # (16, n_output, 96, 7)
```

Plot `attr` as a heatmap over `(seq_len, n_features)` to read off what the model used.

### Getting started

| | |
| --- | --- |
| **[Quickstart notebook](https://colab.research.google.com/github/khairulislam/WinTSR/blob/main/notebooks/quickstart.ipynb)** | 60 seconds, no dataset download. Plants a known signal, trains a small GRU, checks WinTSR recovers it. |
| **[TSlib models notebook](https://colab.research.google.com/github/khairulislam/WinTSR/blob/main/notebooks/tslib_models.ipynb)** | Explaining DLinear, iTransformer, TimesNet and friends. No wrapper class needed. |
| **[Integration cookbook](/docs/integration.md)** | Copy-paste recipes: dict/tuple outputs, classification, baselines, single horizons, speed, troubleshooting. |
| **[Library reference](/docs/pypi_readme.md)** | Every argument, including `legacy_normalize=True` to reproduce the published numbers. |

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

This package implements four attribution methods from the paper:

1. **WinTSR** — the proposed method
2. **TSR** [[NeurIPS 2020]](https://proceedings.neurips.cc/paper_files/paper/2020/file/47a3893cc405396a5c30d91320572d6d-Paper.pdf)
3. **WinIT** [[ICLR 2023]](https://openreview.net/forum?id=C0q9oBc3n4)
4. **GateMask** — the gating mechanism from *ContraLSP* [[ICLR 2024]](https://arxiv.org/pdf/2401.08552)

The paper also benchmarks against Feature Ablation, Feature Permutation, Occlusion,
Augmented Occlusion, Dyna Mask, Extremal Mask, Lime, FIT, Gradient SHAP, and Integrated
Gradients — all available directly from [Captum](https://captum.ai/docs/introduction) and
[tint](https://josephenguehard.github.io/time_interpret/build/html/index.html), no
wrapper needed.

## Repository layout

| Path | What it is |
| --- | --- |
| [src/wintsr/](/src/wintsr/) | The installable library. `WinTSR` plus the paper's baseline methods (`TSR`, `WinIT`, `GateMask`). |
| [notebooks/](/notebooks/) | Runnable quickstart and TSlib walkthrough. |
| [tests/](/tests/) | Test suite, including a numerical-equivalence check against the pre-refactor implementation. |
| [docs/](/docs/) | Integration cookbook and the PyPI-page library reference. |

This repo is the library only. The training/interpretation harness that produced the
paper's results — model zoo, experiment scripts, saved results — lives in
[WinTSR-research](https://github.com/khairulislam/WinTSR-research) and depends on this
package the same way any user would (`pip install wintsr`).

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

WinTSR builds on these open-source projects:

- **[Captum](https://captum.ai/docs/introduction)** — model interpretability library for PyTorch.
- **[Time Interpret (tint)](https://josephenguehard.github.io/time_interpret/build/html/index.html)** — extends Captum with methods designed for time series.
- **[Time-Series-Library (TSlib)](https://github.com/thuml/Time-Series-Library)** — deep time series analysis models used in the benchmark.
