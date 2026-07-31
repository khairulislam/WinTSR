# Interpretation methods

`wintsr` ships four attribution methods that needed custom, time-series-aware
implementations. The paper also benchmarks against ten methods that work out of the box
from [Captum](https://captum.ai/docs/introduction) and
[tint](https://josephenguehard.github.io/time_interpret/build/html/index.html) — those
don't ship here because there's nothing to wrap.

## The four methods in this package

### [WinTSR](reference/wintsr.md) — the proposed method

Two-stage attribution. Stage one scores each **time step** by occluding it entirely
(a *time-relevance score*). Stage two scores each **feature within a time step**, but
only for the time steps that clear a relevance threshold from stage one, using a sliding
window that respects the dependency between neighbouring steps. The two scores multiply
to give the final attribution.

This is the point of the method: prior approaches either ignore the dependency between
consecutive time steps, or score time and features independently and combine them
after the fact. WinTSR does both jointly, and skips low-relevance time steps in stage
two so it stays fast on long sequences.

Use it as the default. Reach for one of the baselines below only if you're reproducing
a specific comparison from the paper.

### [TSR](reference/tsr.md) — Temporal Saliency Rescaling ([Ismail et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/47a3893cc405396a5c30d91320572d6d-Paper.pdf))

The method WinTSR generalizes. Also two-stage (time relevance, then feature relevance),
but computes both stages with Integrated Gradients rather than occlusion, and does not
account for temporal dependency between time steps within a stage. Gradient-based, so
it needs a differentiable model — no black-box support.

### [WinIT](reference/winit.md) — Windowed Feature Importance in Time ([ICLR 2023](https://openreview.net/forum?id=C0q9oBc3n4))

Computes delayed feature importance: how much a feature's *past* values (within a
sliding window) still influence the *current* prediction, using a distributional
distance (Jensen-Shannon or prediction-difference) between the real and counterfactual
forecast. Unlike WinTSR, it does not separate a time-relevance and a feature-relevance
stage — importance is scored per (time step, feature, delay) directly.

### [GateMask](reference/gate_mask.md) — the gating mechanism from ContraLSP ([Liu et al., ICLR 2024](https://arxiv.org/abs/2401.08552))

A learned-mask method: it trains a small network per input to produce sparse,
binary-skewed gates over `(time, feature)`, using counterfactual perturbations and
contrastive learning to keep the masked input's distribution close to the original.
Unlike the other three, this requires fitting a mask network per batch (via a
[`pytorch_lightning.Trainer`](reference/gate_mask.md)), so it is markedly slower than
occlusion- or gradient-based methods.

## The other ten, used directly from Captum/tint

No wrapper needed for any of these — call them exactly as documented upstream:

| Method | Library | Notes |
| --- | --- | --- |
| Feature Ablation | Captum | Occludes one feature (or group) at a time. |
| Feature Permutation | Captum | Shuffles a feature across the batch instead of zeroing it. |
| Occlusion | Captum | Sliding-window ablation; the base of WinTSR's stage one. |
| Augmented Occlusion | tint | Occlusion with a learned/data-driven baseline instead of zeros. |
| Dyna Mask | tint | Learns a per-timestep mask with a smoothness/sparsity penalty. |
| Extremal Mask | tint | Learns masks that push predictions towards a target extremum. |
| Lime | Captum | Local surrogate-model explanation. |
| FIT (Feature Importance in Time) | tint | KL-divergence-based importance; **classification only**. |
| Gradient SHAP | Captum | SHAP values via gradients and a noise-distribution baseline. |
| Integrated Gradients | Captum | Path integral of gradients from a baseline to the input. |

## Which one should I use?

- **Default:** WinTSR. It's the method this package exists for, and the paper's results
  show it best recovers ground-truth relevance in both time and feature dimensions.
- **Need a differentiable, gradient-only method:** TSR or Integrated Gradients.
- **Classification model, want instance-wise delayed importance:** WinIT or FIT.
- **Want a learned, sparse binary mask rather than a continuous score:** GateMask,
  Dyna Mask, or Extremal Mask.
