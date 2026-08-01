# Interpretation methods

Choose an attribution method based on the model interface, explanation you need, and
available compute. The toolkit covers 14 local attribution methods for time-series
models across perturbation, gradient, distributional, learned-mask, and surrogate
approaches. They follow the familiar `.attribute(...)` pattern from
[Captum](https://captum.ai/docs/introduction) and
[Time Interpret](https://josephenguehard.github.io/time_interpret/build/html/index.html).

![WinTSR attribution heatmap recovering a planted ground-truth signal at feature 0, steps 20-25, on a synthetic AR(1) series](assets/wintsr_heatmap.png)

*A synthetic series with a known answer — one feature, one window, everything else
noise. WinTSR's saliency map (right two panels) lights up exactly where the ground
truth (left) says it should. Reproduce this with the
[quickstart notebook](https://colab.research.google.com/github/khairulislam/WinTSR/blob/main/notebooks/quickstart.ipynb).*

## At a glance

Following Captum's
[algorithm comparison matrix](https://captum.ai/docs/algorithms_comparison_matrix), here
is what each method needs from your model and how it's computed:

| Method | Type | Model requirement | Requires baseline | Paper | API doc |
| --- | --- | --- | --- | --- | --- |
| [WinTSR](reference/wintsr.md) | Perturbation | Any callable | Yes | [Islam & Fox, arXiv:2412.04532](https://arxiv.org/abs/2412.04532) | [reference/wintsr.md](reference/wintsr.md) |
| [TSR](reference/tsr.md) | Gradient | Differentiable | Yes | [Ismail et al., NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/47a3893cc405396a5c30d91320572d6d-Paper.pdf) | [reference/tsr.md](reference/tsr.md) |
| [WinIT](reference/winit.md) | Perturbation | Any callable | No (generative) | [Leung et al., ICLR 2023](https://arxiv.org/abs/2107.14317) | [reference/winit.md](reference/winit.md) |
| [GateMask](reference/gate_mask.md) | Learned mask | Any callable | No (learned) | [Liu et al., ICLR 2024](https://arxiv.org/abs/2401.08552) | [reference/gate_mask.md](reference/gate_mask.md) |
| Occlusion | Perturbation | Any callable | Yes | [Zeiler & Fergus, ECCV 2014](https://arxiv.org/abs/1311.2901) | [captum.ai/api/occlusion.html](https://captum.ai/api/occlusion.html) |
| Feature Ablation | Perturbation | Any callable | Yes | [Kokhlikyan et al., arXiv:2009.07896](https://arxiv.org/abs/2009.07896) | [captum.ai/api/feature_ablation.html](https://captum.ai/api/feature_ablation.html) |
| Feature Permutation | Perturbation | Any callable | No (permuted batch) | [Kokhlikyan et al., arXiv:2009.07896](https://arxiv.org/abs/2009.07896) | [captum.ai/api/feature_permutation.html](https://captum.ai/api/feature_permutation.html) |
| Augmented Occlusion | Perturbation | Any callable | No (bootstrapped) | [Enguehard, ICML 2023](https://proceedings.mlr.press/v202/enguehard23a.html) | [tint: attr.AugmentedOcclusion](https://josephenguehard.github.io/time_interpret/build/html/attr.html#tint.attr.AugmentedOcclusion) |
| Integrated Gradients | Gradient | Differentiable | Yes | [Sundararajan et al., ICML 2017](https://arxiv.org/abs/1703.01365) | [captum.ai/api/integrated_gradients.html](https://captum.ai/api/integrated_gradients.html) |
| Gradient SHAP | Gradient | Differentiable | Yes (distribution) | [Lundberg & Lee, NeurIPS 2017](https://arxiv.org/abs/1705.07874) | [captum.ai/api/gradient_shap.html](https://captum.ai/api/gradient_shap.html) |
| FIT | Perturbation | Any callable (classification only) | No (generative) | [Tonekaboni et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/08fa43588c2571ade19bc0fa5936e028-Abstract.html) | [tint: attr.Fit](https://josephenguehard.github.io/time_interpret/build/html/attr.html#tint.attr.Fit) |
| Dyna Mask | Learned mask | Any callable | No (learned) | [Crabbé & van der Schaar, ICML 2021](https://proceedings.mlr.press/v139/crabbe21a.html) | [tint: attr.DynaMask](https://josephenguehard.github.io/time_interpret/build/html/attr.html#tint.attr.DynaMask) |
| Extremal Mask | Learned mask | Any callable | No (learned) | [Enguehard, ICML 2023](https://proceedings.mlr.press/v202/enguehard23a.html) | [tint: attr.ExtremalMask](https://josephenguehard.github.io/time_interpret/build/html/attr.html#tint.attr.ExtremalMask) |
| Lime | Surrogate | Any callable | Yes | [Ribeiro et al., KDD 2016](https://arxiv.org/abs/1602.04938) | [captum.ai/api/lime.html](https://captum.ai/api/lime.html) |

Looking for which model architectures each of these has actually been run against?
See [Supported models](models.md).

## Occlusion-based

**[WinTSR](reference/wintsr.md)** — Two-stage attribution. Stage one scores each
**time step** by occluding it entirely (a *time-relevance score*). Stage two scores
each **feature within a time step**, but only for the time steps that clear a
relevance threshold from stage one, using a sliding window that respects the
dependency between neighbouring steps. The two scores multiply to give the final
attribution. Prior approaches either ignore the dependency between consecutive time
steps, or score time and features independently and combine them after the fact;
WinTSR does both jointly, and skips low-relevance time steps in stage two so it
stays fast on long sequences.

**Occlusion** — Slides a window over the input, zeroing (or replacing) each region in
turn and measuring the change in output. The base operation WinTSR's stage one builds
on.

**Feature Ablation** — Occludes one feature (or a user-defined group of features) at a
time across the whole sequence, rather than a sliding window.

**Feature Permutation** — Same idea as ablation, but shuffles a feature's values
across the batch instead of zeroing it, so the replacement stays in-distribution.

**Augmented Occlusion** — Occlusion with a learned, data-driven baseline (sampled from
a bootstrapped distribution) instead of a fixed zero or mean baseline.

## Gradient-based

**[TSR](reference/tsr.md)** — The method WinTSR generalizes. Also two-stage (time
relevance, then feature relevance), but computes both stages with Integrated
Gradients rather than occlusion, and does not account for temporal dependency between
time steps within a stage. Needs a differentiable model — no black-box support.

**Integrated Gradients** — Accumulates gradients along a straight-line path from a
baseline to the input, giving an attribution that satisfies completeness and
sensitivity axioms.

**Gradient SHAP** — Approximates Shapley values by averaging Integrated-Gradients-style
paths from multiple noisy baselines.

## Delayed / distributional importance

**[WinIT](reference/winit.md)** — Computes delayed feature importance: how much a
feature's *past* values (within a sliding window) still influence the *current*
prediction, using a distributional distance (Jensen-Shannon or prediction-difference)
between the real and counterfactual forecast. Unlike WinTSR, it does not separate a
time-relevance and a feature-relevance stage — importance is scored per
(time step, feature, delay) directly.

**FIT** — Scores each observation by the KL-divergence between the predictive
distribution with and without it. Classification only.

## Learned masks

**[GateMask](reference/gate_mask.md)** — The gating mechanism from ContraLSP: trains a
small network per input to produce sparse, binary-skewed gates over
`(time, feature)`, using counterfactual perturbations and contrastive learning to keep
the masked input's distribution close to the original. Requires fitting a mask network
per batch (via a [`pytorch_lightning.Trainer`](reference/gate_mask.md)), so it is
markedly slower than occlusion- or gradient-based methods.

**Dyna Mask** — Learns a per-timestep mask with a smoothness/sparsity penalty, trained
to preserve the model's prediction under the masked input.

**Extremal Mask** — Learns masks that push predictions towards a target extremum
(rather than just preserving the original prediction), trained jointly with the
perturbation applied under the mask.

## Surrogate

**Lime** — Fits a local, interpretable surrogate model (typically linear) around the
input to approximate the black-box model's behaviour in that neighbourhood.

## Choosing a method

- **Default:** WinTSR. It's the method this package exists for, and the paper's results
  show it best recovers ground-truth relevance in both time and feature dimensions.
- **Need a differentiable, gradient-only method:** TSR, Integrated Gradients, or
  Gradient SHAP.
- **Classification model, want instance-wise delayed importance:** WinIT or FIT.
- **Want a learned, sparse binary mask rather than a continuous score:** GateMask,
  Dyna Mask, or Extremal Mask.
- **Want a quick, model-agnostic baseline with no training step:** Occlusion, Feature
  Ablation, Feature Permutation, or Augmented Occlusion.
