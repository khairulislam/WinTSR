# Related work

WinTSR builds on a growing body of research in temporal attribution, perturbation,
evaluation, and intrinsically interpretable forecasting. This page provides a map of
the field and positions the methods available through the toolkit.

## Temporal attribution methods

| Work | Venue | Contribution |
| --- | --- | --- |
| [Benchmarking deep learning interpretability in time series predictions](https://proceedings.neurips.cc/paper_files/paper/2020/file/47a3893cc405396a5c30d91320572d6d-Paper.pdf) | NeurIPS 2020 | Introduces Temporal Saliency Rescaling (TSR) and evaluation metrics for time and feature importance. |
| [What went wrong and when? Instance-wise feature importance for time-series black-box models](https://proceedings.neurips.cc/paper/2020/hash/08fa43588c2571ade19bc0fa5936e028-Abstract.html) | NeurIPS 2020 | Introduces FIT, which measures prediction changes using KL divergence. |
| [Temporal Dependencies in Feature Importance for Time Series Prediction](https://arxiv.org/abs/2107.14317) | ICLR 2023 | Introduces WinIT and delayed feature importance over sliding temporal windows. |
| [Learning Perturbations to Explain Time Series Predictions](https://proceedings.mlr.press/v202/enguehard23a.html) | ICML 2023 | Learns attribution masks and the perturbations applied beneath them. |
| [CGS-Mask: Making Time Series Predictions Intuitive for All](https://ojs.aaai.org/index.php/AAAI/article/view/29325) | AAAI 2024 | Produces binary, time-sensitive explanations with cellular genetic strip masks. |
| [Encoding time-series explanations through self-supervised model behavior consistency](https://proceedings.neurips.cc/paper_files/paper/2023/file/65ea878cb90b440e8b4cd34fe0959914-Paper-Conference.pdf) | NeurIPS 2023 | Trains interpretable surrogates while preserving pretrained-model behavior. |

## Evaluation and benchmarking

| Work | Venue | Contribution |
| --- | --- | --- |
| [Evaluation of post-hoc interpretability methods in time-series classification](https://www.nature.com/articles/s42256-023-00620-w) | Nature Machine Intelligence 2023 | Proposes quantitative evaluation criteria for classification explanations. |
| [Evaluation of interpretability methods for multivariate time series forecasting](https://link.springer.com/article/10.1007/s10489-021-02662-2) | Applied Intelligence 2022 | Benchmarks attribution methods for multivariate, multi-horizon forecasting. |
| [Optimal local explainer aggregation for interpretable prediction](https://ojs.aaai.org/index.php/AAAI/article/view/21458) | AAAI 2022 | Combines local explanations into near-global views through integer optimization. |

## Interpretable temporal models

| Work | Venue | Contribution |
| --- | --- | --- |
| [Temporal Fusion Transformers for interpretable multi-horizon time series forecasting](https://www.sciencedirect.com/science/article/pii/S0169207021000637) | International Journal of Forecasting 2021 | Combines high-performance forecasting with variable selection and temporal attention. |
| [Self-Interpretable Time Series Prediction with Counterfactual Explanations](https://proceedings.mlr.press/v202/yan23d.html) | ICML 2023 | Uses variational inference for temporal abduction, intervention, and prediction. |
| [iTrendRNN: An Interpretable Trend-Aware RNN for Meteorological Spatiotemporal Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/30217) | AAAI 2024 | Models evolving meteorological trends with interpretable attention units. |

## Surveys

- [Explainable artificial intelligence (XAI) on time-series data: a survey](https://arxiv.org/abs/2104.00950)
- [Interpretation of Time-Series Deep Models: A Survey](https://arxiv.org/abs/2305.14582)
- [Deep learning for time series forecasting: tutorial and literature survey](https://dl.acm.org/doi/10.1145/3533382)

For a practical comparison of methods available in this package, including model and
baseline requirements, see [Interpretation methods](methods.md).
