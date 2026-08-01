"""Windowed Temporal Saliency Rescaling (WinTSR).

Reference implementation for:

    Md Khairul Islam, Judy Fox.
    "WinTSR: A Windowed Temporal Saliency Rescaling Method for Interpreting
    Time Series Deep Learning Models." arXiv:2412.04532.
"""

from contextlib import contextmanager
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch
from captum._utils.common import _format_output
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._utils.attribution import Attribution, GradientAttribution
from captum.attr._utils.common import (
    _format_and_verify_sliding_window_shapes,
    _format_and_verify_strides,
    _format_input_baseline,
)
from captum.log import log_usage
from torch import Tensor
from tint.attr.occlusion import FeatureAblation, Occlusion

from ..functional import normalize_scale

__all__ = ["WinTSR"]


@contextmanager
def _clearer_perturbation_error(perturbations_per_eval: int):
    """Translate an upstream assertion into an actionable message.

    tint's FeatureAblation requires the forward output to have exactly
    ``batch_size`` elements when batching perturbations. Multi-output models
    break that, and the raw assertion mentions a ``feature_mask`` the caller
    never passed.
    """
    try:
        yield
    except AssertionError as exc:
        if perturbations_per_eval > 1 and "perturbations_per_eval" in str(exc):
            raise ValueError(
                "perturbations_per_eval > 1 only works for single-output "
                "models. This model returns more than one output per example, "
                "so use perturbations_per_eval=1 (the default). This is an "
                "upstream limitation of tint's FeatureAblation, and plain "
                "tint.attr.Occlusion fails the same way."
            ) from exc
        raise


class WinTSR(Occlusion):
    """Windowed Temporal Saliency Rescaling.

    A two-stage interpretation method for time series models. Stage one computes a
    *time-relevance score* per time step by occluding whole time steps. Stage
    two computes a *feature-relevance score* only on the time steps that clear a
    relevance threshold, using a sliding window that respects temporal
    dependencies. The two are multiplied to give the final attribution.

    Args:
        model: the model to interpret. Either a plain callable / ``nn.Module``
            taking ``(batch, seq_len, n_features)``, or an already-constructed
            Captum/tint :class:`~captum.attr._utils.attribution.Attribution`
            instance to use as the inner method. A bare model is wrapped in
            :class:`tint.attr.Occlusion`, which is the configuration reported in
            the paper.
        legacy_normalize: reproduce the exact time-relevance normalization of
            the original research code. That code normalized every sample in the
            batch by sample 0's min/max (an ``amax(...)[0]`` slip left over from
            ``torch.max``) instead of normalizing each sample independently.
            Thresholding is unaffected -- it is a per-row quantile, invariant to
            a shared affine transform -- but the final attribution magnitudes
            are. Defaults to ``False`` (per-sample normalization). Set to
            ``True`` to reproduce numbers from the paper's artifact exactly.

    Example:
        >>> import torch
        >>> from tslens import WinTSR
        >>> explainer = WinTSR(model)
        >>> attr = explainer.attribute(
        ...     inputs=torch.randn(8, 96, 7),
        ...     baselines=torch.zeros(8, 96, 7),
        ... )
    """

    def __init__(
        self,
        model: Union[Attribution, Callable],
        legacy_normalize: bool = False,
    ) -> None:
        if isinstance(model, Attribution):
            attribution_method = model
        elif callable(model):
            attribution_method = Occlusion(model)
        else:
            raise TypeError(
                "`model` must be a callable / nn.Module or a Captum "
                f"Attribution instance, got {type(model)!r}."
            )

        self.attribution_method = attribution_method
        self.legacy_normalize = legacy_normalize
        self.is_delta_supported = False
        self._multiply_by_inputs = self.attribution_method.multiplies_by_inputs
        self.is_gradient_method = isinstance(
            self.attribution_method, GradientAttribution
        )

        Occlusion.__init__(self, self.attribution_method.forward_func)
        self.use_weights = False  # We do not use weights for this method

    @property
    def multiplies_by_inputs(self):
        return self._multiply_by_inputs

    def has_convergence_delta(self) -> bool:
        return False

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        sliding_window_shapes: Optional[
            Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]
        ] = None,
        strides: Union[
            None, int, Tuple[int, ...], Tuple[Union[int, Tuple[int, ...]], ...]
        ] = None,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        threshold: float = 0.0,
        normalize: bool = True,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        unflatten: bool = True,
        **kwargs: Any,
    ) -> TensorOrTupleOfTensorsGeneric:
        """Compute WinTSR attributions.

        Args:
            inputs: tensor or tuple of tensors shaped
                ``(batch, seq_len, n_features)``. All inputs must share the time
                dimension.
            sliding_window_shapes: window shape per input, excluding the batch
                dimension. Defaults to all-ones (one time step, one feature),
                the setting used in the paper.
            strides: step size of the sliding window. Defaults to the window
                shape.
            baselines: replacement values for occluded regions. Defaults to zero.
            target: output index to attribute, for multi-output models.
            additional_forward_args: extra arguments passed to the model.
            threshold: quantile in ``[0, 1)`` of the time-relevance score below
                which time steps are skipped in stage two. Higher is faster and
                sparser; ``0.0`` keeps every time step.
            normalize: min-max normalize the time-relevance scores.
            perturbations_per_eval: number of perturbations batched per forward.
                Only supported for single-output models; anything returning more
                than one output per example must leave this at ``1``. This is an
                upstream limitation of tint's FeatureAblation.
            show_progress: display a progress bar.
            unflatten: return attributions shaped
                ``(batch, n_output, seq_len, n_features)``. The underlying
                computation produces a flat ``(batch * n_output, seq_len,
                n_features)``; set to ``False`` for that raw layout. This is a
                pure reshape -- the values are identical either way.

        Returns:
            Attributions shaped ``(batch, n_output, seq_len, n_features)``, or
            ``(batch * n_output, seq_len, n_features)`` when ``unflatten`` is
            ``False``. A tuple of these if ``inputs`` was a tuple.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs, baselines = _format_input_baseline(inputs, baselines)

        assert all(
            x.shape[1] == inputs[0].shape[1] for x in inputs
        ), "All inputs must have the same time dimension. (dimension 1)"

        # Default to a (1, 1, ...) window: one time step, one feature.
        if sliding_window_shapes is None:
            sliding_window_shapes = tuple(
                (1,) * (x.dim() - 1) for x in inputs
            )
            if not is_inputs_tuple:
                sliding_window_shapes = sliding_window_shapes[0]

        # Compute sliding window for the Time-Relevance Score
        # Only the time dimension (dim 1) has a sliding window of 1
        # shape (batch_size * n_output) x seq_len
        with _clearer_perturbation_error(perturbations_per_eval):
            time_relevance_score = self.get_time_relevance_score(
                inputs=inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                perturbations_per_eval=perturbations_per_eval,
                show_progress=show_progress,
            )

        # Normalize if required along the time axis
        if normalize:
            # normalize the last dimension
            time_relevance_score = tuple(
                normalize_scale(
                    tsr, dim=-1, norm_type="minmax", legacy=self.legacy_normalize
                )
                for tsr in time_relevance_score
            )

        # Get indexes where the Time-Relevance Score is
        # higher than the threshold
        is_above_threshold = tuple(
            score > torch.quantile(score, threshold, dim=-1, keepdim=True)
            for score in time_relevance_score
        )

        # Formatting strides
        strides = _format_and_verify_strides(strides, inputs)

        # Formatting sliding window shapes
        sliding_window_shapes = _format_and_verify_sliding_window_shapes(
            sliding_window_shapes, inputs
        )

        # Construct tensors from sliding window shapes
        sliding_window_tensors = tuple(
            torch.ones(window_shape, device=inputs[i].device)
            for i, window_shape in enumerate(sliding_window_shapes)
        )

        # Construct number of steps taking the threshold into account
        shift_counts = []
        for i, inp in enumerate(inputs):
            current_shape = np.subtract(
                inp.shape[2:], sliding_window_shapes[i][1:]
            )

            # On the temporal dim, the count shift is the maximum number
            # of element above the threshold
            non_zero_count = torch.unique(
                is_above_threshold[i].nonzero()[:, 0], return_counts=True
            )[1]
            if non_zero_count.sum() == 0:
                shift_count_time_dim = np.array([0])
            else:
                shift_count_time_dim = np.subtract(
                    non_zero_count.max().item(), sliding_window_shapes[i][0]
                )
            current_shape = np.insert(current_shape, 0, shift_count_time_dim)

            shift_counts.append(
                tuple(
                    np.add(
                        np.ceil(np.divide(current_shape, strides[i])).astype(int),
                        1,
                    )
                )
            )

        # Compute Feature-Relevance Score (step 2)
        with _clearer_perturbation_error(perturbations_per_eval):
            features_relevance_score = FeatureAblation.attribute.__wrapped__(
                self,
                inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                perturbations_per_eval=perturbations_per_eval,
                sliding_window_tensors=sliding_window_tensors,
                shift_counts=tuple(shift_counts),
                is_above_threshold=is_above_threshold,
                strides=strides,
                attributions_fn=abs,
                show_progress=show_progress,
            )

        # Reshape attributions before merge
        time_relevance_score = tuple(
            tsr.reshape(f_imp.shape[:2] + (1,) * len(f_imp.shape[2:]))
            for f_imp, tsr in zip(features_relevance_score, time_relevance_score)
        )

        is_above_threshold = tuple(
            is_above.reshape(f_imp.shape[:2] + (1,) * len(f_imp.shape[2:]))
            for f_imp, is_above in zip(features_relevance_score, is_above_threshold)
        )

        # Merge attributions:
        # Time-Relevance Score x Feature-Relevance Score
        attributions = tuple(
            tsr * frs
            for tsr, frs in zip(time_relevance_score, features_relevance_score)
        )

        if unflatten:
            # Rows are batch-major: row i holds batch i // n_output, output
            # i % n_output. Split them into explicit dimensions.
            attributions = tuple(
                attr.reshape((inp.shape[0], -1) + tuple(attr.shape[1:]))
                for attr, inp in zip(attributions, inputs)
            )

        return _format_output(is_inputs_tuple, attributions)

    def get_time_relevance_score(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ):
        """Stage one: occlude whole time steps to score temporal relevance."""
        tsr_sliding_window_shapes = tuple(
            (1,) + tuple(x.shape[2:]) for x in inputs
        )
        time_relevance_score = Occlusion.attribute.__wrapped__(
            self,
            inputs=inputs,
            sliding_window_shapes=tsr_sliding_window_shapes,
            strides=None,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            attributions_fn=abs,
            show_progress=show_progress,
        )

        # time_relevance_score shape will be ((N x O) x seq_len) after summation
        time_relevance_score = tuple(
            tsr.sum(tuple(i for i in range(2, len(tsr.shape))))
            for tsr in time_relevance_score
        )
        return time_relevance_score

    def _construct_ablated_input(
        self,
        expanded_input: Tensor,
        input_mask: Union[None, Tensor],
        baseline: Union[Tensor, int, float],
        start_feature: int,
        end_feature: int,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Ablates given expanded_input tensor with given feature mask, feature range,
        and baselines, and any additional arguments.
        expanded_input shape is (num_features, num_examples, ...)
        with remaining dimensions corresponding to remaining original tensor
        dimensions and num_features = end_feature - start_feature.
        input_mask is None for occlusion, and the mask is constructed
        using sliding_window_tensors, strides, and shift counts, which are provided in
        kwargs. baseline is expected to be broadcastable to match expanded_input.
        """
        input_mask = torch.stack(
            [
                self._occlusion_mask(
                    expanded_input,
                    j,
                    kwargs["sliding_window_tensors"],
                    kwargs["strides"],
                    kwargs["shift_counts"],
                    kwargs.get("is_above_threshold", None),
                )
                for j in range(start_feature, end_feature)
            ],
            dim=0,
        ).long()

        ablated_tensor = (
            expanded_input
            * (
                torch.ones(1, dtype=torch.long, device=expanded_input.device)
                - input_mask[:, : expanded_input.shape[1]]
            ).to(expanded_input.dtype)
        ) + (
            baseline
            * input_mask[:, : expanded_input.shape[1]].to(expanded_input.dtype)
        )

        return ablated_tensor, input_mask

    def _occlusion_mask(
        self,
        expanded_input: Tensor,
        ablated_feature_num: int,
        sliding_window_tsr: Tensor,
        strides: Union[int, Tuple[int, ...]],
        shift_counts: Tuple[int, ...],
        is_above_threshold: Tensor = None,
    ) -> Tensor:
        """Build the occlusion mask for the current sliding-window position.

        Unlike plain occlusion, the temporal dimension of the window only walks
        over time steps that cleared the relevance threshold.
        """
        if is_above_threshold is None:
            return super()._occlusion_mask(
                expanded_input=expanded_input,
                ablated_feature_num=ablated_feature_num,
                sliding_window_tsr=sliding_window_tsr,
                strides=strides,
                shift_counts=shift_counts,
            )

        # We first compute the hyper-rectangle on the non-temporal dims
        padded_tensor = super()._occlusion_mask(
            expanded_input=expanded_input[:, :, 0],
            ablated_feature_num=ablated_feature_num,
            sliding_window_tsr=torch.ones(sliding_window_tsr.shape[1:]),
            strides=strides[1:] if isinstance(strides, tuple) else strides,
            shift_counts=shift_counts[1:],
        ).to(expanded_input.device)

        # We get the current index and batch size
        bsz = expanded_input.shape[1]
        shift_count = shift_counts[0]
        stride = strides[0] if isinstance(strides, tuple) else strides
        current_index = (ablated_feature_num % shift_count) * stride

        # On the temporal dim, the hyper-rectangle is only applied on
        # non-zeros elements
        is_above = is_above_threshold.clone()
        for batch_idx in range(bsz):
            nonzero = is_above_threshold[batch_idx].nonzero()[:, 0]
            is_above[
                batch_idx,
                nonzero[
                    current_index : current_index + sliding_window_tsr.shape[0]
                ],
            ] = 0

        return is_above.unsqueeze(-1) * padded_tensor.unsqueeze(0)

    def _run_forward(
        self, forward_func: Callable, inputs: Any, **kwargs
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Tuple[int]]]:
        attributions = self.attribution_method.attribute.__wrapped__(
            self.attribution_method, inputs, **kwargs
        )

        # Check if it needs to return convergence delta
        return_convergence_delta = (
            "return_convergence_delta" in kwargs
            and kwargs["return_convergence_delta"]
        )

        # If the method returns delta, we ignore it
        if self.is_delta_supported and return_convergence_delta:
            attributions, _ = attributions

        # Get attr shapes
        attributions_shape = tuple(tuple(attr.shape) for attr in attributions)

        return attributions, attributions_shape

    @staticmethod
    def _reshape_eval_diff(eval_diff: Tensor, shapes: tuple) -> Tensor:
        # For this method, we need to reshape eval_diff to the output shapes
        return eval_diff.reshape((len(eval_diff),) + shapes)

    @staticmethod
    def get_name() -> str:
        return "WinTSR"
