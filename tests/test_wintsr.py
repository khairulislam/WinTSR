import pytest
import torch
from torch import nn

from tslens import WinTSR, get_baseline

SEQ_LEN, N_FEATURES, PRED_LEN, BATCH = 12, 3, 2, 4


class TinyForecaster(nn.Module):
    """Minimal (batch, seq_len, n_features) -> (batch, pred_len) model."""

    def __init__(self, seq_len=SEQ_LEN, n_features=N_FEATURES, pred_len=PRED_LEN):
        super().__init__()
        self.net = nn.Linear(seq_len * n_features, pred_len)

    def forward(self, x):
        return self.net(x.flatten(1))


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = TinyForecaster()
    m.eval()
    return m


@pytest.fixture
def inputs():
    torch.manual_seed(1)
    return torch.randn(BATCH, SEQ_LEN, N_FEATURES)


def test_accepts_bare_module(model, inputs):
    """WinTSR(model) works without hand-building an inner explainer."""
    attr = WinTSR(model).attribute(inputs, baselines=get_baseline(inputs, "zero"))
    assert attr.shape == (BATCH, PRED_LEN, SEQ_LEN, N_FEATURES)
    assert torch.isfinite(attr).all()
    assert attr.abs().sum() > 0


def test_default_window_matches_explicit(model, inputs):
    """Omitting sliding_window_shapes is the same as passing (1, 1)."""
    baselines = get_baseline(inputs, "zero")
    auto = WinTSR(model).attribute(inputs, baselines=baselines)
    explicit = WinTSR(model).attribute(
        inputs, sliding_window_shapes=(1, 1), baselines=baselines
    )
    assert torch.allclose(auto, explicit)


def test_matches_research_implementation(model, inputs):
    """The packaged method reproduces the original repo's numbers.

    The original is called with legacy normalization and the same arguments the
    research harness used (utils/explainer.py::compute_attr).
    """
    from tint.attr import Occlusion

    from ._legacy_wintsr import WinTSR as ResearchWinTSR

    baselines = get_baseline(inputs, "zero")
    expected = ResearchWinTSR(Occlusion(model)).attribute(
        inputs=inputs,
        sliding_window_shapes=(1, 1),
        baselines=baselines,
        threshold=0.5,
        normalize=True,
        attributions_fn=abs,
    )
    actual = WinTSR(model, legacy_normalize=True).attribute(
        inputs=inputs,
        sliding_window_shapes=(1, 1),
        baselines=baselines,
        threshold=0.5,
        normalize=True,
        unflatten=False,
    )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_unflatten_is_a_pure_reshape(model, inputs):
    """unflatten only splits dim 0; it must not change any value."""
    baselines = get_baseline(inputs, "zero")
    flat = WinTSR(model).attribute(inputs, baselines=baselines, unflatten=False)
    split = WinTSR(model).attribute(inputs, baselines=baselines, unflatten=True)
    assert flat.shape == (BATCH * PRED_LEN, SEQ_LEN, N_FEATURES)
    assert torch.equal(split, flat.reshape(BATCH, PRED_LEN, SEQ_LEN, N_FEATURES))


def test_unflatten_ordering_is_batch_major():
    """Row i of the flat output is batch i // n_output, output i % n_output.

    Verified with a model whose output ``o`` depends only on feature ``o``, so
    the attribution mass has to land on the matching feature.
    """

    class Disentangled(nn.Module):
        def forward(self, x):
            return torch.stack([x[:, :, o].sum(dim=1) for o in range(2)], dim=1)

    torch.manual_seed(2)
    x = torch.randn(3, 6, 2)
    attr = WinTSR(Disentangled().eval()).attribute(x, baselines=torch.zeros_like(x))
    assert attr.shape == (3, 2, 6, 2)

    mass = attr.sum(dim=2)  # (batch, n_output, n_features)
    for b in range(3):
        for o in range(2):
            assert mass[b, o].argmax() == o, (
                f"batch {b} output {o} attributed to the wrong feature"
            )


def test_harness_call_site_is_unchanged(model, inputs):
    """The research harness still gets byte-identical results after the move.

    exp/exp_interpret.py builds the explainer with ``legacy_normalize=True`` and
    utils/explainer.py calls it with ``unflatten=False``; this reproduces that
    exact call and compares against the frozen pre-refactor implementation.
    """
    from tint.attr import Occlusion

    from ._legacy_wintsr import WinTSR as ResearchWinTSR

    baselines = get_baseline(inputs, "zero")
    kwargs = dict(
        inputs=inputs,
        sliding_window_shapes=(1, 1),
        baselines=baselines,
        threshold=0.5,
        normalize=True,
        attributions_fn=abs,
    )
    expected = ResearchWinTSR(Occlusion(model)).attribute(**kwargs)
    actual = WinTSR(Occlusion(model), legacy_normalize=True).attribute(
        unflatten=False, **kwargs
    )
    assert actual.shape == expected.shape == (BATCH * PRED_LEN, SEQ_LEN, N_FEATURES)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_baselines_are_lazily_importable():
    """Baseline methods are exposed but not imported until requested."""
    import tslens.attr as attr_pkg

    assert "TSR" in dir(attr_pkg)
    with pytest.raises(AttributeError):
        attr_pkg.NotAMethod


def test_all_methods_import_without_the_research_harness(tmp_path):
    """A bare `pip install tslens` must expose all four methods.

    Runs in a subprocess from an unrelated working directory, so the harness
    packages under research/ are not importable -- exactly the situation of a
    user who ran `pip install tslens` and never cloned the repo.
    """
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent(
        """
        from tslens.attr import WinTSR, TSR, WinIT, GateMask
        from tslens.attr.tsr import DUAL_INPUT_USERS

        assert "iTransformer" in DUAL_INPUT_USERS

        for harness in ("exp", "utils", "models", "layers"):
            try:
                __import__(harness)
            except ImportError:
                pass
            else:
                raise AssertionError(harness + " leaked onto sys.path")
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert "OK" in result.stdout, (
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


def test_tsr_dual_input_users_is_overridable():
    """Callers can name their own dual-input models."""
    from tslens.attr import TSR
    from tslens.attr.tsr import DUAL_INPUT_USERS

    class Args:
        model = "MyCustomModel"
        task_name = "long_term_forecast"
        pred_len = PRED_LEN

    default = TSR(TinyForecaster().eval(), Args())
    assert default.dual_input_users is DUAL_INPUT_USERS

    custom = TSR(
        TinyForecaster().eval(), Args(), dual_input_users=["MyCustomModel"]
    )
    assert custom.dual_input_users == ["MyCustomModel"]


def test_threshold_sparsifies(model, inputs):
    """A higher threshold skips low-relevance time steps, zeroing them out."""
    baselines = get_baseline(inputs, "zero")
    dense = WinTSR(model).attribute(inputs, baselines=baselines, threshold=0.0)
    sparse = WinTSR(model).attribute(inputs, baselines=baselines, threshold=0.8)
    assert (sparse == 0).sum() > (dense == 0).sum()


def test_tuple_inputs(model, inputs):
    """Multi-input models get a tuple of attributions back."""

    class TwoInput(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(SEQ_LEN * N_FEATURES * 2, PRED_LEN)

        def forward(self, x, x_mark):
            return self.net(torch.cat([x.flatten(1), x_mark.flatten(1)], dim=1))

    two = TwoInput().eval()
    pair = (inputs, torch.randn_like(inputs))
    attr = WinTSR(two).attribute(pair, baselines=get_baseline(pair, "zero"))
    assert isinstance(attr, tuple) and len(attr) == 2
    assert all(a.shape == (BATCH, PRED_LEN, SEQ_LEN, N_FEATURES) for a in attr)


def test_recipe_dict_output(model, inputs):
    """Cookbook: wrap a model that returns a dict."""

    class DictOut(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = TinyForecaster()

        def forward(self, x):
            return {"outputs_time": self.inner(x), "aux": x.sum()}

    raw = DictOut().eval()
    attr = WinTSR(lambda t: raw(t)["outputs_time"]).attribute(
        inputs, baselines=get_baseline(inputs, "zero")
    )
    assert attr.shape == (BATCH, PRED_LEN, SEQ_LEN, N_FEATURES)


def test_recipe_tuple_output(model, inputs):
    """Cookbook: wrap a model that returns (predictions, extras)."""

    class TupleOut(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = TinyForecaster()

        def forward(self, x):
            return self.inner(x), "attention"

    raw = TupleOut().eval()
    attr = WinTSR(lambda t: raw(t)[0]).attribute(
        inputs, baselines=get_baseline(inputs, "zero")
    )
    assert attr.shape == (BATCH, PRED_LEN, SEQ_LEN, N_FEATURES)


def test_recipe_classification_with_padding_mask(inputs):
    """Cookbook: extra model args go in additional_forward_args."""
    n_classes = 3

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(SEQ_LEN * N_FEATURES, n_classes)

        def forward(self, x, padding_mask):
            return self.net((x * padding_mask.unsqueeze(-1)).flatten(1))

    attr = WinTSR(Classifier().eval()).attribute(
        inputs,
        baselines=get_baseline(inputs, "zero"),
        additional_forward_args=(torch.ones(BATCH, SEQ_LEN),),
    )
    assert attr.shape == (BATCH, n_classes, SEQ_LEN, N_FEATURES)


@pytest.mark.parametrize("mode", ["zero", "random", "normal", "mean"])
def test_recipe_baseline_modes(inputs, mode):
    """Cookbook: every documented baseline mode is real and shape-correct."""
    baseline = get_baseline(inputs, mode)
    assert baseline.shape == inputs.shape
    assert torch.isfinite(baseline).all()


def test_perturbations_per_eval_error_is_actionable(model, inputs):
    """Multi-output + batched perturbations raises a message that helps."""
    with pytest.raises(ValueError, match="single-output"):
        WinTSR(model).attribute(
            inputs, baselines=get_baseline(inputs, "zero"), perturbations_per_eval=4
        )


def test_perturbations_per_eval_works_for_single_output(inputs):
    """The limitation is genuinely confined to multi-output models."""

    class SingleOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(SEQ_LEN * N_FEATURES, 1)

        def forward(self, x):
            return self.net(x.flatten(1))

    attr = WinTSR(SingleOutput().eval()).attribute(
        inputs, baselines=get_baseline(inputs, "zero"), perturbations_per_eval=4
    )
    assert attr.shape == (BATCH, 1, SEQ_LEN, N_FEATURES)


def test_legacy_flag_changes_result(model, inputs):
    """The normalization fix is a real numerical change, not a no-op."""
    baselines = get_baseline(inputs, "zero")
    fixed = WinTSR(model, legacy_normalize=False).attribute(
        inputs, baselines=baselines
    )
    legacy = WinTSR(model, legacy_normalize=True).attribute(
        inputs, baselines=baselines
    )
    assert not torch.allclose(fixed, legacy)
