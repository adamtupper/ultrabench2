"""Unit tests for the OpenUS VMamba cross-scan/cross-merge and scan ops.

These ops have accelerated implementations (triton for cross-scan/merge, CUDA
extensions for the selective scan) and pure-PyTorch fallbacks. The fallbacks
are what run here and on the compute cluster, so they must match the
accelerated paths' shapes and the reference recurrence's values.
"""

import pytest
import torch
import torch.nn.functional as F

from ultrabench.models.openus.openus_vmamba import (
    _selective_scan_torch,
    cross_merge_fn,
    cross_scan_fn,
)

_BATCH = 2
_GROUPS = 4
_CHANNELS = 8
_HEIGHT = 7
_WIDTH = 5


def _reference_selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
    oflex: bool = True,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Step the selective-scan recurrence one position at a time.

    This is the original, obviously-correct implementation that
    `_selective_scan_torch` replaced with a chunked equivalent. It is kept
    here as the oracle: the chunked version must reproduce it.

    Args:
        u: Input sequence, shape (batch, groups * dim, seqlen).
        delta: Timestep sizes, same shape as u.
        A: State transition matrix, shape (groups * dim, state).
        B: Input projection, shape (batch, groups, state, seqlen).
        C: Output projection, shape (batch, groups, state, seqlen).
        D: Optional skip-connection weights, shape (groups * dim,).
        delta_bias: Optional bias added to delta, shape (groups * dim,).
        delta_softplus: Whether to apply softplus to delta after the bias.
        oflex: If True return float32, otherwise cast back to u's dtype.

    Returns:
        The scan output, shape (batch, groups * dim, seqlen).
    """
    dtype_in = u.dtype
    batch, k, n, length = B.shape
    kc_dim = u.shape[1]
    c_dim = int(kc_dim / k)
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(batch, k, 1, n, length).repeat(1, 1, c_dim, 1, 1)
    B = B.view(batch, kc_dim, n, length)
    C = C.view(batch, k, 1, n, length).repeat(1, 1, c_dim, 1, 1)
    C = C.view(batch, kc_dim, n, length)
    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    delta_b_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    x = A.new_zeros((batch, kc_dim, n))
    ys = []
    for i in range(length):
        x = delta_a[:, :, i] * x + delta_b_u[:, :, i]
        ys.append(torch.einsum("bdn,bdn->bd", x, C[:, :, :, i]))
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


_ScanInputs = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _scan_inputs(length: int, state: int = 1, seed: int = 0) -> _ScanInputs:
    """Build a set of selective-scan inputs with model-realistic magnitudes.

    Args:
        length: Sequence length.
        state: SSM state dimension.
        seed: Seed for the random generator.

    Returns:
        The tensors (u, delta, A, B, C, D, delta_bias).
    """
    g = torch.Generator().manual_seed(seed)
    c_dim = 3
    kc = _GROUPS * c_dim

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    return (
        rand(_BATCH, kc, length),
        # delta is a softplus away from being the model's timestep, which sits
        # well below 1; large values here would not be representative.
        rand(_BATCH, kc, length) * 0.1 - 2.0,
        -torch.rand(kc, state, generator=g),
        rand(_BATCH, _GROUPS, state, length),
        rand(_BATCH, _GROUPS, state, length),
        rand(kc),
        rand(kc) * 0.1,
    )


class TestCrossScanBackward:
    """Tests for the pure-PyTorch cross-scan backward pass."""

    def test_gradient_shape_channel_first(self) -> None:
        x = torch.randn(_BATCH, _CHANNELS, _HEIGHT, _WIDTH, requires_grad=True)
        cross_scan_fn(x, force_torch=True).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_shape_channel_last(self) -> None:
        x = torch.randn(_BATCH, _HEIGHT, _WIDTH, _CHANNELS, requires_grad=True)
        cross_scan_fn(
            x, in_channel_first=False, out_channel_first=False, force_torch=True
        ).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestCrossMergeBackward:
    """Tests for the pure-PyTorch cross-merge backward pass."""

    def test_gradient_shape_channel_first(self) -> None:
        ys = torch.randn(
            _BATCH, _GROUPS, _CHANNELS, _HEIGHT, _WIDTH, requires_grad=True
        )
        cross_merge_fn(ys, force_torch=True).sum().backward()
        assert ys.grad is not None
        assert ys.grad.shape == ys.shape

    def test_gradient_shape_channel_last(self) -> None:
        ys = torch.randn(
            _BATCH, _HEIGHT, _WIDTH, _GROUPS, _CHANNELS, requires_grad=True
        )
        cross_merge_fn(
            ys, in_channel_first=False, out_channel_first=False, force_torch=True
        ).sum().backward()
        assert ys.grad is not None
        assert ys.grad.shape == ys.shape

    def test_gradient_values_match_cross_scan(self) -> None:
        """Cross-merge is linear, so its backward is the cross-scan forward."""
        ys = torch.randn(
            _BATCH,
            _GROUPS,
            _CHANNELS,
            _HEIGHT,
            _WIDTH,
            dtype=torch.double,
            requires_grad=True,
        )
        grad_out = torch.randn(_BATCH, _CHANNELS, _HEIGHT * _WIDTH, dtype=torch.double)
        cross_merge_fn(ys, force_torch=True).backward(grad_out)

        expected = cross_scan_fn(
            grad_out.view(_BATCH, _CHANNELS, _HEIGHT, _WIDTH), force_torch=True
        ).view(_BATCH, _GROUPS, _CHANNELS, _HEIGHT, _WIDTH)
        assert ys.grad is not None
        assert torch.allclose(ys.grad, expected)

    def test_backward_through_scan_merge_roundtrip(self) -> None:
        """Full-finetuning backprops through a scan followed by a merge."""
        x = torch.randn(_BATCH, _CHANNELS, _HEIGHT, _WIDTH, requires_grad=True)
        xs = cross_scan_fn(x, force_torch=True)
        ys = xs.view(_BATCH, _GROUPS, _CHANNELS, _HEIGHT, _WIDTH)
        cross_merge_fn(ys, force_torch=True).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestSelectiveScanForward:
    """The chunked selective scan must match the sequential recurrence."""

    # 63/64/65 and 196 straddle the 64-wide chunk boundary; 1 is degenerate.
    @pytest.mark.parametrize("length", [1, 7, 49, 63, 64, 65, 196])
    def test_matches_reference_across_chunk_boundaries(self, length: int) -> None:
        args = _scan_inputs(length)
        expected = _reference_selective_scan(*args)
        actual = _selective_scan_torch(*args)
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("state", [1, 4])
    def test_matches_reference_for_multiple_state_dims(self, state: int) -> None:
        """The model uses state=1; state>1 guards against specialising."""
        args = _scan_inputs(96, state=state)
        expected = _reference_selective_scan(*args)
        actual = _selective_scan_torch(*args)
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("with_d", [True, False])
    @pytest.mark.parametrize("with_bias", [True, False])
    @pytest.mark.parametrize("softplus", [True, False])
    def test_matches_reference_for_optional_args(
        self, with_d: bool, with_bias: bool, softplus: bool
    ) -> None:
        u, delta, A, B, C, D, bias = _scan_inputs(96)
        d_arg = D if with_d else None
        bias_arg = bias if with_bias else None
        # delta is a timestep and must be positive; softplus is how the model
        # guarantees that, so pre-apply it when the flag is off.
        delta_arg = delta if softplus else F.softplus(delta)
        expected = _reference_selective_scan(
            u, delta_arg, A, B, C, d_arg, bias_arg, softplus
        )
        actual = _selective_scan_torch(u, delta_arg, A, B, C, d_arg, bias_arg, softplus)
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def test_output_is_finite_for_steep_decay(self) -> None:
        """Large delta underflows exp(); the exponent floor must stop NaN."""
        u, delta, A, B, C, D, bias = _scan_inputs(196)
        actual = _selective_scan_torch(u, delta * 0.0 + 3.0, A, B, C, D, bias)
        assert torch.isfinite(actual).all()


class TestSelectiveScanDtype:
    """The oflex flag controls whether the float32 result is cast back."""

    def test_oflex_true_returns_float32(self) -> None:
        u, delta, A, B, C, D, bias = _scan_inputs(96)
        out = _selective_scan_torch(
            u.to(torch.bfloat16), delta, A, B, C, D, bias, True, True
        )
        assert out.dtype == torch.float32

    def test_oflex_false_casts_back_to_input_dtype(self) -> None:
        u, delta, A, B, C, D, bias = _scan_inputs(96)
        out = _selective_scan_torch(
            u.to(torch.bfloat16), delta, A, B, C, D, bias, True, False
        )
        assert out.dtype == torch.bfloat16


class TestSelectiveScanBackward:
    """Gradients must match the sequential recurrence for every input."""

    def test_gradients_match_reference(self) -> None:
        base = _scan_inputs(196)

        def grads_of(scan_fn) -> list[torch.Tensor | None]:
            u, delta, A, B, C, D, bias = (t.clone().requires_grad_(True) for t in base)
            args = [u, delta, A, B, C, D, bias]
            scan_fn(u, delta, A, B, C, D, bias).square().mean().backward()
            return [t.grad for t in args]

        for name, expected, actual in zip(
            ["u", "delta", "A", "B", "C", "D", "delta_bias"],
            grads_of(_reference_selective_scan),
            grads_of(_selective_scan_torch),
        ):
            assert expected is not None and actual is not None
            assert torch.allclose(actual, expected, rtol=1e-3, atol=1e-5), name
