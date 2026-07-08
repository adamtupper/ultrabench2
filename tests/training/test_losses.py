"""Unit tests for training loss functions."""

import pytest
import torch

from ultrabench.training.losses import DiceCELoss, MonaiDiceCELoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-seed [B=2, C=2, H=64, W=64] logits and [B, H, W] target."""
    torch.manual_seed(42)
    pred = torch.randn(2, 2, 64, 64)
    target = torch.randint(0, 2, (2, 64, 64))
    return pred, target


@pytest.fixture
def multiclass_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-seed [B=2, C=4, H=64, W=64] logits and [B, H, W] target."""
    torch.manual_seed(42)
    pred = torch.randn(2, 4, 64, 64)
    target = torch.randint(0, 4, (2, 64, 64))
    return pred, target


# ---------------------------------------------------------------------------
# DiceCELoss
# ---------------------------------------------------------------------------


class TestDiceCELoss:
    def test_binary_output_is_scalar_float(
        self, binary_inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        pred, target = binary_inputs
        loss = DiceCELoss()
        out = loss(pred, target)
        assert out.shape == torch.Size([])
        assert out.dtype == torch.float32

    def test_multiclass_output_is_scalar_float(
        self, multiclass_inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        pred, target = multiclass_inputs
        loss = DiceCELoss()
        out = loss(pred, target)
        assert out.shape == torch.Size([])
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# MonaiDiceCELoss
# ---------------------------------------------------------------------------


class TestMonaiDiceCELoss:
    def test_binary_output_is_scalar_float(
        self, binary_inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        pytest.importorskip("monai", reason="monai not installed")
        pred, target = binary_inputs
        loss = MonaiDiceCELoss()
        out = loss(pred, target)
        assert out.shape == torch.Size([])
        assert out.dtype == torch.float32

    def test_multiclass_output_is_scalar_float(
        self, multiclass_inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        pytest.importorskip("monai", reason="monai not installed")
        pred, target = multiclass_inputs
        loss = MonaiDiceCELoss()
        out = loss(pred, target)
        assert out.shape == torch.Size([])
        assert out.dtype == torch.float32

    def test_binary_matches_custom_loss(
        self, binary_inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        pytest.importorskip("monai", reason="monai not installed")
        pred, target = binary_inputs
        custom = DiceCELoss()
        monai_loss = MonaiDiceCELoss()
        assert torch.allclose(custom(pred, target), monai_loss(pred, target), atol=1e-3)

    def test_multiclass_matches_custom_loss(
        self, multiclass_inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        pytest.importorskip("monai", reason="monai not installed")
        pred, target = multiclass_inputs
        custom = DiceCELoss()
        monai_loss = MonaiDiceCELoss()
        assert torch.allclose(
            custom(pred, target), monai_loss(pred, target), atol=1e-3
        )
