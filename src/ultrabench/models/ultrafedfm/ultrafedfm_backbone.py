"""UltraFedFMBackbone: Backbone wrapper for the UltraFedFM encoder (stub)."""

from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms

from ultrabench.models.backbone import Backbone, BackboneOutput


class UltraFedFMBackbone(Backbone):
    """Backbone wrapper for the UltraFedFM encoder.

    TODO: Fill in architecture details and complete the implementation.

    Attributes:
        feature_dim: TODO
        output_type: TODO
        input_size: TODO
        input_mean: TODO
        input_std: TODO
    """

    feature_dim: int = -1  # TODO
    output_type: str = "patch_tokens"  # TODO
    input_size: int = 224  # TODO
    input_mean: list[float] = [0.485, 0.456, 0.406]  # TODO
    input_std: list[float] = [0.229, 0.224, 0.225]  # TODO

    def __init__(
        self, checkpoint_path: str, output_intermediate_features: bool
    ) -> None:
        """Initialise UltraFedFMBackbone.

        Args:
            checkpoint_path: Path to the UltraFedFM checkpoint.
            output_intermediate_features: Whether to collect intermediate
                encoder features in BackboneOutput.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
            NotImplementedError: This stub has not been implemented yet.
        """
        raise NotImplementedError("UltraFedFMBackbone is not yet implemented.")

    def _load_checkpoint(self) -> None:
        """Load encoder weights from checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
        """
        raise NotImplementedError

    def _resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize, convert to float32, and normalise input images.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Processed float32 tensor [B, C, input_size, input_size].
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor, pool: bool = False) -> BackboneOutput:
        """Extract features from input images.

        Args:
            x: uint8 input image tensor [B, C, H, W].
            pool: If True, mean-pool spatial/sequence features to [B, D].

        Returns:
            BackboneOutput with features and optionally intermediate features.
        """
        raise NotImplementedError
