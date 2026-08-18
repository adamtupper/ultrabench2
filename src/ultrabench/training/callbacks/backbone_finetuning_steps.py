"""Step-based backbone finetuning callback."""

import logging
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from typing_extensions import override

log = logging.getLogger(__name__)


class BackboneFinetuningSteps(BaseFinetuning):
    """Finetune a backbone model based on a user-defined LR schedule.

    Step-based equivalent of
    :class:`lightning.pytorch.callbacks.BackboneFinetuning`. Unfreezes the
    backbone at a specific global step instead of an epoch boundary. Once
    unfrozen, the backbone is split into ``backbone_num_layer_groups``
    contiguous groups (by parameter registration order, shallowest first)
    and each group's LR is computed every step as a fixed ratio of the
    head's current LR, scaled by a per-group layer-decay factor and a
    linear warmup — with no backbone-specific customization required.

    Args:
        unfreeze_backbone_at_step: Global step at which the backbone will be
            unfrozen.
        backbone_max_lr_ratio: Ceiling on the backbone LR, expressed as a
            ratio of the head's current LR. The deepest layer group's LR
            approaches (but never exceeds) ``head_lr * backbone_max_lr_ratio``.
        backbone_warmup_steps: Number of steps after unfreezing over which
            the backbone LR ramps up linearly from 0 to its target ratio.
        backbone_layer_decay: Multiplicative LR decay applied per layer
            group, shallowest first. The deepest group always gets a scale
            of 1.0 (i.e. the full ``backbone_max_lr_ratio``); shallower
            groups get progressively smaller scales. Set to 1.0 to disable
            per-layer decay (all groups share the same LR).
        backbone_num_layer_groups: Number of contiguous parameter groups to
            split the backbone into, ordered shallowest to deepest. Grouping
            is based purely on parameter registration order, so it applies
            to any backbone architecture without customization.
        train_bn: Whether to make Batch Normalization trainable.
        verbose: Display current learning rate for model and backbone.
        rounding: Precision for displaying learning rate.
    """

    def __init__(
        self,
        unfreeze_backbone_at_step: int = 1000,
        backbone_max_lr_ratio: float = 0.1,
        backbone_warmup_steps: int = 500,
        backbone_layer_decay: float = 0.75,
        backbone_num_layer_groups: int = 4,
        train_bn: bool = False,
        verbose: bool = False,
        rounding: int = 12,
    ) -> None:
        super().__init__()
        self.unfreeze_step = unfreeze_backbone_at_step
        self.backbone_max_lr_ratio = backbone_max_lr_ratio
        self.backbone_warmup_steps = backbone_warmup_steps
        self.backbone_layer_decay = backbone_layer_decay
        self.backbone_num_layer_groups = backbone_num_layer_groups
        self.train_bn = train_bn
        self.verbose = verbose
        self.rounding = rounding
        self._unfrozen = False
        self._backbone_group_start: int = 0
        self._layer_scales: list[float] = []

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "unfrozen": self._unfrozen,
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._unfrozen = state_dict["unfrozen"]
        super().load_state_dict(state_dict)

    @override
    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Validate backbone attribute; restore param groups on resume.

        Raises:
            MisconfigurationException: If LightningModule has no nn.Module
                ``backbone`` attribute.
        """
        backbone = getattr(pl_module, "backbone", None)
        if isinstance(backbone, Module):
            super().on_fit_start(trainer, pl_module)
            if self._unfrozen:
                # Step-based deviation: backbone may have been unfrozen
                # mid-epoch before checkpoint. freeze_before_training() in
                # setup() re-froze it and gave it a fresh optimizer with
                # only the head's param group, so rebuild the backbone's
                # layer groups now.
                self._add_backbone_param_groups(trainer.optimizers[0], backbone)
            return
        raise MisconfigurationException(
            "The LightningModule should have a nn.Module `backbone` attribute"
        )

    @override
    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.backbone)

    @override
    def finetune_function(
        self,
        pl_module: "pl.LightningModule",
        epoch: int,
        optimizer: Optimizer,
    ) -> None:
        """No-op. LR scaling is applied per-step in on_train_batch_start."""
        return

    @staticmethod
    def _split_into_buckets(
        params: list[Parameter], num_groups: int
    ) -> list[list[Parameter]]:
        """Split params into num_groups contiguous, near-equal-size chunks.

        Args:
            params: Parameters in shallow-to-deep registration order.
            num_groups: Desired number of chunks.

        Returns:
            List of parameter chunks, shallowest first.
        """
        num_groups = max(1, min(num_groups, len(params)))
        size, remainder = divmod(len(params), num_groups)
        buckets = []
        start = 0
        for i in range(num_groups):
            end = start + size + (1 if i < remainder else 0)
            buckets.append(params[start:end])
            start = end
        return buckets

    def _add_backbone_param_groups(
        self, optimizer: Optimizer, backbone: Module
    ) -> None:
        """Unfreeze the backbone and add one param group per layer bucket.

        Args:
            optimizer: The optimizer to add the new param groups to.
            backbone: The backbone module to unfreeze.
        """
        self.make_trainable(backbone)
        params = list(
            self.filter_params(backbone, train_bn=self.train_bn, requires_grad=True)
        )
        buckets = [
            bucket
            for bucket in self._split_into_buckets(
                params, self.backbone_num_layer_groups
            )
            if bucket
        ]
        self._backbone_group_start = len(optimizer.param_groups)
        self._layer_scales = [
            self.backbone_layer_decay ** (len(buckets) - 1 - j)
            for j in range(len(buckets))
        ]
        for bucket in buckets:
            optimizer.add_param_group({"params": bucket, "lr": 0.0})

    def _update_backbone_lrs(self, trainer: "pl.Trainer", optimizer: Optimizer) -> None:
        """Set each backbone layer group's LR from the head's current LR.

        Args:
            trainer: The Lightning Trainer.
            optimizer: The optimizer holding the head and backbone groups.
        """
        head_lr = optimizer.param_groups[0]["lr"]
        steps_since_unfreeze = trainer.global_step - self.unfreeze_step
        warmup_factor = (
            min(1.0, steps_since_unfreeze / self.backbone_warmup_steps)
            if self.backbone_warmup_steps > 0
            else 1.0
        )
        backbone_groups = optimizer.param_groups[self._backbone_group_start :]
        for group, scale in zip(backbone_groups, self._layer_scales):
            group["lr"] = head_lr * self.backbone_max_lr_ratio * scale * warmup_factor

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Unfreeze the backbone at the target global step, then scale LR.

        Args:
            trainer: The Lightning Trainer.
            pl_module: The LightningModule being trained.
            batch: The current batch.
            batch_idx: The index of the current batch.
        """
        optimizer = trainer.optimizers[0]

        if not self._unfrozen:
            if trainer.global_step < self.unfreeze_step:
                return

            # on_fit_start() already validated that this is a nn.Module.
            backbone = pl_module.backbone
            assert isinstance(backbone, Module)

            num_param_groups = len(optimizer.param_groups)
            self._add_backbone_param_groups(optimizer, backbone)
            self._unfrozen = True
            # Must call _store() here. BaseFinetuning.on_train_epoch_start()
            # already ran for this epoch and captured num_param_groups before
            # unfreeze. At the next epoch start it will see
            # num_param_groups == len(current) and be a no-op — the backbone
            # groups would never be recorded without this.
            self._store(pl_module, 0, num_param_groups, optimizer.param_groups)

        self._update_backbone_lrs(trainer, optimizer)

        if self.verbose:
            backbone_lrs = [
                round(group["lr"], self.rounding)
                for group in optimizer.param_groups[self._backbone_group_start :]
            ]
            log.info(
                "Head lr: %s, Backbone lrs: %s",
                round(optimizer.param_groups[0]["lr"], self.rounding),
                backbone_lrs,
            )
