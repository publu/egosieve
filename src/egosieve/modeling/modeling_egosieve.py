"""PyTorch implementation of the EgoSieve video-readiness model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Dinov2Model, PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_egosieve import EgoSieveConfig


@dataclass
class EgoSieveModelOutput(ModelOutput):
    """Outputs from :class:`EgoSieveModel`.

    Attributes:
        loss: Weighted sum of the supplied task losses.
        logits: Standard Hugging Face video-classification logits in
            ``KEEP, REVIEW, REJECT`` order, with shape ``[batch, 3]``.
        issue_logits: Clip-level multi-label logits in
            :data:`egosieve.ISSUE_LABELS` order, with shape ``[batch, 8]``.
        boundary_logits: Per-frame ``start, end`` logits with shape
            ``[batch, frames, 2]``. Masked-frame entries are zero.
        clip_embedding: L2-normalized clip representation with shape
            ``[batch, projection_dim]``.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    issue_logits: torch.FloatTensor | None = None
    boundary_logits: torch.FloatTensor | None = None
    clip_embedding: torch.FloatTensor | None = None

    @property
    def readiness_logits(self) -> torch.FloatTensor | None:
        """Backwards-compatible descriptive alias for :attr:`logits`."""

        return self.logits


class _GatedTemporalMixer(nn.Module):
    """A mask-aware local temporal mixer followed by a channel mixer."""

    def __init__(self, config: EgoSieveConfig, dilation: int) -> None:
        super().__init__()
        hidden_size = config.temporal_hidden_size
        padding = dilation * (config.temporal_kernel_size - 1) // 2

        self.temporal_norm = nn.LayerNorm(hidden_size)
        self.depthwise_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=config.temporal_kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hidden_size,
        )
        self.temporal_gate = nn.Linear(hidden_size, hidden_size * 2)

        self.channel_norm = nn.LayerNorm(hidden_size)
        self.channel_in = nn.Linear(hidden_size, config.temporal_intermediate_size * 2)
        self.channel_out = nn.Linear(config.temporal_intermediate_size, hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        # Masking both before and after the convolution ensures that arbitrary
        # values in padding frames cannot influence neighboring valid frames.
        mask = frame_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        residual = hidden_states * mask
        mixed = self.temporal_norm(residual) * mask
        mixed = self.depthwise_conv(mixed.transpose(1, 2)).transpose(1, 2)
        values, gates = self.temporal_gate(mixed).chunk(2, dim=-1)
        mixed = F.gelu(values) * torch.sigmoid(gates)
        hidden_states = (residual + self.dropout(mixed)) * mask

        residual = hidden_states
        mixed = self.channel_norm(hidden_states) * mask
        values, gates = self.channel_in(mixed).chunk(2, dim=-1)
        mixed = F.gelu(values) * torch.sigmoid(gates)
        mixed = self.channel_out(self.dropout(mixed))
        return (residual + self.dropout(mixed)) * mask


class EgoSievePreTrainedModel(PreTrainedModel):
    """Base class carrying EgoSieve initialization and config metadata."""

    config_class = EgoSieveConfig
    base_model_prefix = "egosieve"
    main_input_name = "pixel_values"

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class EgoSieveModel(EgoSievePreTrainedModel):
    """A DINOv2-based, mask-aware model for video-dataset readiness.

    Frames are independently encoded by DINOv2.  A compact stack of dilated,
    gated depthwise temporal mixers then supplies clip classification,
    multi-label issue detection, per-frame boundary detection, and a normalized
    clip embedding.

    For fast tests and feature-caching workflows, callers may pass
    ``frame_embeddings`` with shape ``[B, T, vision_config.hidden_size]`` in
    place of ``pixel_values``.  Exactly one of those inputs must be supplied.
    """

    def __init__(self, config: EgoSieveConfig) -> None:
        super().__init__(config)
        self.vision_model = Dinov2Model(config.vision_config)
        self.vision_projection = nn.Linear(
            config.vision_config.hidden_size,
            config.temporal_hidden_size,
        )
        self.position_embeddings = nn.Embedding(config.max_frames, config.temporal_hidden_size)
        self.input_norm = nn.LayerNorm(config.temporal_hidden_size)
        self.input_dropout = nn.Dropout(config.dropout)

        self.temporal_encoder = nn.ModuleList(
            _GatedTemporalMixer(config, dilation=2 ** (layer_index % 3))
            for layer_index in range(config.temporal_num_layers)
        )
        self.output_norm = nn.LayerNorm(config.temporal_hidden_size)
        self.pool_score = nn.Linear(config.temporal_hidden_size, 1, bias=False)
        self.head_dropout = nn.Dropout(config.dropout)

        self.readiness_classifier = nn.Linear(
            config.temporal_hidden_size,
            config.num_readiness_labels,
        )
        self.issue_classifier = nn.Linear(
            config.temporal_hidden_size,
            config.num_issue_labels,
        )
        self.boundary_classifier = nn.Linear(
            config.temporal_hidden_size,
            config.num_boundary_labels,
        )
        self.clip_projection = nn.Linear(
            config.temporal_hidden_size,
            config.projection_dim,
            bias=False,
        )

        if config.issue_pos_weight is None:
            issue_pos_weight = None
        else:
            issue_pos_weight = torch.tensor(config.issue_pos_weight, dtype=torch.float32)
        self.register_buffer("issue_pos_weight", issue_pos_weight, persistent=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Return the DINOv2 patch projection used for input pixels."""

        return self.vision_model.embeddings.patch_embeddings.projection

    def get_vision_backbone(self) -> Dinov2Model:
        """Return the underlying DINOv2 encoder."""

        return self.vision_model

    @staticmethod
    def _prepare_frame_mask(
        frame_mask: torch.Tensor | None,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        if frame_mask is None:
            return torch.ones((batch_size, num_frames), dtype=torch.bool, device=device)
        if tuple(frame_mask.shape) != (batch_size, num_frames):
            raise ValueError(
                "frame_mask must have shape [batch, frames]; expected "
                f"{(batch_size, num_frames)}, received {tuple(frame_mask.shape)}."
            )
        return frame_mask.to(device=device, dtype=torch.bool)

    @staticmethod
    def _broadcast_label_mask(
        label_mask: torch.Tensor | None,
        target: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if label_mask is None:
            return torch.ones_like(target, dtype=torch.bool)

        label_mask = label_mask.to(device=target.device, dtype=torch.bool)
        # A sample/frame mask conventionally omits the final class axis.
        if label_mask.ndim + 1 == target.ndim and tuple(label_mask.shape) == tuple(
            target.shape[:-1]
        ):
            label_mask = label_mask.unsqueeze(-1)
        elif label_mask.ndim == 1 and target.ndim > 1 and label_mask.shape[0] == target.shape[0]:
            label_mask = label_mask.reshape(target.shape[0], *([1] * (target.ndim - 1)))
        try:
            return torch.broadcast_to(label_mask, target.shape)
        except RuntimeError as error:
            raise ValueError(
                f"{name} with shape {tuple(label_mask.shape)} cannot be broadcast to "
                f"label shape {tuple(target.shape)}."
            ) from error

    def _encode_frames(
        self,
        pixel_values: torch.Tensor | None,
        frame_embeddings: torch.Tensor | None,
        frame_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (pixel_values is None) == (frame_embeddings is None):
            raise ValueError("Pass exactly one of pixel_values or frame_embeddings.")

        if frame_embeddings is not None:
            if frame_embeddings.ndim != 3:
                raise ValueError(
                    "frame_embeddings must have shape [batch, frames, hidden_size]; "
                    f"received {tuple(frame_embeddings.shape)}."
                )
            batch_size, num_frames, feature_size = frame_embeddings.shape
            if feature_size != self.config.vision_config.hidden_size:
                raise ValueError(
                    "frame_embeddings last dimension must equal vision_config.hidden_size "
                    f"({self.config.vision_config.hidden_size}); received {feature_size}."
                )
            mask = self._prepare_frame_mask(
                frame_mask,
                batch_size,
                num_frames,
                frame_embeddings.device,
            )
            features = frame_embeddings
        else:
            if pixel_values.ndim != 5:
                raise ValueError(
                    "pixel_values must have shape [batch, frames, channels, height, width]; "
                    f"received {tuple(pixel_values.shape)}."
                )
            batch_size, num_frames, channels, height, width = pixel_values.shape
            if num_frames != self.config.num_frames:
                raise ValueError(
                    "pixel_values frame dimension must match checkpoint num_frames; "
                    f"expected {self.config.num_frames}, received {num_frames}."
                )
            expected_channels = self.config.vision_config.num_channels
            if channels != expected_channels:
                raise ValueError(
                    f"pixel_values must contain {expected_channels} channels; received {channels}."
                )
            mask = self._prepare_frame_mask(
                frame_mask,
                batch_size,
                num_frames,
                pixel_values.device,
            )
            # Sanitizing before the vision encoder makes masked-frame invariance
            # exact, including when padding contains non-finite values.
            visible = mask[:, :, None, None, None]
            pixels = torch.where(visible, pixel_values, torch.zeros_like(pixel_values))
            flat_pixels = pixels.reshape(batch_size * num_frames, channels, height, width)
            vision_output = self.vision_model(pixel_values=flat_pixels, return_dict=True)
            features = vision_output.last_hidden_state[:, 0]
            features = features.reshape(batch_size, num_frames, -1)

        features = torch.where(mask.unsqueeze(-1), features, torch.zeros_like(features))
        return features, mask

    def _readiness_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if tuple(labels.shape) != (logits.shape[0],):
            raise ValueError(
                "readiness_labels must have shape [batch]; expected "
                f"{(logits.shape[0],)}, received {tuple(labels.shape)}."
            )
        labels = labels.to(device=logits.device)
        valid = labels.ne(self.config.ignore_index)
        if label_mask is not None:
            if tuple(label_mask.shape) != tuple(labels.shape):
                raise ValueError("readiness_label_mask must have shape [batch].")
            valid = valid & label_mask.to(device=logits.device, dtype=torch.bool)
        if torch.any(valid):
            return F.cross_entropy(logits[valid], labels[valid].long())
        return logits.sum() * 0.0

    def _binary_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor | None,
        name: str,
        pos_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tuple(labels.shape) != tuple(logits.shape):
            raise ValueError(
                f"{name} must have shape {tuple(logits.shape)}; received {tuple(labels.shape)}."
            )
        labels = labels.to(device=logits.device, dtype=logits.dtype)
        valid = torch.isfinite(labels) & labels.ne(float(self.config.ignore_index))
        valid = valid & self._broadcast_label_mask(label_mask, labels, f"{name[:-1]}_mask")
        safe_labels = torch.where(valid, labels, torch.zeros_like(labels))
        element_loss = F.binary_cross_entropy_with_logits(
            logits,
            safe_labels,
            pos_weight=None if pos_weight is None else pos_weight.to(dtype=logits.dtype),
            reduction="none",
        )
        if torch.any(valid):
            valid_weight = valid.to(dtype=element_loss.dtype)
            return (element_loss * valid_weight).sum() / valid_weight.sum()
        return logits.sum() * 0.0

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        frame_embeddings: torch.Tensor | None = None,
        readiness_labels: torch.Tensor | None = None,
        issue_labels: torch.Tensor | None = None,
        boundary_labels: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        readiness_label_mask: torch.Tensor | None = None,
        issue_label_mask: torch.Tensor | None = None,
        boundary_label_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> EgoSieveModelOutput | tuple[torch.Tensor, ...]:
        """Run inference and optionally calculate one or more task losses.

        ``labels`` is a Transformers-friendly alias for ``readiness_labels``.
        Multi-label targets may use ``ignore_index`` or ``NaN`` per element;
        the explicit ``*_label_mask`` arguments provide additional boolean
        masking.  ``frame_mask`` is always applied to boundary loss.
        """

        if labels is not None:
            if readiness_labels is not None:
                raise ValueError("Pass only one of labels or readiness_labels.")
            readiness_labels = labels
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        frame_features, mask = self._encode_frames(pixel_values, frame_embeddings, frame_mask)
        batch_size, num_frames, _ = frame_features.shape
        if num_frames == 0:
            raise ValueError("Videos must contain at least one frame.")
        if num_frames > self.config.max_frames:
            raise ValueError(
                f"Received {num_frames} frames, exceeding max_frames={self.config.max_frames}."
            )

        hidden_states = self.vision_projection(frame_features)
        positions = torch.arange(num_frames, device=hidden_states.device)
        hidden_states = hidden_states + self.position_embeddings(positions).unsqueeze(0)
        hidden_states = self.input_dropout(self.input_norm(hidden_states))
        hidden_states = hidden_states * mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        for temporal_layer in self.temporal_encoder:
            hidden_states = temporal_layer(hidden_states, mask)
        hidden_states = self.output_norm(hidden_states)
        hidden_states = hidden_states * mask.unsqueeze(-1).to(dtype=hidden_states.dtype)

        # Learned attention pooling with an explicit renormalization handles an
        # all-masked sample without producing NaNs.
        pool_logits = self.pool_score(hidden_states).squeeze(-1)
        minimum = torch.finfo(pool_logits.dtype).min
        pool_weights = torch.softmax(pool_logits.masked_fill(~mask, minimum), dim=1)
        pool_weights = pool_weights * mask.to(dtype=pool_weights.dtype)
        pool_weights = pool_weights / pool_weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(pool_weights.dtype).tiny
        )
        pooled = torch.sum(hidden_states * pool_weights.unsqueeze(-1), dim=1)
        head_input = self.head_dropout(pooled)

        readiness_logits = self.readiness_classifier(head_input)
        issue_logits = self.issue_classifier(head_input)
        boundary_logits = self.boundary_classifier(self.head_dropout(hidden_states))
        boundary_logits = boundary_logits * mask.unsqueeze(-1).to(dtype=boundary_logits.dtype)
        clip_embedding = F.normalize(self.clip_projection(pooled), p=2, dim=-1, eps=1e-12)

        weighted_losses = []
        if readiness_labels is not None:
            weighted_losses.append(
                self.config.readiness_loss_weight
                * self._readiness_loss(readiness_logits, readiness_labels, readiness_label_mask)
            )
        if issue_labels is not None:
            weighted_losses.append(
                self.config.issue_loss_weight
                * self._binary_loss(
                    issue_logits,
                    issue_labels,
                    issue_label_mask,
                    "issue_labels",
                    self.issue_pos_weight,
                )
            )
        if boundary_labels is not None:
            frame_boundary_mask = mask.unsqueeze(-1).expand_as(boundary_logits)
            if boundary_label_mask is None:
                effective_boundary_mask = frame_boundary_mask
            else:
                expanded = self._broadcast_label_mask(
                    boundary_label_mask,
                    boundary_logits,
                    "boundary_label_mask",
                )
                effective_boundary_mask = expanded & frame_boundary_mask
            weighted_losses.append(
                self.config.boundary_loss_weight
                * self._binary_loss(
                    boundary_logits,
                    boundary_labels,
                    effective_boundary_mask,
                    "boundary_labels",
                )
            )
        loss = None if not weighted_losses else sum(weighted_losses[1:], weighted_losses[0])

        output = EgoSieveModelOutput(
            loss=loss,
            logits=readiness_logits,
            issue_logits=issue_logits,
            boundary_logits=boundary_logits,
            clip_embedding=clip_embedding,
        )
        return output if return_dict else output.to_tuple()


__all__ = ["EgoSieveModel", "EgoSieveModelOutput", "EgoSievePreTrainedModel"]
