"""Image-sequence processor for EgoSieve checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature

DEFAULT_IMAGE_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGE_STD = (0.229, 0.224, 0.225)


def _is_frame(value: Any) -> bool:
    if isinstance(value, (Image.Image, str, PathLike)):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim in (2, 3)
    try:
        import torch

        return isinstance(value, torch.Tensor) and value.ndim in (2, 3)
    except ImportError:
        return False


class EgoSieveProcessor(BaseImageProcessor):
    """Prepare one or more frame sequences for :class:`EgoSieveModel`.

    Video decoding intentionally lives outside this class. Passing decoded
    frames keeps timestamp selection explicit and makes the processor useful
    with PyAV, decord, camera streams, or the bundled ffmpeg sampler.
    """

    model_input_names = ["pixel_values", "frame_mask"]

    def __init__(
        self,
        size: int | dict[str, int] = 224,
        resize_shortest_edge: int | None = 256,
        num_frames: int = 12,
        image_mean: Sequence[float] = DEFAULT_IMAGE_MEAN,
        image_std: Sequence[float] = DEFAULT_IMAGE_STD,
        resample: int = Image.Resampling.BICUBIC,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(size, dict):
            height = int(size.get("height", size.get("shortest_edge", 224)))
            width = int(size.get("width", height))
        else:
            height = width = int(size)
        if height <= 0 or width <= 0:
            raise ValueError("size must be positive")
        if resize_shortest_edge is not None and (
            isinstance(resize_shortest_edge, bool)
            or not isinstance(resize_shortest_edge, int)
            or resize_shortest_edge <= 0
        ):
            raise ValueError("resize_shortest_edge must be a positive integer or None")
        if isinstance(num_frames, bool) or not isinstance(num_frames, int) or num_frames <= 0:
            raise ValueError("num_frames must be a positive integer")
        self.size = {"height": height, "width": width}
        self.resize_shortest_edge = resize_shortest_edge
        self.num_frames = int(num_frames)
        self.image_mean = [float(v) for v in image_mean]
        self.image_std = [float(v) for v in image_std]
        self.resample = int(resample)

    @staticmethod
    def _as_pil(frame: Any) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame.convert("RGB")
        if isinstance(frame, (str, PathLike)):
            with Image.open(Path(frame)) as image:
                return image.convert("RGB")
        try:
            import torch

            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
        except ImportError:
            pass
        array = np.asarray(frame)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.ndim != 3:
            raise ValueError(f"each frame must have 2 or 3 dimensions, got {array.shape}")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if np.issubdtype(array.dtype, np.floating):
            if array.size and float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if array.shape[-1] == 4:
            return Image.fromarray(array).convert("RGB")
        return Image.fromarray(array)

    def _prepare_frame(self, frame: Any) -> np.ndarray:
        image = self._as_pil(frame)
        if self.resize_shortest_edge is not None:
            shortest = min(image.size)
            scale = self.resize_shortest_edge / shortest
            resized = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                ),
                resample=self.resample,
            )
            crop_width = self.size["width"]
            crop_height = self.size["height"]
            left = (resized.width - crop_width) // 2
            top = (resized.height - crop_height) // 2
            fitted = resized.crop((left, top, left + crop_width, top + crop_height))
        else:
            resized = image
            fitted = ImageOps.fit(
                resized,
                (self.size["width"], self.size["height"]),
                method=self.resample,
                centering=(0.5, 0.5),
            )
        array = np.asarray(fitted, dtype=np.float32) / 255.0
        array = (array - np.asarray(self.image_mean, dtype=np.float32)) / np.asarray(
            self.image_std, dtype=np.float32
        )
        return np.moveaxis(array, -1, 0)

    @staticmethod
    def _uniform_indices(length: int, count: int) -> np.ndarray:
        if length <= count:
            return np.arange(length, dtype=np.int64)
        return np.rint(np.linspace(0, length - 1, count)).astype(np.int64)

    def preprocess(
        self,
        videos: Sequence[Any] | Any,
        return_tensors: str | None = None,
        **_: Any,
    ) -> BatchFeature:
        """Normalize decoded frames.

        `videos` may be a single sequence of frames or a batch of frame
        sequences. Long inputs are sampled uniformly. Short inputs are padded
        by repeating their final frame and marked false in `frame_mask`.
        """

        if videos is None:
            raise ValueError("videos is required")
        if _is_frame(videos):
            batch = [[videos]]
        else:
            values = list(videos)
            if not values:
                raise ValueError("videos cannot be empty")
            batch = [values] if _is_frame(values[0]) else [list(v) for v in values]

        pixel_batch: list[np.ndarray] = []
        mask_batch: list[np.ndarray] = []
        for frames in batch:
            if not frames:
                raise ValueError("a video cannot contain zero frames")
            indices = self._uniform_indices(len(frames), self.num_frames)
            selected = [frames[int(i)] for i in indices]
            valid = len(selected)
            if valid < self.num_frames:
                selected.extend([selected[-1]] * (self.num_frames - valid))
            pixel_batch.append(np.stack([self._prepare_frame(frame) for frame in selected]))
            mask = np.zeros(self.num_frames, dtype=np.int64)
            mask[:valid] = 1
            mask_batch.append(mask)

        return BatchFeature(
            {
                "pixel_values": np.stack(pixel_batch).astype(np.float32),
                "frame_mask": np.stack(mask_batch),
            },
            tensor_type=return_tensors,
        )

    __call__ = preprocess
