from __future__ import annotations

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")

from egosieve.processing_egosieve import EgoSieveProcessor


def _frame(value: int, height: int = 20, width: int = 30) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_processor_batches_and_pads() -> None:
    processor = EgoSieveProcessor(size=16, num_frames=4)
    result = processor([[_frame(0), _frame(255)], [_frame(64)]], return_tensors="np")
    assert result["pixel_values"].shape == (2, 4, 3, 16, 16)
    assert result["frame_mask"].tolist() == [[1, 1, 0, 0], [1, 0, 0, 0]]


def test_processor_treats_list_of_frames_as_one_video() -> None:
    processor = EgoSieveProcessor(size={"height": 12, "width": 10}, num_frames=2)
    result = processor([_frame(10), _frame(20), _frame(30)], return_tensors="np")
    assert result["pixel_values"].shape == (1, 2, 3, 12, 10)
    assert result["frame_mask"].tolist() == [[1, 1]]


def test_processor_round_trip(tmp_path) -> None:
    processor = EgoSieveProcessor(size=18, num_frames=5)
    processor.save_pretrained(tmp_path)
    loaded = EgoSieveProcessor.from_pretrained(tmp_path)
    assert loaded.size == {"height": 18, "width": 18}
    assert loaded.num_frames == 5


def test_processor_rejects_empty_video() -> None:
    processor = EgoSieveProcessor(size=16, num_frames=4)
    with pytest.raises(ValueError, match="empty"):
        processor([])


@pytest.mark.parametrize("num_frames", [True, 0, 2.5])
def test_processor_rejects_invalid_num_frames(num_frames) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        EgoSieveProcessor(size=16, num_frames=num_frames)


def test_processor_accepts_frame_paths(tmp_path) -> None:
    from PIL import Image

    path = tmp_path / "frame.png"
    Image.fromarray(_frame(128)).save(path)
    processor = EgoSieveProcessor(size=16, num_frames=2)
    result = processor([path], return_tensors="np")
    assert result["pixel_values"].shape == (1, 2, 3, 16, 16)
    assert result["frame_mask"].tolist() == [[1, 0]]


def test_default_spatial_transform_matches_dinov2_processor() -> None:
    from transformers import BitImageProcessorPil

    from egosieve.processing_egosieve import DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD

    frame = np.random.default_rng(5).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    ours = EgoSieveProcessor(num_frames=1)([frame], return_tensors="np")["pixel_values"][0, 0]
    upstream = BitImageProcessorPil(
        size={"shortest_edge": 256},
        crop_size={"height": 224, "width": 224},
        resample=3,
        image_mean=DEFAULT_IMAGE_MEAN,
        image_std=DEFAULT_IMAGE_STD,
    )(images=frame, return_tensors="np")["pixel_values"][0]
    np.testing.assert_array_equal(ours, upstream)
