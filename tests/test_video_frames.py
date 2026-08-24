from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from egosieve.video.frames import (
    build_frame_batch_extract_command,
    build_frame_extract_command,
    extract_plan_frames,
    frame_to_tensor,
    frames_to_tensor,
)
from egosieve.video.sampling import plan_frame_samples


def test_frame_command_is_shell_free_data_and_preserves_spaces(tmp_path: Path) -> None:
    source = tmp_path / "source $(not-executed).mp4"
    output = tmp_path / "frame with spaces.jpg"
    command = build_frame_extract_command(source, 1.25, output, output_size=(320, 180))

    assert isinstance(command, list)
    assert str(source.resolve()) in command
    assert str(output.resolve()) == command[-1]
    assert "1.25" in command
    assert "scale=320:180" in command[command.index("-vf") + 1]


def test_extract_batches_unique_samples_without_duplicate_work(tmp_path: Path) -> None:
    plan = plan_frame_samples(8, window_duration_s=4, stride_s=1, frames_per_window=4)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def runner(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        calls.append((command, kwargs))
        return CompletedProcess(command, 0, stdout="", stderr="")

    frames = extract_plan_frames(
        tmp_path / "source.mp4",
        plan,
        tmp_path / "frames",
        runner=runner,
        verify_outputs=False,
        batch_size=5,
    )

    assert len(calls) == (len(plan.samples) + 4) // 5
    assert (
        len(calls) < len(plan.samples) < sum(len(window.sample_indices) for window in plan.windows)
    )
    assert all(kwargs["shell"] is False for _, kwargs in calls)
    assert all(kwargs["timeout"] == 120.0 for _, kwargs in calls)
    assert tuple(frame.sample for frame in frames) == plan.samples


def test_batch_command_seeks_once_and_has_one_output_per_request(tmp_path: Path) -> None:
    outputs = [tmp_path / f"frame {index}.jpg" for index in range(3)]
    command = build_frame_batch_extract_command(
        tmp_path / "source.mp4",
        list(zip((10.0, 10.5, 11.0), outputs, strict=True)),
        output_size=(80, 60),
    )

    assert command.count("-i") == 1
    assert command[command.index("-i") - 2 : command.index("-i")] == ["-ss", "10"]
    assert all(str(path.resolve()) in command for path in outputs)
    assert command.count("-frames:v") == 3


def test_numpy_tensor_transform_is_model_agnostic() -> None:
    np = pytest.importorskip("numpy")
    frame = np.array(
        [
            [[0, 127, 255], [255, 127, 0]],
            [[255, 255, 255], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )

    tensor = frame_to_tensor(frame, output_layout="CHW")
    batch = frames_to_tensor([frame, frame], size=(4, 4), output_layout="CHW")

    assert tensor.shape == (3, 2, 2)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 0] == 0
    assert tensor[2, 0, 0] == 1
    assert batch.shape == (2, 3, 4, 4)
