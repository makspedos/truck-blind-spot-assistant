# Truck Blind Spot Assistant

Prototype for detecting potential blind-spot danger around a truck using YOLO object detection and camera-specific risk zones.

The project currently supports two input modes:

- nuScenes camera frame groups.
- One selected video file processed frame by frame.

This is not a production-ready driver assistance system. The current risk logic is heuristic and is meant for experimentation with object detection, camera zones, and simple consecutive-frame danger counting.

## Current Features

- YOLO-based object detection with Ultralytics.
- Camera-specific horizontal danger zones.
- Bottom-zone filtering so objects must appear close to the lower part of the frame before they count as dangerous.
- Consecutive-frame risk state:
  - `clear`
  - `possible`
  - `warning`
  - `danger`
- Risk count is capped at the configured danger threshold.
- Video mode keeps showing the latest risk state on frames where YOLO detection is skipped.
- Unit tests for configuration, danger threshold logic, risk state, and basic video processor behavior.

## Project Structure

```text
truck-blind-spot-assistant/
  app.py
  config.py
  detector/
    danger_detector.py
    detection_manager.py
    nuscenes_converter.py
    risk_state.py
    video_processor.py
  scripts/
    preview_images.py
  tests/
    test_config.py
    test_danger_detector.py
    test_risk_state.py
    test_video_processor.py
  requirements.txt
  pyproject.toml
  uv.lock
```

Local datasets, videos, model weights, generated outputs, and notebooks are ignored by Git.

## Requirements

Python 3.12 or newer is expected.

Install dependencies with `uv`:

```bash
uv sync
```

The main libraries are:

- `ultralytics`
- `opencv-python`
- `nuscenes-devkit`
- `matplotlib`
- `numpy`

## Expected Local Files

The project expects local data and model files when running the full pipeline:

```text
models/yolov8n.pt
dataset/
videos/
```

These files are not committed to the repository.

## Running With nuScenes Frames

Default run:

```bash
uv run python app.py
```

Custom nuScenes paths and sample range:

```bash
uv run python app.py --dataroot dataset --version v1.0-mini --start 90 --end 95
```

The nuScenes mode loads selected camera frames and displays YOLO detections with camera-specific danger state.

## Running With One Video

```bash
uv run python app.py --video videos/your_video.mp4
```

By default, the video is treated as `CAM_FRONT`. You can assign another camera name:

```bash
uv run python app.py --video videos/your_video.mp4 --video-camera CAM_FRONT_LEFT
```

Video mode processes the selected file frame by frame. YOLO detection runs according to `VIDEO_PROCESS_EVERY_N_FRAMES` in `config.py`, while the last known risk state remains visible on skipped frames.

## Running Tests

```bash
uv run python -m unittest discover -s tests -v
```

Current test coverage is focused on the core threshold and risk-state behavior.

## Current Limitations

- Video mode processes one selected video file, not synchronized multi-camera video.
- nuScenes data is used as synchronized camera frame groups, not as regular MP4 video files.
- Danger detection is based on camera zones, object size, confidence, and bottom-frame position. These thresholds are heuristic.
- The project does not yet include a production-grade calibration system, tracking system, or real-time deployment pipeline.
- The current logic is intended for prototyping and visual testing, not for safety-critical decisions.
