from pathlib import Path

import cv2

from config import (
    CONSECUTIVE_DANGER_THRESHOLD,
    CONSECUTIVE_WARNING_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEFAULT_VIDEO_CAMERA,
    VIDEO_PROCESS_EVERY_N_FRAMES,
)
from detector.danger_detector import DangerDetector
from detector.risk_state import RiskState


class VideoProcessor:
  """Processes one video frame by frame and displays detections as the video plays."""

  def __init__(
      self,
      video_path,
      camera=DEFAULT_VIDEO_CAMERA,
      model_path=DEFAULT_MODEL_PATH,
      danger_detector=None,
      warning_threshold=CONSECUTIVE_WARNING_THRESHOLD,
      danger_threshold=CONSECUTIVE_DANGER_THRESHOLD,
      process_every_n_frames=VIDEO_PROCESS_EVERY_N_FRAMES,
      window_name='Truck Blind Spot Assistant',
  ):
    if process_every_n_frames < 1:
      raise ValueError('process_every_n_frames must be 1 or greater')

    self.video_path = Path(video_path)
    self.camera = camera
    self.window_name = window_name
    self.process_every_n_frames = process_every_n_frames
    self.danger_detector = danger_detector or DangerDetector(model_path=model_path)
    self.risk_state = RiskState(
        camera_names=[camera],
        warning_threshold=warning_threshold,
        danger_threshold=danger_threshold,
    )

  def process(self):
    """Runs detection on the selected video until it ends or the user presses q."""
    if not self.video_path.exists():
      raise FileNotFoundError(f'Video does not exist: {self.video_path}')

    capture = cv2.VideoCapture(str(self.video_path))
    if not capture.isOpened():
      raise ValueError(f'Could not open video: {self.video_path}')

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_delay_ms = int(1000 / fps) if fps and fps > 0 else 1
    frame_index = 0

    try:
      while True:
        has_frame, frame = capture.read()
        if not has_frame:
          break

        if frame_index % self.process_every_n_frames == 0:
          result = self.danger_detector.detect_with_yolo(frame)
          danger = self.danger_detector.check_threshold(self.camera, result[0].boxes)
          annotated_frame = result[0].plot()
          self.risk_state.update(self.camera, danger)
        else:
          annotated_frame = frame.copy()

        self.danger_detector.draw_thresholds(annotated_frame, self.camera)
        risk_level = self.risk_state.get_level(self.camera)
        risk_count = self.risk_state.get_count(self.camera)
        self._draw_risk_label(annotated_frame, risk_level, risk_count)

        cv2.imshow(self.window_name, annotated_frame)
        if cv2.waitKey(frame_delay_ms) & 0xFF == ord('q'):
          break

        frame_index += 1
    finally:
      capture.release()
      cv2.destroyWindow(self.window_name)

  def _label_color(self, risk_level):
    if risk_level == 'danger':
      return (0, 0, 255)
    if risk_level == 'warning':
      return (0, 165, 255)
    if risk_level == 'possible':
      return (0, 255, 255)
    return (0, 255, 0)

  def _draw_risk_label(self, frame, risk_level, risk_count):
    label = f'{self.camera}: {risk_level} ({risk_count})'
    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        self._label_color(risk_level),
        2,
        cv2.LINE_AA,
    )
