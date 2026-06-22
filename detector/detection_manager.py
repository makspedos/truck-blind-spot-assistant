import matplotlib.pyplot as plt
from detector.danger_detector import DangerDetector
from detector.risk_state import RiskState
import cv2

from config import (
    CAMERA_NAMES,
    CONSECUTIVE_DANGER_THRESHOLD,
    CONSECUTIVE_WARNING_THRESHOLD,
    DEFAULT_MODEL_PATH,
)


class DetectionManager:
  """Coordinates detection, danger counting, and camera-frame visualization."""

  def __init__(
      self,
      data,
      model_path=DEFAULT_MODEL_PATH,
      camera_names=None,
      warning_threshold=CONSECUTIVE_WARNING_THRESHOLD,
      danger_threshold=CONSECUTIVE_DANGER_THRESHOLD,
  ):
    self.data = data
    self.camera_names = camera_names or CAMERA_NAMES
    self.risk_state = RiskState(
        camera_names=self.camera_names,
        warning_threshold=warning_threshold,
        danger_threshold=danger_threshold,
    )
    self.danger_detector = DangerDetector(model_path=model_path)


  def display(self):
    if not self.data:
      raise ValueError('No frames were provided for detection.')

    fig, axes = plt.subplots(len(self.data), len(self.camera_names), figsize=(15,10))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for frame in range(len(self.data)):
      for i, snapshot in enumerate(self.data[frame]):
        current_camera = snapshot['camera']
        image_input = snapshot['image'] if 'image' in snapshot else snapshot['path']
        result = self.danger_detector.detect_with_yolo(image_input)
        danger = self.danger_detector.check_threshold(current_camera,result[0].boxes)
        risk_level = self.risk_state.update(current_camera, danger)
        risk_count = self.risk_state.get_count(current_camera)

        if risk_level != 'clear':
            print(f'{current_camera}: {risk_level} ({risk_count})')

        idx = frame * len(self.camera_names) + i
        image_bgr = result[0].plot()
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(image)
        if risk_level == 'danger':
          axes[idx].set_title(f'Image {idx+1} - danger: {current_camera}')
        elif risk_level == 'warning':
          axes[idx].set_title(f'Image {idx+1} - warning: {current_camera}')
        else:
          axes[idx].set_title(f'Image {idx + 1}: {current_camera}')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()
