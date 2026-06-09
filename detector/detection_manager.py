import matplotlib.pyplot as plt
from detector.danger_detector import DangerDetector
import cv2

from config import CAMERA_NAMES, CONSECUTIVE_DANGER_THRESHOLD, DEFAULT_MODEL_PATH


class DetectionManager:
  """Coordinates detection, danger counting, and camera-frame visualization."""

  def __init__(
      self,
      data,
      model_path=DEFAULT_MODEL_PATH,
      camera_names=None,
      danger_threshold=CONSECUTIVE_DANGER_THRESHOLD,
  ):
    self.data = data
    self.camera_names = camera_names or CAMERA_NAMES
    self.danger_threshold = danger_threshold
    self.danger_detector = DangerDetector(model_path=model_path)


  def display(self):
    fig, axes = plt.subplots(len(self.data), len(self.camera_names), figsize=(15,10))
    axes = axes.flatten()
    total_danger = {camera: 0 for camera in self.camera_names}
    for frame in range(len(self.data)):
      for i, snapshot in enumerate(self.data[frame]):
        current_camera = snapshot['camera']
        result = self.danger_detector.detect_with_yolo(snapshot['path'])
        danger = self.danger_detector.check_threshold(current_camera,result[0].boxes)

        if danger:
            total_danger[current_camera]+=1
            print(total_danger[current_camera])
        else:
            total_danger[current_camera]=0

        idx = frame * len(self.camera_names) + i
        image_bgr = result[0].plot()
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(image)
        if total_danger[current_camera] >= self.danger_threshold:
            # print(f'Danger in {current_camera}')
          axes[idx].set_title(f'Image {idx+1} - danger: {current_camera}')
        else:
          axes[idx].set_title(f'Image {idx + 1}')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()
