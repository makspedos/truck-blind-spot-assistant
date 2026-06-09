import matplotlib.pyplot as plt
from models.danger_detector import DangerDetector
import cv2

class DetectionManager:
  def __init__(self, data):
    self.data = data
    self.danger_detector = DangerDetector()


  def display(self):
    fig, axes = plt.subplots(len(self.data), 4, figsize=(15,10))
    axes = axes.flatten()
    total_danger = {
      'CAM_FRONT':0,
      'CAM_BACK':0,
      'CAM_FRONT_LEFT':0,
      'CAM_FRONT_RIGHT':0,
    }
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
        if total_danger[current_camera] >=5:
            print(f'Danger in {current_camera}')

        idx = frame * 4 + i
        image_bgr = result[0].plot()
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(image)
        axes[idx].set_title(f'Image {idx+1}')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()