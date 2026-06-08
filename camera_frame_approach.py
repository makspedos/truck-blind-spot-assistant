from nuscenes import NuScenes
from variables import dataroot
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


nuscenes = NuScenes('v1.0-mini', dataroot=dataroot)
data = nuscenes.sample[90:95]

class CameraFrame:
  def __init__(self, path, position):
    self.path = path
    self.position = position
    self.danger_classes = {
      'person',
      'bicycle',
      'motorcycle',
      'car',
      'bus',
      'truck'
    }



  def check_threshold(self, boxes):

    orig_height = boxes.orig_shape[0]
    orig_width = boxes.orig_shape[1]
    orig_area = orig_height*orig_width
    threshold=0.05
    extreme_threshold=0.5
    for obj in boxes:
      class_name = self.map_cls(int(obj.cls[0]))
      if obj.conf[0] > 0.5 and class_name in self.danger_classes:
        obj_cord = obj.xyxy[0].tolist()
        x1,y1,x2,y2 = obj_cord[0], obj_cord[1], obj_cord[2],obj_cord[3]
        object_position = (x1 + x2)/2
        object_position_norm = object_position / orig_width
        object_area_norm = (x2 - x1) * (y2 - y1) / orig_area

        obj_in_threshold = object_area_norm > threshold

        if self.position == 'CAM_FRONT':
          obj_in_center = object_position_norm >= 0.4 and object_position_norm <= 0.6
          if (obj_in_center and obj_in_threshold) or object_area_norm > extreme_threshold:
            print(f'Danger ahead\t Object: {class_name}\t {object_area_norm}')

        elif self.position == 'CAM_BACK':
          obj_in_center = object_position_norm >= 0.4 and object_position_norm <= 0.6
          if (obj_in_center and obj_in_threshold) or object_area_norm > extreme_threshold:
            print(f'Danger behind\t Object: {class_name}\t {object_area_norm}')

        elif self.position == 'CAM_FRONT_LEFT':
          obj_in_left = object_position_norm <= 0.4
          if (obj_in_left and obj_in_threshold) or object_area_norm > extreme_threshold:
            print(f'Danger on the left\t Object: {class_name}\t {object_area_norm}')

        elif self.position == 'CAM_FRONT_RIGHT':
          object_in_right = object_position_norm >=0.6
          if (object_in_right and obj_in_threshold) or object_area_norm > extreme_threshold:
            print(f'Danger on the right\t Object: {class_name}\t {object_area_norm}')


  def map_cls(self, class_num):
    map_list = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
    return map_list[class_num]


samples = nuscenes.sample[150:151] #test150:151
camera_names = [
    'CAM_FRONT',
    'CAM_BACK',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
]
frame_list = []
for sample in samples:
  frames = []
  for camera in camera_names:
    token = sample['data'][camera]
    frame = CameraFrame(path=nuscenes.get_sample_data_path(token), position=camera)
    frames.append(frame)
  frame_list.append(frames)

model = YOLO('yolov8n.pt')
fig, axes = plt.subplots(len(frame_list), 4, figsize=(15, 10))
axes = axes.flatten() # Flatten the 2x3 array of axes to easily iterate
boxes = []
for frame in range(len(frame_list)):
  for i, camera in enumerate(frame_list[frame]):
      results = model(camera.path)
      camera.check_threshold(results[0].boxes)
      image_bgr = results[0].plot()
      image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
      idx = frame * 4 + i
      axes[idx].imshow(image)
      axes[idx].set_title(f'Image {idx+1}')
      axes[idx].axis('off')
plt.tight_layout()
plt.show()

