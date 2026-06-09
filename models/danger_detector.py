from ultralytics import YOLO

class DangerDetector:
  def __init__(self):
    self.model = YOLO('yolov8n.pt')
    self.danger_classes = {
      'person',
      'bicycle',
      'motorcycle',
      'car',
      'bus',
      'truck'
    }

  def detect_with_yolo(self, image):
    results = self.model(image)
    return results

  def check_threshold(self, camera, boxes):
    danger = False
    orig_shape = boxes.orig_shape[0], boxes.orig_shape[1]
    orig_area = orig_shape[0]*orig_shape[1]
    base_threshold=0.05
    extreme_threshold=0.3
    for obj in boxes:
      class_name = self.model.names[int(obj.cls[0])]
      if obj.conf[0] > 0.5 and class_name in self.danger_classes:
        obj_cords = obj.xyxy[0].tolist()
        x1,y1,x2,y2 = obj_cords[0], obj_cords[1], obj_cords[2],obj_cords[3]
        object_position = (x1 + x2)/2
        object_position_norm = object_position / orig_shape[1]
        object_area_norm = (x2 - x1) * (y2 - y1) / orig_area

        obj_in_threshold = object_area_norm > base_threshold

        if camera == 'CAM_FRONT':
          obj_in_center = object_position_norm >= 0.4 and object_position_norm <= 0.6
          if (obj_in_center and obj_in_threshold) or object_area_norm > extreme_threshold:
            danger = True

        elif camera == 'CAM_BACK':
          obj_in_center = object_position_norm >= 0.4 and object_position_norm <= 0.6
          if (obj_in_center and obj_in_threshold) or object_area_norm > extreme_threshold:
            danger = True

        elif camera == 'CAM_FRONT_LEFT':
          obj_in_left = object_position_norm <= 0.4
          if (obj_in_left and obj_in_threshold) or object_area_norm > extreme_threshold:
            danger = True

        elif camera == 'CAM_FRONT_RIGHT':
          object_in_right = object_position_norm >=0.6
          if (object_in_right and obj_in_threshold) or object_area_norm > extreme_threshold:
            danger=True

    return danger