from ultralytics import YOLO

from config import (
    BASE_THRESHOLD,
    BOTTOM_POSITION_THRESHOLD,
    CAMERA_HORIZONTAL_ZONES,
    DANGER_CLASSES,
    DEFAULT_MODEL_PATH,
    EXTREME_THRESHOLD,
    MIN_CONFIDENCE,
)


class DangerDetector:
  """Runs YOLO detections and checks if objects enter camera-specific danger zones."""

  def __init__(
      self,
      model_path=DEFAULT_MODEL_PATH,
      model=None,
      danger_classes=None,
      min_confidence=MIN_CONFIDENCE,
      base_threshold=BASE_THRESHOLD,
      extreme_threshold=EXTREME_THRESHOLD,
      bottom_position_threshold=BOTTOM_POSITION_THRESHOLD,
      camera_horizontal_zones=None,
  ):
    self.model = model or YOLO(model_path)
    self.danger_classes = danger_classes if danger_classes is not None else DANGER_CLASSES
    self.min_confidence = min_confidence
    self.base_threshold = base_threshold
    self.extreme_threshold = extreme_threshold
    self.bottom_position_threshold = bottom_position_threshold
    self.camera_horizontal_zones = camera_horizontal_zones or CAMERA_HORIZONTAL_ZONES

  def detect_with_yolo(self, image):
    results = self.model(image)
    return results

  def check_threshold(self, camera, boxes):
    """Returns True when a detected object is dangerous for the given camera."""
    camera_zone = self.camera_horizontal_zones.get(camera)
    if camera_zone is None:
      return False

    zone_min_x, zone_max_x = camera_zone
    orig_shape = boxes.orig_shape[0], boxes.orig_shape[1]
    orig_area = orig_shape[0]*orig_shape[1]
    for obj in boxes:
      class_name = self.model.names[int(obj.cls[0])]
      if obj.conf[0] > self.min_confidence and class_name in self.danger_classes:
        obj_cords = obj.xyxy[0].tolist()
        x1,y1,x2,y2 = obj_cords[0], obj_cords[1], obj_cords[2],obj_cords[3]
        object_position = (x1 + x2)/2
        object_position_norm = object_position / orig_shape[1]
        object_bottom_position_norm = y2 / orig_shape[0]
        object_area_norm = (x2 - x1) * (y2 - y1) / orig_area

        object_is_large = object_area_norm > self.base_threshold
        object_is_extremely_large = object_area_norm > self.extreme_threshold
        object_is_near_bottom = object_bottom_position_norm >= self.bottom_position_threshold
        object_in_camera_zone = zone_min_x <= object_position_norm <= zone_max_x

        if self._is_dangerous_object(
            object_in_camera_zone,
            object_is_near_bottom,
            object_is_large,
            object_is_extremely_large,
        ):
          return True

    return False

  def _is_dangerous_object(
      self,
      object_in_camera_zone,
      object_is_near_bottom,
      object_is_large,
      object_is_extremely_large,
  ):
    """Applies the standard and extreme danger rules to one detected object."""
    is_standard_danger = all([
        object_is_near_bottom,
        object_in_camera_zone,
        object_is_large,
    ])

    is_extreme_danger = all([
        object_is_near_bottom,
        object_is_extremely_large,
    ])

    return is_standard_danger or is_extreme_danger
