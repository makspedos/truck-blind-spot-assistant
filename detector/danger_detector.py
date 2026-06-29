import cv2
from ultralytics import YOLO

from config import (
    BOTTOM_POSITION_THRESHOLD,
    CAMERA_HORIZONTAL_ZONES,
    DANGER_CLASSES,
    DEFAULT_MODEL_PATH,
    MIN_CONFIDENCE,
    MIN_OBJECT_AREA,
    MIN_ZONE_OVERLAP_RATIO,
)


class DangerDetector:
  """Runs YOLO detections and checks if objects enter camera-specific danger zones."""

  def __init__(
      self,
      model_path=DEFAULT_MODEL_PATH,
      model=None,
      danger_classes=None,
      min_confidence=MIN_CONFIDENCE,
      min_object_area=MIN_OBJECT_AREA,
      min_zone_overlap_ratio=MIN_ZONE_OVERLAP_RATIO,
      bottom_position_threshold=BOTTOM_POSITION_THRESHOLD,
      camera_horizontal_zones=None,
  ):
    self.model = model or YOLO(model_path)
    self.danger_classes = danger_classes if danger_classes is not None else DANGER_CLASSES
    self.min_confidence = min_confidence
    self.min_object_area = min_object_area
    self.min_zone_overlap_ratio = min_zone_overlap_ratio
    self.bottom_position_threshold = bottom_position_threshold
    self.camera_horizontal_zones = camera_horizontal_zones or CAMERA_HORIZONTAL_ZONES

  def detect_with_yolo(self, image):
    results = self.model(image)
    return results

  def draw_thresholds(self, frame, camera):
    """Draws the active danger zone used by the threshold check."""
    camera_zone = self.camera_horizontal_zones.get(camera)
    if camera_zone is None:
      return frame

    height, width = frame.shape[:2]
    zone_min_x, zone_max_x = camera_zone
    left_x = int(zone_min_x * width)
    right_x = int(zone_max_x * width)
    bottom_y = int(self.bottom_position_threshold * height)

    overlay = frame.copy()
    cv2.rectangle(overlay, (left_x, bottom_y), (right_x, height - 1), (0, 255, 255), -1)
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
    cv2.rectangle(frame, (left_x, bottom_y), (right_x, height - 1), (0, 255, 255), 2)
    cv2.putText(
        frame,
        'threshold zone',
        (left_x + 8, max(bottom_y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame

  def check_threshold(self, camera, boxes):
    """Returns True when a detected object is dangerous for the given camera."""
    camera_zone = self.camera_horizontal_zones.get(camera)
    if camera_zone is None:
      return False

    height, width = boxes.orig_shape
    orig_area = height * width
    zone_box = self._threshold_zone_box(camera, width, height)
    for obj in boxes:
      class_name = self.model.names[int(obj.cls[0])]
      if obj.conf[0] > self.min_confidence and class_name in self.danger_classes:
        obj_cords = obj.xyxy[0].tolist()
        x1, y1, x2, y2 = obj_cords[0], obj_cords[1], obj_cords[2], obj_cords[3]
        object_area_norm = (x2 - x1) * (y2 - y1) / orig_area
        overlap_ratio = self._box_overlap_ratio((x1, y1, x2, y2), zone_box)

        if object_area_norm >= self.min_object_area and overlap_ratio >= self.min_zone_overlap_ratio:
          return True

    return False

  def _threshold_zone_box(self, camera, width, height):
    zone_min_x, zone_max_x = self.camera_horizontal_zones[camera]
    return (
        zone_min_x * width,
        self.bottom_position_threshold * height,
        zone_max_x * width,
        height,
    )

  def _box_overlap_ratio(self, object_box, zone_box):
    x1, y1, x2, y2 = object_box
    zone_x1, zone_y1, zone_x2, zone_y2 = zone_box

    overlap_width = max(0, min(x2, zone_x2) - max(x1, zone_x1))
    overlap_height = max(0, min(y2, zone_y2) - max(y1, zone_y1))
    overlap_area = overlap_width * overlap_height
    object_area = (x2 - x1) * (y2 - y1)

    if object_area <= 0:
      return 0

    return overlap_area / object_area
