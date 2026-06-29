import unittest

from detector.danger_detector import DangerDetector


class FakeValue:
  def __init__(self, value):
    self.value = value

  def __getitem__(self, index):
    return self.value


class FakeCoordinates:
  def __init__(self, coordinates):
    self.coordinates = coordinates

  def __getitem__(self, index):
    return self

  def tolist(self):
    return self.coordinates


class FakeBox:
  def __init__(self, class_id, confidence, coordinates):
    self.cls = FakeValue(class_id)
    self.conf = FakeValue(confidence)
    self.xyxy = FakeCoordinates(coordinates)


class FakeBoxes:
  def __init__(self, boxes, orig_shape=(100, 100)):
    self.boxes = boxes
    self.orig_shape = orig_shape

  def __iter__(self):
    return iter(self.boxes)


class FakeModel:
  names = {
      0: 'person',
      1: 'bicycle',
      2: 'car',
      3: 'motorcycle',
      5: 'bus',
      7: 'truck',
      9: 'traffic light',
  }


def make_detector():
  return DangerDetector(model=FakeModel())


def make_boxes(class_id=2, confidence=0.9, coordinates=None):
  return FakeBoxes([FakeBox(class_id, confidence, coordinates or [40, 20, 60, 50])])


class DangerDetectorThresholdTest(unittest.TestCase):
  def test_front_center_object_above_threshold_is_danger(self):
    detector = make_detector()

    danger = detector.check_threshold('CAM_FRONT', make_boxes(coordinates=[35, 65, 65, 95]))

    self.assertTrue(danger)

  def test_front_side_object_outside_zone_is_not_danger(self):
    detector = make_detector()

    danger = detector.check_threshold('CAM_FRONT', make_boxes(coordinates=[5, 65, 35, 95]))

    self.assertFalse(danger)

  def test_left_camera_object_on_left_is_danger(self):
    detector = make_detector()

    danger = detector.check_threshold('CAM_FRONT_LEFT', make_boxes(coordinates=[5, 65, 35, 95]))

    self.assertTrue(danger)

  def test_person_overlapping_zone_is_danger(self):
    detector = make_detector()

    danger = detector.check_threshold('CAM_FRONT_LEFT', make_boxes(class_id=0, coordinates=[10, 65, 20, 95]))

    self.assertTrue(danger)

  def test_right_camera_object_on_right_is_danger(self):
    detector = make_detector()

    danger = detector.check_threshold('CAM_FRONT_RIGHT', make_boxes(coordinates=[65, 65, 95, 95]))

    self.assertTrue(danger)

  def test_front_center_object_in_middle_is_not_danger(self):
    detector = make_detector()

    danger = detector.check_threshold('CAM_FRONT', make_boxes(coordinates=[35, 10, 65, 50]))

    self.assertFalse(danger)

  def test_large_object_without_overlap_is_not_danger(self):
    detector = make_detector()

    danger = detector.check_threshold('CAM_FRONT', make_boxes(coordinates=[10, 0, 90, 50]))

    self.assertFalse(danger)

  def test_tiny_object_with_overlap_is_not_danger(self):
    detector = make_detector()

    danger = detector.check_threshold('CAM_FRONT', make_boxes(coordinates=[45, 70, 47, 72]))

    self.assertFalse(danger)

  def test_low_confidence_object_is_not_danger(self):
    detector = make_detector()

    danger = detector.check_threshold('CAM_FRONT', make_boxes(confidence=0.4, coordinates=[35, 10, 65, 40]))

    self.assertFalse(danger)

  def test_ignored_class_is_not_danger(self):
    detector = make_detector()

    danger = detector.check_threshold('CAM_FRONT', make_boxes(class_id=9, coordinates=[35, 10, 65, 40]))

    self.assertFalse(danger)

  def test_unknown_camera_is_not_danger(self):
    detector = make_detector()

    danger = detector.check_threshold('UNKNOWN_CAMERA', make_boxes(coordinates=[35, 65, 65, 95]))

    self.assertFalse(danger)


if __name__ == '__main__':
  unittest.main()
