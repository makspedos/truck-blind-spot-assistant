import unittest

from config import BASE_THRESHOLD, CAMERA_NAMES, CONSECUTIVE_DANGER_THRESHOLD, DANGER_CLASSES, EXTREME_THRESHOLD


class ConfigTest(unittest.TestCase):
  def test_camera_names_have_expected_detection_cameras(self):
    self.assertEqual(
        CAMERA_NAMES,
        [
            'CAM_FRONT',
            'CAM_BACK',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
        ],
    )

  def test_thresholds_are_valid(self):
    self.assertGreater(BASE_THRESHOLD, 0)
    self.assertGreater(EXTREME_THRESHOLD, BASE_THRESHOLD)
    self.assertGreater(CONSECUTIVE_DANGER_THRESHOLD, 0)

  def test_danger_classes_include_vehicle_classes(self):
    self.assertTrue({'car', 'bus', 'truck'}.issubset(DANGER_CLASSES))


if __name__ == '__main__':
  unittest.main()
