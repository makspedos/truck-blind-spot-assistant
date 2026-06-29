import unittest

from detector.risk_state import RiskState


class RiskStateTest(unittest.TestCase):
  def test_initial_level_is_clear(self):
    risk_state = RiskState(['CAM_FRONT'])

    self.assertEqual(risk_state.get_level('CAM_FRONT'), 'clear')

  def test_first_danger_frame_is_possible(self):
    risk_state = RiskState(['CAM_FRONT'], warning_threshold=3, danger_threshold=5)

    level = risk_state.update('CAM_FRONT', True)

    self.assertEqual(level, 'possible')
    self.assertEqual(risk_state.get_count('CAM_FRONT'), 1)

  def test_warning_level_after_warning_threshold(self):
    risk_state = RiskState(['CAM_FRONT'], warning_threshold=3, danger_threshold=5)

    risk_state.update('CAM_FRONT', True)
    risk_state.update('CAM_FRONT', True)
    level = risk_state.update('CAM_FRONT', True)

    self.assertEqual(level, 'warning')
    self.assertEqual(risk_state.get_count('CAM_FRONT'), 3)

  def test_danger_level_after_danger_threshold(self):
    risk_state = RiskState(['CAM_FRONT'], warning_threshold=2, danger_threshold=3)

    risk_state.update('CAM_FRONT', True)
    risk_state.update('CAM_FRONT', True)
    level = risk_state.update('CAM_FRONT', True)

    self.assertEqual(level, 'danger')
    self.assertEqual(risk_state.get_count('CAM_FRONT'), 3)

  def test_danger_count_does_not_exceed_danger_threshold(self):
    risk_state = RiskState(['CAM_FRONT'], warning_threshold=2, danger_threshold=3)

    for _ in range(5):
      level = risk_state.update('CAM_FRONT', True)

    self.assertEqual(level, 'danger')
    self.assertEqual(risk_state.get_count('CAM_FRONT'), 3)

  def test_clear_frame_decreases_count(self):
    risk_state = RiskState(['CAM_FRONT'], warning_threshold=2, danger_threshold=3)

    risk_state.update('CAM_FRONT', True)
    risk_state.update('CAM_FRONT', True)
    level = risk_state.update('CAM_FRONT', False)

    self.assertEqual(level, 'possible')
    self.assertEqual(risk_state.get_count('CAM_FRONT'), 1)

  def test_clear_frame_does_not_decrease_count_below_zero(self):
    risk_state = RiskState(['CAM_FRONT'], warning_threshold=2, danger_threshold=3)

    level = risk_state.update('CAM_FRONT', False)

    self.assertEqual(level, 'clear')
    self.assertEqual(risk_state.get_count('CAM_FRONT'), 0)

  def test_cameras_are_tracked_independently(self):
    risk_state = RiskState(['CAM_FRONT', 'CAM_BACK'], warning_threshold=2, danger_threshold=3)

    front_level = risk_state.update('CAM_FRONT', True)
    back_level = risk_state.update('CAM_BACK', False)

    self.assertEqual(front_level, 'possible')
    self.assertEqual(back_level, 'clear')


if __name__ == '__main__':
  unittest.main()
