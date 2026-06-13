from config import CONSECUTIVE_DANGER_THRESHOLD, CONSECUTIVE_WARNING_THRESHOLD


class RiskState:
  """Tracks consecutive danger frames and converts them into risk levels."""

  def __init__(
      self,
      camera_names,
      warning_threshold=CONSECUTIVE_WARNING_THRESHOLD,
      danger_threshold=CONSECUTIVE_DANGER_THRESHOLD,
  ):
    self.warning_threshold = warning_threshold
    self.danger_threshold = danger_threshold
    self.danger_counts = {camera: 0 for camera in camera_names}

  def update(self, camera, is_danger):
    """Updates one camera state and returns the current risk level."""
    if is_danger:
      self.danger_counts[camera] += 1
    else:
      self.danger_counts[camera] = 0

    return self.get_level(camera)

  def get_level(self, camera):
    count = self.danger_counts[camera]

    if count >= self.danger_threshold:
      return 'danger'

    if count >= self.warning_threshold:
      return 'warning'

    return 'clear'

  def get_count(self, camera):
    return self.danger_counts[camera]
