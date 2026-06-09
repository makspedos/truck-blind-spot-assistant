from nuscenes import NuScenes

from config import CAMERA_NAMES, DEFAULT_DATAROOT, DEFAULT_NUSCENES_VERSION, DEFAULT_SAMPLE_END, DEFAULT_SAMPLE_START


class NuScenesConverter:
  """Loads nuScenes samples and converts selected cameras into frame dictionaries."""

  def __init__(self,
      dataroot=DEFAULT_DATAROOT,
      version=DEFAULT_NUSCENES_VERSION,
      camera_names=None,
      sample_start=DEFAULT_SAMPLE_START,
      sample_end=DEFAULT_SAMPLE_END,
  ):
    self.nuscenes = NuScenes(version, dataroot=dataroot)
    self.camera_names = camera_names or CAMERA_NAMES
    self.data = self.nuscenes.sample[sample_start:sample_end]

  def fetch_data(self):
    """Returns camera image paths grouped by nuScenes sample."""
    frame_list = []
    for sample in self.data:
      frames = []
      for camera in self.camera_names:
        token = sample['data'][camera]
        frame = {'path':self.nuscenes.get_sample_data_path(token), 'camera':camera}
        frames.append(frame)
      frame_list.append(frames)
    return frame_list
