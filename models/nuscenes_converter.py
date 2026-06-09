from nuscenes import NuScenes
from variables import dataroot, camera_names

class NuscenesConverter:
  def __init__(self):
    self.nuscenes = NuScenes('v1.0-mini', dataroot=dataroot)
    self.data = self.nuscenes.sample[90:95]

  def fetch_data(self):
    frame_list = []
    for sample in self.data:
      frames = []
      for camera in camera_names:
        token = sample['data'][camera]
        frame = {'path':self.nuscenes.get_sample_data_path(token), 'camera':camera}
        frames.append(frame)
      frame_list.append(frames)
    return frame_list