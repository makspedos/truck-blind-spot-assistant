import unittest

from config import DEFAULT_VIDEO_CAMERA
from detector.video_processor import VideoProcessor


class FakeDangerDetector:
  pass


class VideoProcessorTest(unittest.TestCase):
  def test_missing_video_file_raises_error(self):
    processor = VideoProcessor(video_path='missing-video.mp4', danger_detector=FakeDangerDetector())

    with self.assertRaises(FileNotFoundError):
      processor.process()

  def test_label_color_depends_on_risk_level(self):
    processor = VideoProcessor(video_path='missing-video.mp4', danger_detector=FakeDangerDetector())

    self.assertEqual(processor._label_color('danger'), (0, 0, 255))
    self.assertEqual(processor._label_color('warning'), (0, 165, 255))
    self.assertEqual(processor._label_color('possible'), (0, 255, 255))
    self.assertEqual(processor._label_color('clear'), (0, 255, 0))

  def test_default_camera_is_front_camera(self):
    processor = VideoProcessor(video_path='missing-video.mp4', danger_detector=FakeDangerDetector())

    self.assertEqual(DEFAULT_VIDEO_CAMERA, 'CAM_FRONT')
    self.assertEqual(processor.camera, 'CAM_FRONT')

  def test_process_every_n_frames_must_be_positive(self):
    with self.assertRaises(ValueError):
      VideoProcessor(
          video_path='missing-video.mp4',
          danger_detector=FakeDangerDetector(),
          process_every_n_frames=0,
      )


if __name__ == '__main__':
  unittest.main()
