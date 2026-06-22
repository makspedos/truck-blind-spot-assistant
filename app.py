import argparse

from config import (
    DEFAULT_DATAROOT,
    DEFAULT_MODEL_PATH,
    DEFAULT_NUSCENES_VERSION,
    DEFAULT_SAMPLE_END,
    DEFAULT_SAMPLE_START,
    DEFAULT_VIDEO_CAMERA,
)
from detector.detection_manager import DetectionManager
from detector.nuscenes_converter import NuScenesConverter
from detector.video_processor import VideoProcessor


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run truck blind spot detection on nuScenes camera frames.')
    parser.add_argument('--dataroot', default=DEFAULT_DATAROOT, help='Path to the nuScenes dataset folder.')
    parser.add_argument('--version', default=DEFAULT_NUSCENES_VERSION, help='nuScenes dataset version.')
    parser.add_argument('--video', metavar='PATH', help='Front-camera video file to process instead of nuScenes.')
    parser.add_argument('--video-camera', default=DEFAULT_VIDEO_CAMERA, help='Camera name assigned to video frames.')
    parser.add_argument(
        '--start',
        type=int,
        default=DEFAULT_SAMPLE_START,
        metavar='INDEX',
        help='First nuScenes sample index to process.',
    )
    parser.add_argument(
        '--end',
        type=int,
        default=DEFAULT_SAMPLE_END,
        metavar='INDEX',
        help='Sample index to stop before. Must be greater than --start.',
    )
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help='Path to YOLO model weights.')
    args = parser.parse_args(argv)

    if args.start < 0:
        parser.error('--start must be 0 or greater')
    if args.end <= args.start:
        parser.error('--end must be greater than --start')

    return args

if __name__ == '__main__':
    args = parse_args()

    if args.video:
        video_processor = VideoProcessor(
            video_path=args.video,
            camera=args.video_camera,
            model_path=args.model,
        )
        video_processor.process()
    else:
        nuscenes_converter = NuScenesConverter(
            dataroot=args.dataroot,
            version=args.version,
            sample_start=args.start,
            sample_end=args.end,
        )
        data = nuscenes_converter.fetch_data()
        camera_names = None

        print(f'Loaded {len(data)} frame groups for detection.')

        detector = DetectionManager(data=data, model_path=args.model, camera_names=camera_names)
        detector.display()
