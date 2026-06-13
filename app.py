import argparse

from config import DEFAULT_DATAROOT, DEFAULT_MODEL_PATH, DEFAULT_NUSCENES_VERSION, DEFAULT_SAMPLE_END, DEFAULT_SAMPLE_START
from detector.detection_manager import DetectionManager
from detector.nuscenes_converter import NuScenesConverter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run truck blind spot detection on nuScenes camera frames.')
    parser.add_argument('--dataroot', default=DEFAULT_DATAROOT, help='Path to the nuScenes dataset folder.')
    parser.add_argument('--version', default=DEFAULT_NUSCENES_VERSION, help='nuScenes dataset version.')
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
    nuscenes_converter = NuScenesConverter(
        dataroot=args.dataroot,
        version=args.version,
        sample_start=args.start,
        sample_end=args.end,
    )
    nusc_data = nuscenes_converter.fetch_data()
    print(nusc_data)

    detector = DetectionManager(data=nusc_data, model_path=args.model)
    detector.display()
