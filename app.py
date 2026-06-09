import argparse

from config import DEFAULT_DATAROOT, DEFAULT_MODEL_PATH, DEFAULT_NUSCENES_VERSION, DEFAULT_SAMPLE_END, DEFAULT_SAMPLE_START
from detector.detection_manager import DetectionManager
from detector.nuscenes_converter import NuScenesConverter


def parse_args():
    parser = argparse.ArgumentParser(description='Run truck blind spot detection on nuScenes camera frames.')
    parser.add_argument('--dataroot', default=DEFAULT_DATAROOT)
    parser.add_argument('--version', default=DEFAULT_NUSCENES_VERSION)
    parser.add_argument('--start', type=int, default=DEFAULT_SAMPLE_START)
    parser.add_argument('--end', type=int, default=DEFAULT_SAMPLE_END)
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH)
    return parser.parse_args()

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
