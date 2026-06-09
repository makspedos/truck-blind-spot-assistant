from models.nuscenes_converter import NuscenesConverter
from models.detection_manager import  DetectionManager

if __name__ == '__main__':
    nuscenes_converter = NuscenesConverter()
    nusc_data = nuscenes_converter.fetch_data()
    print(nusc_data)

    detector = DetectionManager(data=nusc_data)
    detector.display()

