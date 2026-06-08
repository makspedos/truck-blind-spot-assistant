!cp /content/drive/MyDrive/TruckDataset/v1.0-mini.tgz /content/

!mkdir /content/dataset

!pip install nuscenes-devkit numpy==1.26.4
!pip install ultralytics --no-deps

!tar -xvzf /content/v1.0-mini.tgz -C /content/dataset

