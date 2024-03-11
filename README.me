# yolov7 sensor fusion detection 

## Installment
Follow YoloV7, 
```
conda create -n YOLOENV python=3.9 -y
conda activate YOLOENV
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip intsall -r requirement.txt
```

## Data Preparation

- **For KITTI dataset**:

  Please make sure the folder organized like follows before run Preprocess_lidar_map.py:
    ```
    ├── cfg
    ├── datasets
    │   ├── kitti
    │   │   ├── training
    │   │   ├── testing
    │   │   ├── ImageSets
    ```
    Then run the generation script
    ```
    python Preprocess_lidar_map.py
    ```
## Training
change the path inside the cfg/dataset/Kitti.yaml.
Download yolov7 from the https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```shell
python train.py --cfg cfg/training/yolov7-BGF.yaml --data cfg/dataset/Kitti.yaml --weights 'yolov7.pt' --hyp cfg/dataset/hyp.scratch.p5-kitti.yaml --epochs 100 --img 640 640 --batch-size 8 --freeze 50
```
## Testing
```shell
python test.py --cfg cfg/models/yolov7-BGF.yaml --data cfg/datasets/Kitti.yaml --weights {path_to_weights} --hyp cfg/datasets/hyp.scratch.p5-kitti.yaml --no-trace --batch-size 1 
```
