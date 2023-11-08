# Anti-aliasing-Tiny-Object-Detection
This is the official repository for the paper 'The Importance of Anti-Aliasing in Tiny Object Detection'.

## Requirements
```
python = 3.7.10
pytorch = 1.10.0
cuda = 10.2
numpy = 1.21.2
mmcv-full = 1.4.7 
mmdet = 2.19.0
```

## Dataset
### TinyPerson
Following [SSPNet](https://github.com/jbwang1997/OBBDetection/blob/master/docs/oriented_model_starting.md#prepare-dataset) to prepare the TinyPerson.
### WiderFace
Following [vedadet](https://github.com/jbwang1997/OBBDetection/blob/master/docs/oriented_model_starting.md#prepare-dataset) to prepare the WiderFace.
### DOTA
Following [OBBDetection](https://github.com/jbwang1997/OBBDetection/blob/master/docs/oriented_model_starting.md#prepare-dataset) to prepare the DOTA.

## Installation
Please refer to [Installation](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation instructions.

## Usage
### Training
```
./tools/dist_train.sh configs/anti/bhwave_ch3.3.py 4 --cfg-options optimizer.lr=0.004 --work-dir work_dirs/bhwave_ch3.3/
```

### Testing
```
./dist_test.sh configs/anti/bhwave_ch3.3.py work_dirs/bhwave_ch3.3/latest.pth 2 --format-only
```

More usage please refer to [mmdetection](https://github.com/open-mmlab/mmdetection/tree/main/docs).

## Citation

If you use this codebase or idea, please cite our paper:
```
@ARTICLE{ning2023acml,
       author = {{Ning}, Jinlai and {Spratling}, Michael},
        title = {The Importance of Anti-Aliasing in Tiny Object Detection},
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = {2023},
        month = {oct},
}
```

## Acknowledgement
This work is developed based on the [MMDetection](https://github.com/open-mmlab/mmdetection), [SSPNet](https://github.com/MingboHong/SSPNet), [vedadet](https://github.com/Media-Smart/vedadet) and [OBBDetection](https://github.com/jbwang1997/OBBDetection). 

