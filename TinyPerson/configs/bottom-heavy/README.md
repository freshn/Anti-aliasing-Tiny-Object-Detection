# Bottom-Heavy Backbones for Tiny Object Detection
This is the official repository for the paper 'Rethinking the backbone architecture for tiny object detection'.

## Requirements
```
python = 3.7.10
pytorch = 1.10.0
cuda = 10.2
numpy = 1.21.2
mmcv-full = 1.4.7 
mmdet = 2.19.0
```


## Installation
This repository is based on the [SSPNet](https://github.com/jbwang1997/OBBDetection/blob/master/docs/oriented_model_starting.md#prepare-dataset). Please refer to [Installation](https://github.com/MingboHong/SSPNet/blob/master/README.md#how-to-use) for installation instructions.

## Usage
### Download the pre-trained checkpoint
Download the [bh_resnet.pth](https://drive.google.com/file/d/12XZE6wMNum0DedhuBufS4vl1m2sPMz2L) and put it under the folder `Anti-aliasing-Tiny-Object-Detection/TinyPerson/ckpt/`.

### Training
```
./tools/dist_train.sh configs/bottom-heavy/bh_sspnet.py 2 --work-dir work_dirs/bh_sspnet/
```

### Testing
```
./dist_test.sh configs/bottom-heavy/bh_sspnet.py work_dirs/bh_sspnet/latest.pth 2 --format-only
```

More usage please refer to [mmdetection](https://github.com/open-mmlab/mmdetection/tree/main/docs).

### Performance 
  
| Configuration | mAP<sub>tiny</sub> | mAP<sub>tiny1</sub> | mAP<sub>tiny2</sub> | mAP<sub>tiny3</sub> | Checkpoint |
|:------------------------|:--------:|:--------:|:--------:|:--------:|:--------:|
| faster_rcnn_bh50_sspnet | 59.23 | 48.19 | 61.39 | 67.34 | [Google Drive](https://drive.google.com/file/d/14E9wmF_-anIP0YP73ub5SZ8di-RsAA7R)                  

## Citation

If you use this codebase or idea, please cite our paper:
```
@inproceedings{DBLP:conf/visapp/NingGS23,
  author       = {Jinlai Ning and
                  Haoyan Guan and
                  Michael W. Spratling},
  title        = {Rethinking the Backbone Architecture for Tiny Object Detection},
  booktitle    = {Proceedings of the 18th International Joint Conference on Computer
                  Vision, Imaging and Computer Graphics Theory and Applications, {VISIGRAPP}
                  2023, Volume 5: VISAPP, Lisbon, Portugal, February 19-21, 2023},
  pages        = {103--114},
  publisher    = {{SCITEPRESS}},
  year         = {2023},
  url          = {https://doi.org/10.5220/0011643500003417},
}
```

## Acknowledgement
This work is developed based on the [MMDetection](https://github.com/open-mmlab/mmdetection) and [SSPNet](https://github.com/MingboHong/SSPNet).
