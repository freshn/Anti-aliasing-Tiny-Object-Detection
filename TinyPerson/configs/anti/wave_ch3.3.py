_base_ = [
    '../_base_/models/faster_rcnn_r50_sspnet.py',
    '../_base_/datasets/tinyperson.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

custom_imports = dict(imports=['mmdet.models.backbones.wavecnet'], allow_failed_imports=False)
model = dict(
    type='FasterSSPNet',
    pretrained=None,
    backbone=dict(
        type='WaveCResNet',
        wavename='bior3.3',
        frozen_stages=1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./ckpt/wave_ch3.3.pth',
            prefix='backbone')))

optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001) # 2GPU
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12

data = dict(workers_per_gpu=4)
evaluation = dict(interval=4, metric='bbox', proposal_nums=[1000], iou_thrs=[0.5])