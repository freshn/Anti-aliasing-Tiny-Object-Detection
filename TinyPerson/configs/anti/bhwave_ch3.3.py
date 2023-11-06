_base_ = [
    '../_base_/models/shallow50_sspnet.py',
    '../_base_/datasets/tinyperson.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

custom_imports = dict(imports=['mmdet.models.backbones.bh_wavecnet_resnet'], allow_failed_imports=False)
model = dict(
    type='FasterSSPNet',
    pretrained=None,
    backbone=dict(
        type='BHWaveCResNet',
        wavename='bior3.3',
        frozen_stages=1,
        init_cfg=dict(type='Pretrained',checkpoint='../mmclassification/work_dirs/bh_wavecnet_ch3.3_in1k/epoch_150.pth',prefix='backbone')))

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