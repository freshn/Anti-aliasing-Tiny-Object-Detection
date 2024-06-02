_base_ = [
    '../_base_/models/faster_rcnn_bh50_sspnet.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001) # 2GPU
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12

# set more workers to speed up training 
# data = dict(workers_per_gpu=4)
evaluation = dict(interval=4, metric='bbox', proposal_nums=[1000], iou_thrs=[0.5])
