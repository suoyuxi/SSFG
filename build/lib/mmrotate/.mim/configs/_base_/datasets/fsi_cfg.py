# dataset settings
dataset_type = 'FSIDataset'
data_root = '/media/syx/新加卷/MyResearch/VOC-FSI/'
train_pipeline = [
    dict(type='LoadSLCMatFromFile'),
    dict(type='RayleighQuan', SelRatio=7.5, ScaRatio=4.5, random_range=0.1),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadSLCMatFromFile'),
    dict(type='RayleighQuan', SelRatio=7.5, ScaRatio=4.5, random_range=0.0),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval/Annotations/',
        img_prefix=data_root + 'trainval/SLCMats/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/Annotations/',
        img_prefix=data_root + 'test/SLCMats/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/Annotations/',
        img_prefix=data_root + 'test/SLCMats/',
        pipeline=test_pipeline))
