dataset_type = 'FSIDataset'
data_root = '/media/syx/新加卷/MyResearch/VOC-FSI'
train_pipeline = [
    dict(type='LoadSLCMatFromFile'),
    dict(
        type='MCAnalysis',
        look_num=6,
        stride=60,
        aperture_size=600,
        random_factor=0.1,
        rm_taylor=24,
        ad_taylor=30,
        std_num=2.0),
    dict(type='RayleighQuan', SelRatio=7.5, ScaRatio=4.5, random_range=0.1),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='oc'),
    dict(type='Pad', size_divisor=32),
    dict(type='MCADefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_mca', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadSLCMatFromFile'),
    dict(
        type='MCAnalysis',
        look_num=6,
        stride=60,
        aperture_size=600,
        random_factor=0.0,
        rm_taylor=24,
        ad_taylor=30,
        std_num=2.0),
    dict(type='RayleighQuan', SelRatio=7.5, ScaRatio=4.5, random_range=0.0),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Pad', size_divisor=32),
            dict(type='MCADefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'img_mca'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='FSIDataset',
        ann_file='/media/syx/新加卷/MyResearch/VOC-FSI/trainval/Annotations/',
        img_prefix='/media/syx/新加卷/MyResearch/VOC-FSI/trainval/SLCMats/',
        pipeline=[
            dict(type='LoadSLCMatFromFile'),
            dict(
                type='MCAnalysis',
                look_num=6,
                stride=60,
                aperture_size=600,
                random_factor=0.1,
                rm_taylor=24,
                ad_taylor=30,
                std_num=2.0),
            dict(
                type='RayleighQuan',
                SelRatio=7.5,
                ScaRatio=4.5,
                random_range=0.1),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(1024, 1024)),
            dict(
                type='RRandomFlip',
                flip_ratio=[0.25, 0.25, 0.25],
                direction=['horizontal', 'vertical', 'diagonal'],
                version='oc'),
            dict(type='Pad', size_divisor=32),
            dict(type='MCADefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'img_mca', 'gt_bboxes', 'gt_labels'])
        ],
        version='oc'),
    val=dict(
        type='FSIDataset',
        ann_file='/media/syx/新加卷/MyResearch/VOC-FSI/test/Annotations/',
        img_prefix='/media/syx/新加卷/MyResearch/VOC-FSI/test/SLCMats/',
        pipeline=[
            dict(type='LoadSLCMatFromFile'),
            dict(
                type='MCAnalysis',
                look_num=6,
                stride=60,
                aperture_size=600,
                random_factor=0.0,
                rm_taylor=24,
                ad_taylor=30,
                std_num=2.0),
            dict(
                type='RayleighQuan',
                SelRatio=7.5,
                ScaRatio=4.5,
                random_range=0.0),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(type='Pad', size_divisor=32),
                    dict(type='MCADefaultFormatBundle'),
                    dict(type='Collect', keys=['img', 'img_mca'])
                ])
        ],
        version='oc'),
    test=dict(
        type='FSIDataset',
        ann_file='/media/syx/新加卷/MyResearch/VOC-FSI/test/Annotations/',
        img_prefix='/media/syx/新加卷/MyResearch/VOC-FSI/test/SLCMats/',
        pipeline=[
            dict(type='LoadSLCMatFromFile'),
            dict(
                type='MCAnalysis',
                look_num=6,
                stride=60,
                aperture_size=600,
                random_factor=0.0,
                rm_taylor=24,
                ad_taylor=30,
                std_num=2.0),
            dict(
                type='RayleighQuan',
                SelRatio=7.5,
                ScaRatio=4.5,
                random_range=0.0),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(type='Pad', size_divisor=32),
                    dict(type='MCADefaultFormatBundle'),
                    dict(type='Collect', keys=['img', 'img_mca'])
                ])
        ],
        version='oc'))
evaluation = dict(interval=1, metric='mAP')
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
angle_version = 'oc'
model = dict(
    type='MCADetector',
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    backbone_mca=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RotatedRPNHead',
        in_channels=256,
        feat_channels=256,
        version='oc',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='RoITransRoIHead',
        version='oc',
        num_stages=2,
        stage_loss_weights=[1, 1],
        bbox_roi_extractor=[
            dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            dict(
                type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=7,
                    sample_num=2,
                    clockwise=True),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32])
        ],
        bbox_head=[
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHAHBBoxCoder',
                    angle_range='oc',
                    norm_factor=None,
                    edge_swap=False,
                    target_means=[0.0, 0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2, 1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='RotatedShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHAOBBoxCoder',
                    angle_range='oc',
                    norm_factor=None,
                    edge_swap=False,
                    proj_xy=False,
                    target_means=[0.0, 0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1, 0.5]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='BboxOverlaps2D')),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                sampler=dict(
                    type='RRandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='oc', iou_thr=0.1),
            max_per_img=2000)))
work_dir = '/media/syx/新加卷/MyResearch/mmrotate/workdir/20250114/fsi_pearson'
auto_resume = False
gpu_ids = range(0, 1)
