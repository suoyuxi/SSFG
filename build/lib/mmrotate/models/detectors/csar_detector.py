# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ROTATED_DETECTORS
from .pyramid_two_stage import PyramidTwoStageDetector


@ROTATED_DETECTORS.register_module()
class CSARDetector(PyramidTwoStageDetector):

    def __init__(self,
                 backbone,
                 pyramid,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(CSARDetector, self).__init__(
            backbone=backbone,
            pyramid=pyramid,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
