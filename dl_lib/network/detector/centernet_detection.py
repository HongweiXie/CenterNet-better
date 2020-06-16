import torch
import math
import numpy as np
from dl_lib.structures import Instances
from dl_lib.builder import DETECTORS
from .single_stage import SingleStageDetector

@DETECTORS.register_module()
class CenterNetDetection(SingleStageDetector):

    def __init__(self,
                 cfg,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CenterNetDetection, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                 test_cfg, pretrained)
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.to(self.device)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_dict):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_dict (dict[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(img.to(self.device))
        outs = self.bbox_head(x)
        losses = self.bbox_head.loss(outs, gt_dict)
        return losses

    def simple_test(self, images, img_metas, rescale=False):
        x = self.extract_feat(images)
        outs = self.bbox_head(x)
        results = self.bbox_head.get_bboxes(outs, img_metas)
        ori_w, ori_h = img_metas['center'] * 2
        det_instance = Instances((int(ori_h), int(ori_w)), **results)
        return [{"instances": det_instance}]

