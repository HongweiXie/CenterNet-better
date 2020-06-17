#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

from dl_lib.builder import HEADS
from dl_lib.network.loss import modified_focal_loss, reg_l1_loss
from dl_lib.network.generator import CenterNetDecoder, CenterNetGT
from dl_lib.structures import Boxes


class SingleHead(nn.Module):

    def __init__(self, in_channel, out_channel, bias_fill=False, bias_value=0):
        super(SingleHead, self).__init__()
        self.feat_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            self.out_conv.bias.data.fill_(bias_value)

    def forward(self, x):
        x = self.feat_conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x

@HEADS.register_module()
class CenterNetDetectionHead(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self, cfg):
        super(CenterNetDetectionHead, self).__init__()
        self.cfg = cfg
        self.cls_head = SingleHead(
            64,
            cfg.MODEL.CENTERNET.NUM_CLASSES,
            bias_fill=True,
            bias_value=cfg.MODEL.CENTERNET.BIAS_VALUE,
        )
        self.wh_head = SingleHead(64, 2)
        self.reg_head = SingleHead(64, 2)

    def forward(self, x):
        cls = self.cls_head(x)
        cls = torch.sigmoid(cls)
        wh = self.wh_head(x)
        reg = self.reg_head(x)
        pred = {
            'cls': cls,
            'wh': wh,
            'reg': reg
        }
        return pred

    def init_weights(self, pretrained=False):
        pass

    @torch.no_grad()
    def get_ground_truth(self, annotations):
        return CenterNetGT.generate(self.cfg, annotations)

    def loss(self, pred_dict, annotations):
        gt_dict = self.get_ground_truth(annotations)
        pred_score = pred_dict['cls']
        cur_device = pred_score.device
        for k in gt_dict:
            gt_dict[k] = gt_dict[k].to(cur_device)

        loss_cls = modified_focal_loss(pred_score, gt_dict['score_map'])

        mask = gt_dict['reg_mask']
        index = gt_dict['index']
        index = index.to(torch.long)
        # width and height loss, better version
        loss_wh = reg_l1_loss(pred_dict['wh'], mask, index, gt_dict['wh'])

        # regression loss
        loss_reg = reg_l1_loss(pred_dict['reg'], mask, index, gt_dict['reg'])

        loss_cls *= self.cfg.MODEL.LOSS.CLS_WEIGHT
        loss_wh *= self.cfg.MODEL.LOSS.WH_WEIGHT
        loss_reg *= self.cfg.MODEL.LOSS.REG_WEIGHT

        loss = {
            "loss_cls": loss_cls,
            "loss_box_wh": loss_wh,
            "loss_center_reg": loss_reg,
        }
        # print(loss)
        return loss

    def get_bboxes(self, pred_dict, img_info):
        """
        Args:
            pred_dict(dict): a dict contains all information of prediction
            img_info(dict): a dict contains needed information of origin image
        """
        fmap = pred_dict["cls"]
        reg = pred_dict["reg"]
        wh = pred_dict["wh"]

        boxes, scores, classes = CenterNetDecoder.decode(fmap, wh, reg)
        # boxes = Boxes(boxes.reshape(boxes.shape[-2:]))
        scores = scores.reshape(-1)
        classes = classes.reshape(-1).to(torch.int64)

        # dets = CenterNetDecoder.decode(fmap, wh, reg)
        boxes = CenterNetDecoder.transform_boxes(boxes, img_info)
        boxes = Boxes(boxes)
        return dict(pred_boxes=boxes, scores=scores, pred_classes=classes)
