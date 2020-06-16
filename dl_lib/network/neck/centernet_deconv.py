#!/usr/bin/python3
# -*- coding:utf-8 -*-
import math

import torch.nn as nn

from dl_lib.layers import DeformConvWithOff, ModulatedDeformConvWithOff
from dl_lib.builder import NECKS

class DeconvLayer(nn.Module):

    def __init__(
        self, in_planes,
        out_planes, deconv_kernel,
        deconv_stride=2, deconv_pad=1,
        deconv_out_pad=0, modulate_deform=True,
            enable_dcn=True
    ):
        super(DeconvLayer, self).__init__()
        if modulate_deform and enable_dcn:
            self.dcn = ModulatedDeformConvWithOff(
                in_planes, out_planes,
                kernel_size=3, deformable_groups=1,
            )
        elif enable_dcn:
            self.dcn = DeformConvWithOff(
                in_planes, out_planes,
                kernel_size=3, deformable_groups=1,
            )
        else:
            self.dcn = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=5, padding=2, groups=in_planes),
                nn.ReLU(),
                nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)
            )

        self.dcn_bn = nn.BatchNorm2d(out_planes)
        self.up_sample = nn.ConvTranspose2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=deconv_kernel,
            stride=deconv_stride, padding=deconv_pad,
            output_padding=deconv_out_pad,
            bias=False,
        )
        self._deconv_init()
        self.up_bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dcn(x)
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x

    def _deconv_init(self):
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]

@NECKS.register_module()
class SequentialUpsample(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self, cfg):
        super(SequentialUpsample, self).__init__()
        # modify into config
        channels = cfg.MODEL.CENTERNET.DECONV_CHANNEL
        deconv_kernel = cfg.MODEL.CENTERNET.DECONV_KERNEL
        modulate_deform = cfg.MODEL.CENTERNET.MODULATE_DEFORM
        in_channel = channels[0]
        self.deconvs = nn.ModuleList()
        for c, k in zip(channels[1:], deconv_kernel):
            self.deconvs.append(
                DeconvLayer(
                    in_channel, c,
                    deconv_kernel=k,
                    modulate_deform=modulate_deform,
                    enable_dcn=cfg.MODEL.CENTERNET.ENABLE_DCN
                )
            )
            in_channel = c

    def forward(self, x):
        for layer in self.deconvs:
            x = layer(x)
        return x

    def init_weights(self, pretrained=False):
        pass
