# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .utils.env import setup_environment

setup_environment()


__version__ = "0.1"

from .network.backbone import *
from .network.neck import *
from .network.head import *
from .builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                      ROI_EXTRACTORS, SHARED_HEADS, build_backbone,
                      build_detector, build_head, build_loss, build_neck,
                      build_roi_extractor, build_shared_head, build_encoder)
from .data.encoder import *

__all__ = [
    'BACKBONES', 'NECKS', 'ROI_EXTRACTORS', 'SHARED_HEADS', 'HEADS', 'LOSSES',
    'DETECTORS', 'build_backbone', 'build_neck', 'build_roi_extractor',
    'build_shared_head', 'build_head', 'build_loss', 'build_detector', 'build_encoder'
]