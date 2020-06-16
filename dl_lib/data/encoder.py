import torch
import math
import numpy as np
from abc import ABCMeta, abstractmethod
from dl_lib.network.generator.centernet_gt import CenterNetGT
from dl_lib.builder import ENCODERS


class AnnotationEncoder(metaclass=ABCMeta):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)

    def encode(self,data, training=True):
        if training:
            return self.encode_train(data)
        else:
            return self.encode_test(data)

    @abstractmethod
    def encode_train(self,data):
        pass

    @abstractmethod
    def encode_test(self, data):
        pass

@ENCODERS.register_module()
class CenterNetDetectionEncoder(AnnotationEncoder):
    def encode_train(self,data):
        image_tensor=[]
        image_metas=[]
        for item in data:
            image_tensor.append(item['image'])
            image_meta={}
            for k in item.keys():
                if k !='image' and k!='instances':
                    image_meta[k]=item[k]
            image_metas.append(image_meta)
        gt_dict = CenterNetGT.generate(self.cfg, data)
        image_tensor = torch.stack(image_tensor, dim=0)
        return image_tensor, image_metas, gt_dict

    def encode_test(self, data):

        mean, std = self.cfg.MODEL.PIXEL_MEAN, self.cfg.MODEL.PIXEL_STD
        image_tensor = []
        image_metas = []
        for item in data:
            image_tensor.append(item['image'])
            image_meta = {}
            for k in item.keys():
                if k != 'image' and k != 'instances':
                    image_meta[k] = item[k]
            image_metas.append(image_meta)
        images = torch.stack(image_tensor, dim=0)
        pixel_mean = torch.Tensor(mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(std).to(self.device).view(3, 1, 1)
        images = images.to(self.device)
        images = (images/255.-pixel_mean)/pixel_std
        n, c, h, w = images.shape
        new_h, new_w = (h | 31) + 1, (w | 31) + 1
        center_wh = np.array([w // 2, h // 2], dtype=np.float32)
        size_wh = np.array([new_w, new_h], dtype=np.float32)
        down_scale = self.cfg.MODEL.CENTERNET.DOWN_SCALE
        img_info = dict(center=center_wh, size=size_wh,
                        height=new_h // down_scale,
                        width=new_w // down_scale)

        pad_value = [-x / y for x, y in zip(mean, std)]
        aligned_img = torch.Tensor(pad_value).reshape((1, -1, 1, 1)).expand(n, c, new_h, new_w)
        aligned_img = aligned_img.to(images.device)

        pad_w, pad_h = math.ceil((new_w - w) / 2), math.ceil((new_h - h) / 2)
        aligned_img[..., pad_h:h + pad_h, pad_w:w + pad_w] = images
        return aligned_img, img_info
