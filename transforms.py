import random

from torchvision.transforms import functional as F

import torch
import torch.nn as nn


from model.train_model import Cycle_Model
from model.config import parse_args
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class resize(object):
    def __init__(self, dim):
        self.dims = dim 

    def __call__(self, image, target, return_percent_coords=True):
       
        # Resize image
        new_image = F.resize(image, size=self.dims)
        #new_image =image.resize

        # Resize bounding boxes
        height, width = image.shape[-2:]
        height /= self.dims[0]
        width /= self.dims[1]
        # x,y,x,y width,height
        old_dims = torch.FloatTensor([width, height, width, height]).unsqueeze(0)
        new_boxes = target["boxes"] / old_dims  # percent coordinates


        if not return_percent_coords:
            new_dims = torch.FloatTensor([self.dims[1], self.dims[0], self.dims[1], self.dims[0]]).unsqueeze(0)
            new_boxes = new_boxes * new_dims
        target["boxes"] = new_boxes
        return new_image, target

class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

