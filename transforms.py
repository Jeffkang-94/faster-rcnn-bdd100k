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


class DomainTransfer(object):
    def __init__(self):
        super(DomainTransfer, self).__init__()
        args = parse_args()
        self.model = Cycle_Model(args)
        
        self.Domain_transfer = self.model.netG_D2N
        #self.Domain_Refiner = self.model.netG_N2D
        
        #self.Domain_transfer.load_state_dict(torch.load('./model/checkpoint_500000.pth')['G_D2N'])
        #self.Domain_transfer.load_state_dict(torch.load('./checkpoint_990000.pth')['G_D2N'])
        self.Domain_transfer.load_state_dict(torch.load('./model/noTV_norefine.pth')['G_D2N'])
        #self.Domain_Refiner.load_state_dict(torch.load('./model/checkpoint_500000.pth')['G_N2D'])
        self.Domain_transfer.eval()
        #self.Domain_Refiner.eval()
    """@torch.no_grad()
    def forward(self, img):
        #print("processing data")
        night_img = self.Domain_transfer(img)[0]
        night_img = self.Domain_Refiner(night_img)[1]
        night_img = (night_img +1)/2.0
        return night_img"""
    @torch.no_grad()
    def __call__(self, img):
        night_img = self.Domain_transfer(img)[0]
        #print(night_img.max(), night_img.min())
        #night_img = self.Domain_Refiner(night_img)[1]
        night_img = (night_img +1)/2.0
        return night_img