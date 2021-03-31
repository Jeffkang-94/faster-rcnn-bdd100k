import json
import os
from pathlib import Path

import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import glob
import torch
import transforms as T
import utils
from torch import Tensor, nn
from torch.utils.data import Dataset


def get_ground_truths(train_img_path_list, anno_data):

    bboxes, total_bboxes = [], []
    labels, total_labels = [], []
    total_img_path = []
    classes = {
                "bike":0,
                "bus":1,
                "car":2,
                "motor":3,
                "person":4,
                "rider":5,
                "traffic light":6,
                "traffic sign":7,
                "train":8,
                "truck":9
    }
    count = 0 
    for i in tqdm(range(len(train_img_path_list))):
        if anno_data[i]["attributes"]['timeofday'] =='night' or anno_data[i]["attributes"]['timeofday'] =='daytime':
            total_img_path.append(train_img_path_list[i])
            for j in range(len(anno_data[i]["labels"])):
                if "box2d" in anno_data[i]["labels"][j]:
                    xmin = anno_data[i]["labels"][j]["box2d"]["x1"]
                    ymin = anno_data[i]["labels"][j]["box2d"]["y1"]
                    xmax = anno_data[i]["labels"][j]["box2d"]["x2"]
                    ymax = anno_data[i]["labels"][j]["box2d"]["y2"]
                    bbox = [xmin, ymin, xmax, ymax]
                    category = anno_data[i]["labels"][j]["category"]
                    cls = classes[category]

                    bboxes.append(bbox)
                    labels.append(cls)
            
            total_bboxes.append(torch.tensor(bboxes))
            total_labels.append(torch.tensor(labels, dtype=torch.int64))
            bboxes = []
            labels = []
            count += 1

    return total_bboxes, total_labels, total_img_path, count 

def get_ground_truths_from_files(image_dir, anno_data):

    bboxes, total_bboxes = [], []
    labels, total_labels = [], []
    paths, total_paths = [],[]
    classes = {
                "bike":0,
                "bus":1,
                "car":2,
                "motor":3,
                "person":4,
                "rider":5,
                "traffic light":6,
                "traffic sign":7,
                "train":8,
                "truck":9
    }
    count = 0 
    for i,label_path in enumerate(anno_data): # train/*.json
        label = _load_json(label_path)
        image_path = os.path.join(
            image_dir, os.path.basename(label_path).replace('.json', '.jpg'))
        if label['attributes']['timeofday'] == 'daytime':
            total_paths.append(image_path)
            for j in range(len(label['labels'])):
                if os.path.exists(image_path) and 'box2d' in label['labels'][j]:
                    xmin = label['labels'][j]['box2d']['x1'] 
                    ymin = label['labels'][j]['box2d']['y1'] 
                    xmax = label['labels'][j]['box2d']['x2'] 
                    ymax = label['labels'][j]['box2d']['y2'] 
                    bbox = [xmin, ymin, xmax, ymax]
                    category = label['labels'][j]['category']
                    cls = classes[category]
                    bboxes.append(bbox)
                    labels.append(cls)
                    
                    
                total_bboxes.append(torch.tensor(bboxes))
                total_labels.append(torch.tensor(labels, dtype=torch.int64))
                
                
                bboxes = []
                labels = []
            count += 1

    return total_bboxes, total_labels, total_paths ,count

def _load_json(path_list_idx):
    with open(path_list_idx, "r") as file:
        data = json.load(file)
    return data


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    # 2.8125 height  y 
    # 2.5 width  x
    #if train:
        # 720 , 1280
    transforms.append(T.resize((256,512)))
    transforms.append(T.RandomHorizontalFlip(0.5))
    
    return T.Compose(transforms)


class BDD(torch.utils.data.Dataset):
    def __init__(
        self, img_path, anno_json_path, transforms=None
    ):  # total_bboxes_list,total_labels_list,transforms=None):
        super(BDD, self).__init__()
        self.img_path = img_path # 이걸 day path로 변경
        #self.train_img_path  = os.path.join('/mnt2/datasets/bdd100k', 'images/100k/train')
        self.anno_data = _load_json(anno_json_path)
        self.total_bboxes_list, self.total_labels_list, self.total_img_path, self.count = get_ground_truths(
            self.img_path, self.anno_data
        )
        #self.anno_data = glob.glob('/mnt2/datasets/bdd100k/labels/train/*json')
        #self.total_bboxes_list, self.total_labels_list, self.length = get_ground_truths_from_files(
        #    self.train_img_path, self.anno_data
        #)
        print("total dataset : {}".format(self.count))
        self.transforms = transforms
        self.classes = {
                "bike":0,
                "bus":1,
                "car":2,
                "motor":3,
                "person":4,
                "rider":5,
                "traffic light":6,
                "traffic sign":7,
                "train":8,
                "truck":9
        }

    def __len__(self):
        return self.count 

    def __getitem__(self, idx):
        img_path = self.total_img_path[idx]
        img = Image.open(img_path).convert("RGB")

        labels = self.total_labels_list[idx]
        bboxes = self.total_bboxes_list[idx]
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        

        img_id = Tensor([idx])
        iscrowd = torch.zeros(len(bboxes,), dtype=torch.int64)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
