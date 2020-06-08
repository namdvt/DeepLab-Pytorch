# from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import tifffile
import torchvision.transforms.functional as F
import cv2
import torch.nn.functional as TF
import torch
import random
from helper import mask_to_anns_img
import os
import pandas as pd
from collections import OrderedDict


# class CocoDataset(Dataset):
#     def __init__(self, data_folder, anns_file, size):
#         super(CocoDataset, self).__init__()
#         self.data_folder = data_folder
#         self.size = size
#
#         self.coco = COCO(anns_file)
#         self.img_list = list(self.coco.imgs.keys())
#         self.category_ids = self.coco.getCatIds()
#
#     def __len__(self):
#         return len(self.img_list)
#
#     def __getitem__(self, index):
#         image_id = self.img_list[index]
#         image = Image.open(self.data_folder + self.coco.imgs.get(image_id).get('file_name'))
#         image = F.to_tensor(F.resize(image, [self.size, self.size]))
#         if image.shape[0] != 3:
#             image = image.expand(3, self.size, self.size)
#
#         annotations_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids, iscrowd=None)
#         annotations = self.coco.loadAnns(annotations_ids)
#
#         mask = np.zeros((self.size, self.size))
#         for ann in annotations:
#             mask = np.maximum(mask, cv2.resize(self.coco.annToMask(ann), (self.size, self.size)) * (ann['category_id'] - 1))
#
#         # binary_mask = np.zeros((90, self.size, self.size))
#         # for ann in annotations:
#         #     binary_mask[ann['category_id'] - 1] += cv2.resize(self.coco.annToMask(ann), (self.size, self.size))
#
#         return image, mask

camvid_colors = OrderedDict([
    ("Animal", np.array([64, 128, 64])),
    ("Archway", np.array([192, 0, 128])),
    ("Bicyclist", np.array([0, 128, 192])),
    ("Bridge", np.array([0, 128, 64])),
    ("Building", np.array([128, 0, 0])),
    ("Car", np.array([64, 0, 128])),
    ("CartLuggagePram", np.array([64, 0, 192])),
    ("Child", np.array([192, 128, 64])),
    ("Column_Pole", np.array([192, 192, 128])),
    ("Fence", np.array([64, 64, 128])),
    ("LaneMkgsDriv", np.array([128, 0, 192])),
    ("LaneMkgsNonDriv", np.array([192, 0, 64])),
    ("Misc_Text", np.array([128, 128, 64])),
    ("MotorcycleScooter", np.array([192, 0, 192])),
    ("OtherMoving", np.array([128, 64, 64])),
    ("ParkingBlock", np.array([64, 192, 128])),
    ("Pedestrian", np.array([64, 64, 0])),
    ("Road", np.array([128, 64, 128])),
    ("RoadShoulder", np.array([128, 128, 192])),
    ("Sidewalk", np.array([0, 0, 192])),
    ("SignSymbol", np.array([192, 128, 128])),
    ("Sky", np.array([128, 128, 128])),
    ("SUVPickupTruck", np.array([64, 128, 192])),
    ("TrafficCone", np.array([0, 0, 64])),
    ("TrafficLight", np.array([0, 64, 64])),
    ("Train", np.array([192, 64, 128])),
    ("Tree", np.array([128, 128, 0])),
    ("Truck_Bus", np.array([192, 128, 192])),
    ("Tunnel", np.array([64, 0, 64])),
    ("VegetationMisc", np.array([192, 192, 0])),
    ("Wall", np.array([64, 192, 0])),
    ("Void", np.array([0, 0, 0]))
])


class CamVidDataset(Dataset):
    def __init__(self, data_folder, lablel_folder, size):
        super(CamVidDataset, self).__init__()
        self.data_folder = data_folder
        self.label_folder = lablel_folder
        self.img_list = os.listdir(data_folder)
        self.size = size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image = Image.open(self.data_folder + self.img_list[index])
        image = F.resize(image, [self.size, self.size])
        label = Image.open(self.label_folder + self.img_list[index].split('.')[0] + '_L.png')
        label = F.resize(label, [self.size, self.size])

        image, label = transform(image, label)

        return image, label


def get_mask(im):
    im = np.asarray(im)
    out = (np.zeros(im.shape[:2])).astype(np.uint8)
    for gray_val, (label, rgb) in enumerate(camvid_colors.items()):
        match_pxls = ((im == rgb).sum(-1) == 3)
        out[match_pxls] = gray_val
    return torch.Tensor(out)


def transform(image, label):
    rand = random.randint(0, 1)
    if rand:
        image = F.hflip(image)
        label = F.hflip(label)

    image = F.to_tensor(image)
    label = get_mask(label)

    return image, label


if __name__ == '__main__':
    # anns_file = 'data/annotations/instances_val2017.json'
    # data_folder = 'data/val2017/'
    # size = 512
    #
    # coco_dataset = CocoDataset(anns_file=anns_file, data_folder=data_folder, size=size)
    # coco_dataloader = DataLoader(coco_dataset, batch_size=2, shuffle=True, drop_last=True)
    # for image, mask in coco_dataloader:
    #     print()

    train_folder = 'data/CamVid/train/'
    train_label_folder = 'data/CamVid/train_labels/'
    train_dataset = CamVidDataset(train_folder, train_label_folder, size=256)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
    for image, label in train_loader:
        print()

