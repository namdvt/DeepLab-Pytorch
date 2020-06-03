from pycocotools.coco import COCO
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


class CocoDataset(Dataset):
    def __init__(self, data_folder, anns_file, size):
        super(CocoDataset, self).__init__()
        self.data_folder = data_folder
        self.size = size

        self.coco = COCO(anns_file)
        self.img_list = list(self.coco.imgs.keys())
        self.category_ids = self.coco.getCatIds()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image_id = self.img_list[index]
        image = Image.open(self.data_folder + self.coco.imgs.get(image_id).get('file_name'))
        image = F.to_tensor(F.resize(image, [self.size, self.size]))
        if image.shape[0] != 3:
            image = image.expand(3, self.size, self.size)

        annotations_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids, iscrowd=None)
        annotations = self.coco.loadAnns(annotations_ids)

        mask = np.zeros((self.size, self.size))
        for ann in annotations:
            mask = np.maximum(mask, cv2.resize(self.coco.annToMask(ann), (self.size, self.size)) * (ann['category_id'] - 1))

        # binary_mask = np.zeros((90, self.size, self.size))
        # for ann in annotations:
        #     binary_mask[ann['category_id'] - 1] += cv2.resize(self.coco.annToMask(ann), (self.size, self.size))

        return image, mask


if __name__ == '__main__':
    anns_file = 'data/annotations/instances_val2017.json'
    data_folder = 'data/val2017/'
    size = 512

    coco_dataset = CocoDataset(anns_file=anns_file, data_folder=data_folder, size=size)
    coco_dataloader = DataLoader(coco_dataset, batch_size=2, shuffle=True, drop_last=True)
    for image, mask in coco_dataloader:
        print()

