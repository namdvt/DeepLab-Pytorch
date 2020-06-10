import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
    def __init__(self, data_folder, label_folder, size, augment=False):
        super(CamVidDataset, self).__init__()
        self.data_folder = data_folder
        self.label_folder = label_folder
        self.img_list = os.listdir(data_folder)
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        image = Image.open(self.data_folder + self.img_list[index])
        label = Image.open(self.label_folder + self.img_list[index].split('.')[0] + '_L.png')

        if self.augment:
            image, label = transform(image, label)

        image = F.to_tensor(F.resize(image, [self.size, self.size]))
        label = F.resize(label, ([self.size, self.size]))
        label = get_mask(label)
        label = torch.tensor(label)

        return image, label


def get_mask(im):
    im = np.asarray(im)
    out = (np.zeros(im.shape[:2])).astype(np.uint8)
    for gray_val, (label, rgb) in enumerate(camvid_colors.items()):
        match_pxls = ((im == rgb).sum(-1) == 3)
        out[match_pxls] = gray_val
    return torch.Tensor(out)


def transform(image, label):
    # hflip
    if random.randint(0, 1):
        image = F.hflip(image)
        label = F.hflip(label)

    # crop
    x = random.randint(0, 224)
    w = random.randint(512, image.width - x)
    y = random.randint(0, 104)
    h = random.randint(512, image.height - y)

    image = F.crop(image, x, y, h, w)
    label = F.crop(label, x, y, h, w)

    # adjust brightness, contrast, saturation
    image = F.adjust_brightness(image, random.randint(5, 15) / 10)
    image = F.adjust_contrast(image, random.randint(5, 15) / 10)
    image = F.adjust_saturation(image, random.randint(0, 5))

    return image, label


if __name__ == '__main__':
    train_folder = 'data/CamVid/train/'
    train_label_folder = 'data/CamVid/train_labels/'
    train_dataset = CamVidDataset(train_folder, train_label_folder, size=512, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
    for image, label in tqdm(train_loader):
        print()
