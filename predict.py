import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from model import DeepLab
from dataset import camvid_colors
import os
from tqdm import tqdm

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
model = DeepLab(num_classes=32).to(device)
model.load_state_dict(torch.load('output/weight.pth', map_location=device))
model.eval()


if __name__ == '__main__':
    for img_name in tqdm(os.listdir('data/CamVid/test/')):
        # input
        image = Image.open('data/CamVid/test/' + img_name)
        image = F.resize(image, [512, 512])
        image = F.to_tensor(image).unsqueeze(0)

        # ground truth
        gt = Image.open('data/CamVid/test_labels/' + img_name.split('.')[0] + '_L.png')
        gt = F.resize(gt, [512, 512])

        # output
        with torch.no_grad():
            mask = model(image.float().to(device))

        mask = torch.argmax(mask.squeeze(), dim=0)
        im_mask = np.zeros([512, 512, 3]).astype(np.uint8)
        for i in range(512):
            for j in range(512):
                im_mask[i, j] = list(camvid_colors.items())[mask[i, j]][1]

        # save result
        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(image.squeeze().permute(1, 2, 0))
        plt.axis('off')
        plt.title('Input')

        fig.add_subplot(1, 3, 2)
        plt.imshow(im_mask)
        plt.axis('off')
        plt.title('Output')

        fig.add_subplot(1, 3, 3)
        plt.imshow(gt)
        plt.axis('off')
        plt.title('Ground truth')

        plt.savefig('results/' + img_name, dpi=200, bbox_inches='tight')
        plt.close()