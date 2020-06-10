import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

from model import DeepLab

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
model = DeepLab(num_classes=32).to(device)
model.load_state_dict(torch.load('output/weight.pth', map_location=device))
model.eval()

if __name__ == '__main__':
    image = Image.open('data/CamVid/train/0001TP_009450.png')
    image = F.resize(image, [512, 512])
    image = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        mask = model(image.float().to(device))

    mask = torch.argmax(mask.squeeze(), dim=0)
    image = image.squeeze().permute(1, 2, 0)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Input')

    fig.add_subplot(1, 2, 2)
    plt.imshow(mask.cpu())
    plt.axis('off')
    plt.title('Output')

    plt.show()