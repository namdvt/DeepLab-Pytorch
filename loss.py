import torch.nn as nn
import torch
import torch.nn.functional as F


class MultipleLoss(nn.Module):
    def __init__(self, smooth):
        super(MultipleLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets)

        # dice loss
        outputs = torch.softmax(outputs, dim=1)
        target_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).permute(0,3,1,2)

        numerator = 2 * torch.sum(outputs * target_one_hot) + self.smooth
        denominator = torch.sum(outputs ** 2) + torch.sum(target_one_hot ** 2) + self.smooth
        soft_dice_loss = 1 - numerator / denominator

        return soft_dice_loss + ce_loss
