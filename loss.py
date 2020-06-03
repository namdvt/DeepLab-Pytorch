import torch.nn as nn
import torch
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets.long())
        return ce_loss

        # numerator = 2 * torch.sum(outputs * targets) + self.smooth
        # denominator = torch.sum(outputs ** 2) + torch.sum(targets ** 2) + self.smooth
        # soft_dice_loss = 1 - numerator / denominator

        # return soft_dice_loss + ce_loss
