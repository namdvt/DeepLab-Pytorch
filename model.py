import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=False):
        super(ResBlock, self).__init__()
        bottle_neck_channels = int(out_channels / 4)
        self.down_sample = down_sample
        self.conv_down_sample = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv_1x1 = Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv = nn.Sequential(
            Conv2d(in_channels, bottle_neck_channels, kernel_size=1, stride=(2 if down_sample else 1), padding=(2 if down_sample else 1)),
            Conv2d(bottle_neck_channels, bottle_neck_channels, kernel_size=3),
            Conv2d(bottle_neck_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        if self.down_sample:
            x = self.conv_down_sample(x) + self.conv(x)
        else:
            x = self.conv_1x1(x) + self.conv(x)

        return x


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            nn.AvgPool2d(kernel_size=2, ceil_mode=True),
            ResBlock(64, 64), ResBlock(64, 64), ResBlock(64, 64)
        )
        self.conv2 = nn.Sequential(
            ResBlock(64, 128, down_sample=True), ResBlock(128, 128), ResBlock(128, 128), ResBlock(128, 128),
            ResBlock(128, 256, down_sample=True), ResBlock(256, 256), ResBlock(256, 256), ResBlock(256, 256), ResBlock(256, 256), ResBlock(256, 256),
            ResBlock(256, 512), ResBlock(512, 512), ResBlock(512, 512)
        )

    def forward(self, x):
        low_level_feature = self.conv1(x)
        x = self.conv2(low_level_feature)

        return x, low_level_feature


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3 = Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.conv4 = Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x2 = Conv2d(in_channels + 4 * out_channels, 256, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv_1x2(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv_1x1 = Conv2d(64, 16, kernel_size=1)
        self.conv = nn.Sequential(
            Conv2d(272, 128, kernel_size=3, padding=1),
            Conv2d(128, 128, kernel_size=3, padding=1),
            Conv2d(128, 90, kernel_size=1)
        )

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv_1x1(low_level_feature)
        x = F.interpolate(x, size=([64, 64]), mode='bilinear', align_corners=True)
        x = torch.cat([low_level_feature, x], dim=1)
        x = self.conv(x)
        x = F.interpolate(x, size=([256, 256]), mode='bilinear', align_corners=True)

        return x


class DeepLab(nn.Module):
    def __init__(self):
        super(DeepLab, self).__init__()
        self.resnet = Resnet()
        self.aspp = ASPP(512, 256)
        self.decoder = Decoder()

    def forward(self, x):
        x, low_level_feature = self.resnet(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feature)
        return x
