import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import conv1x1, Bottleneck, BasicBlock


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(UpConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
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
            Conv2d(in_channels, bottle_neck_channels, kernel_size=1, stride=(2 if down_sample else 1),
                   padding=(2 if down_sample else 1)),
            Conv2d(bottle_neck_channels, bottle_neck_channels, kernel_size=3),
            Conv2d(bottle_neck_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        if self.down_sample:
            x = self.conv_down_sample(x) + self.conv(x)
        else:
            x = self.conv_1x1(x) + self.conv(x)

        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        low_level_feature = x

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_feature

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth') # resnet101
        
        model.load_state_dict(state_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)


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
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.conv_1x1 = Conv2d(64, 16, kernel_size=1)
        self.conv = nn.Sequential(
            Conv2d(272, 128, kernel_size=3, padding=1),
            Conv2d(128, 128, kernel_size=3, padding=1),
            Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv_1x1(low_level_feature)
        x = F.interpolate(x, size=([x.shape[3] * 4, x.shape[3] * 4]), mode='bilinear', align_corners=True)
        x = torch.cat([low_level_feature, x], dim=1)
        x = self.conv(x)
        x = F.interpolate(x, size=([x.shape[3] * 4, x.shape[3] * 4]), mode='bilinear', align_corners=True)

        return x


class DeepLab(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(DeepLab, self).__init__()
        self.resnet = resnet101(pretrained=pretrained)
        self.aspp = ASPP(2048, 256)
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        x, low_level_feature = self.resnet(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feature)
        return x
