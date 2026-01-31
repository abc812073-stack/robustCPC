import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torch.hub import load_state_dict_from_url


class SplAtConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, radix=2, reduction_factor=4, norm_layer=None):
        super(SplAtConv2d, self).__init__()
        self.radix = radix
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * radix, kernel_size, stride, padding, dilation, groups * radix, bias=bias)
        self.bn0 = norm_layer(out_channels * radix)
        self.relu = nn.ReLU(inplace=True)
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.fc1 = nn.Conv2d(out_channels, inter_channels, 1, groups=self.groups)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, out_channels * radix, 1, groups=self.groups)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splits = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splits)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)

        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        if self.radix > 1:
            atten = self.rsoftmax(atten).view(batch, -1, 1, 1)
            out = sum([att * split for (att, split) in zip(torch.split(atten, rchannel // self.radix, dim=1), splits)])
        else:
            atten = torch.sigmoid(atten).view(batch, -1, 1, 1)
            out = atten * x
        return out.contiguous()

class rSoftMax(nn.Module):
    def __init__(self, radix, groups):
        super(rSoftMax, self).__init__()
        self.radix = radix
        self.groups = groups

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2).contiguous()
            x = F.softmax(x, dim=1)
            x = x.view(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x



class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=2, bottleneck_width=64, avd=False, avd_first=False, is_first=False, reduction_factor=4, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        group_width = int(planes * (bottleneck_width / 64.)) * radix
        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.conv2 = SplAtConv2d(group_width, group_width, kernel_size=3, stride=1 if self.avd else stride, padding=1, groups=radix, bias=False, radix=radix, reduction_factor=reduction_factor, norm_layer=norm_layer)
        self.conv3 = nn.Conv2d(group_width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNeSt(nn.Module):

    def __init__(self, block, layers, radix=2, groups=1, bottleneck_width=64, num_classes=1000, deep_stem=False, stem_width=64, avg_down=False, avd=False, avd_first=False, norm_layer=None, input_channel=3):
        super(ResNeSt, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.radix = radix
        self.groups = groups
        self.bottleneck_width = bottleneck_width
        self.avg_down = avg_down
        self.avd = avd
        self.avd_first = avd_first
        self.deep_stem = deep_stem

        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_channel, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.bn1 = norm_layer(stem_width * 2)
        else:
            self.conv1 = nn.Conv2d(input_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.radix, self.bottleneck_width, self.avd, self.avd_first, is_first=False, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, radix=self.radix, bottleneck_width=self.bottleneck_width, avd=self.avd, avd_first=self.avd_first, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)       # (128, 64, 512, 384)
        x = self.bn1(x)         # (128, 64, 512, 384)
        x = self.relu(x)        # (128, 64, 512, 384)
        x = self.maxpool(x)     # (128, 64, 256, 192)

        x = self.layer1(x)      # (128, 256, 256, 192)
        x = self.layer2(x)      # (128, 512, 128, 96)
        x = self.layer3(x)      # (128, 1024, 64, 48)
        x = self.layer4(x)      # (128, 2048, 32, 24)

        # 修改这里以匹配全连接层的输入大小
        # 原始的ResNeSt模型的最后一层输出的通道数应该是512 * block.expansion
        # 你需要修改这里，使得输出的通道数符合你的全连接层的输入大小
        x = self.avgpool(x)     # (128, 2048, 1, 1)
        x = torch.flatten(x, 1) # (128, 2048)
        feature = x
        x = self.fc(x)          # (128, 1000)
        return x, feature



def resnest50(pretrained=False, progress=True, **kwargs):
    model = ResNeSt(Bottleneck, [3, 4, 6, 3], radix=2, groups=1, bottleneck_width=64, deep_stem=True, stem_width=32, avg_down=True, avd=True, avd_first=False, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url('https://hangzh.s3.amazonaws.com/encoding/models/resnest50-528c19ca.pth', progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model
