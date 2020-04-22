import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import numpy as np

'''

Just a modification of the torchvision inception model to get the before-to-last activation


'''

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1,dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1+(dilation-1), bias=False,dilation=dilation)

def convKxK(in_planes, out_planes, stride=1,dilation=1,k=(3,3),padding=True):
    """3x3 convolution with padding"""

    if padding:
        return nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=stride,
                     padding=(k[0]//2+(dilation-1),k[1]//2+(dilation-1)), bias=False,dilation=dilation)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=stride,
                     padding=((dilation-1),(dilation-1)), bias=False,dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None,feat=False,dilation=1,convKer=(3,3)):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = convKxK(inplanes, planes, stride,dilation,k=convKer)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convKxK(planes, planes,k=convKer)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.feat = feat

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if not self.feat:
            out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None,feat=False,dilation=1,convKer=(3,3)):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)

        #if convKer[0] == 1:
        #    self.conv2 = convKxK(planes, planes, stride=1,dilation,k=convKer)
        #    self.linLay = nn.Linear()
        #else:
        #    self.conv2 = convKxK(planes, planes, stride,dilation,k=convKer)

        self.conv2 = convKxK(planes, planes, stride,dilation,k=convKer)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.feat = feat

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if not self.feat:
            out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None,maxPoolKer=(3,3),\
                maxPoolPad=(1,1),stride=(2,2),convKer=(3,3),firstConvKer=(7,7),featMap=False,chan=64,inChan=3,dilation=1):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = chan
        self.conv1 = nn.Conv2d(inChan, chan, kernel_size=firstConvKer, stride=stride, padding=(firstConvKer[0]//2,firstConvKer[1]//2),bias=False)
        self.bn1 = norm_layer(chan)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=maxPoolKer, stride=stride, padding=maxPoolPad)
        self.layer1 = self._make_layer(block, chan*1, layers[0], stride=1,      norm_layer=norm_layer,feat=False,convKer=convKer)
        self.layer2 = self._make_layer(block, chan*2, layers[1], stride=stride, norm_layer=norm_layer,feat=False,dilation=dilation,convKer=convKer)
        self.layer3 = self._make_layer(block, chan*4, layers[2], stride=stride, norm_layer=norm_layer,feat=False,dilation=dilation,convKer=convKer)
        self.layer4 = self._make_layer(block, chan*8, layers[3], stride=stride, norm_layer=norm_layer,feat=True,dilation=dilation,convKer=convKer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(chan*8 * block.expansion, num_classes)

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

        self.featMap = featMap

        self.layers = [self.layer1,self.layer2,self.layer3,self.layer4]

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None,feat=False,dilation=1,convKer=(3,3)):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer,convKer=convKer))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):

            if i == blocks-1 and feat:
                layers.append(block(self.inplanes, planes, norm_layer=norm_layer,feat=True,dilation=dilation,convKer=convKer))
            else:
                layers.append(block(self.inplanes, planes, norm_layer=norm_layer,feat=False,dilation=dilation,convKer=convKer))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if not self.featMap:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

        return x

def load_state_dict(model,type):
    params = model_zoo.load_url(model_urls[type])
    paramsToLoad = {}
    for key in params:
        if key in model.state_dict() and model.state_dict()[key].size() == params[key].size():
            paramsToLoad[key] = params[key]

    res = model.load_state_dict(paramsToLoad,strict=False)
    if not res is None:
        print(res)
    return model

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model = load_state_dict(model,'resnet18')
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model = load_state_dict(model,'resnet34')
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model = load_state_dict(model,'resnet50')
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model = load_state_dict(model,'resnet101')
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model = load_state_dict(model,'resnet152')
    return model
