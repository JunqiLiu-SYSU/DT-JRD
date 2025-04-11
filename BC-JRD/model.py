""""
The binary OW-JRD predictor
refercence：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing
"""
import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

# EfficientNetV2
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ConvBNAct(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result

class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,
                 expand_c: int,
                 se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x

class MBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)

        activation_layer = nn.SiLU
        expanded_c = input_c * expand_ratio

        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = ConvBNAct(input_c,
                                     expanded_c,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer)

        # Depth-wise convolution
        self.dwconv = ConvBNAct(expanded_c,
                                expanded_c,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)

        self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()

        # Point-wise linear projection
        self.project_conv = ConvBNAct(expanded_c,
                                      out_planes=out_c,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity)

        self.out_channels = out_c

        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result

class FusedMBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(FusedMBConv, self).__init__()

        assert stride in [1, 2]
        assert se_ratio == 0

        self.has_shortcut = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        activation_layer = nn.SiLU
        expanded_c = input_c * expand_ratio

        if self.has_expansion:
            self.expand_conv = ConvBNAct(input_c,
                                         expanded_c,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)

            self.project_conv = ConvBNAct(expanded_c,
                                          out_c,
                                          kernel_size=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity)
        else:
            self.project_conv = ConvBNAct(input_c,
                                          out_c,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          activation_layer=activation_layer)

        self.out_channels = out_c
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)

            result += x

        return result

class EfficientNetV2_backbone(nn.Module):
    def __init__(self,
                 model_cnf: list,
                 drop_connect_rate: float = 0.2):
        super(EfficientNetV2_backbone, self).__init__()

        for cnf in model_cnf:
            assert len(cnf) == 8

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        stem_filter_num = model_cnf[0][4]

        self.stem = ConvBNAct(3,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        for kk, cnf in enumerate(model_cnf):
            blocks.append([])
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks[kk].append(op(kernel_size=cnf[1],
                                     input_c=cnf[4] if i == 0 else cnf[5],
                                     out_c=cnf[5],
                                     expand_ratio=cnf[3],
                                     stride=cnf[2] if i == 0 else 1,
                                     se_ratio=cnf[-1],
                                     drop_rate=drop_connect_rate * block_id / total_blocks,
                                     norm_layer=norm_layer))
                block_id += 1

        self.blocks0 = nn.Sequential(*blocks[0])
        self.blocks1 = nn.Sequential(*blocks[1])
        self.blocks2 = nn.Sequential(*blocks[2])
        self.blocks3 = nn.Sequential(*blocks[3])

        self.AVP = nn.AdaptiveAvgPool2d(40)  # 640x640

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x0 = self.blocks0(x)
        x1 = self.blocks1(x0)
        x2 = self.blocks2(x1)
        x3 = self.blocks3(x2)
        x0 = self.AVP(x0)
        x1 = self.AVP(x1)
        x2 = self.AVP(x2)
        return x0, x1, x2, x3

class EffTrainableCNN(nn.Module):
    def __init__(self,
                 model_cnf: list):
        super(EffTrainableCNN, self).__init__()

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
        backbone_output_c = sum([cnf[-3] for cnf in model_cnf]) * 3
        trainableCNN = []

        trainableCNN.append(ConvBNAct(in_planes=backbone_output_c,
                                      out_planes=640,
                                      kernel_size=1,
                                      stride=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.ReLU))

        for i in range(2):
            trainableCNN.append(ConvBNAct(in_planes=640,
                                          out_planes=640,
                                          kernel_size=3,
                                          stride=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.ReLU))

        self.trainableCNN = nn.Sequential(*trainableCNN)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.trainableCNN(x)
        return x

class EffProjectionHead(nn.Module):
    def __init__(self,
                 model_cnf: list,
                 num_classes: int = 2,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2):
        super(EffProjectionHead, self).__init__()
        for cnf in model_cnf:
            assert len(cnf) == 8

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        head_input_c = 640
        head = OrderedDict()

        head.update({"project_conv": ConvBNAct(head_input_c,
                                               num_features,
                                               kernel_size=1,       # 这里把project_conv的大小更新为(2,2)了，导致权重文件和模型不匹配
                                               norm_layer=norm_layer)})

        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})

        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
        head.update({"classifier": nn.Linear(num_features, num_classes)})

        self.head = nn.Sequential(head)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.head(x)
        return x

class EffPredictor(nn.Module):
    def __init__(self,
                 model_cnf: list,
                 num_classes: int = 2,
                 dropout_rate=0.2):
        super(EffPredictor, self).__init__()

        self.EB = EfficientNetV2_backbone(model_cnf=model_cnf)

        self.TC = EffTrainableCNN(model_cnf=model_cnf)

        self.PH = EffProjectionHead(model_cnf=model_cnf,
                                    num_classes=num_classes,
                                    dropout_rate=dropout_rate)

    def forward(self, ori, dis):
        ori_x0, ori_x1, ori_x2, ori_x3 = self.EB(ori)
        dis_x0, dis_x1, dis_x2, dis_x3 = self.EB(dis)
        x = torch.cat((ori_x0, dis_x0, ori_x1, dis_x1,
                       ori_x2, dis_x2, ori_x3, dis_x3,
                       ori_x0 - dis_x0, ori_x1 - dis_x1,
                       ori_x2 - dis_x2, ori_x3 - dis_x3), 1)
        x = self.TC(x)
        x = self.PH(x)
        return x

def CreateEffModel(num_classes: int = 2, dropout_rate: float = 0):
    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25]]

    model = EffPredictor(model_cnf=model_config,
                         num_classes=num_classes,
                         dropout_rate=dropout_rate)
    return model


# Resnet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 blocks_num,
                 model_layers,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if model_layers == 18:
            self.AVP = nn.AdaptiveAvgPool2d(20)  # 640x640
        elif model_layers == 34:
            self.AVP = nn.AdaptiveAvgPool2d(20)  # 640x640
        elif model_layers == 50:
            self.AVP = nn.AdaptiveAvgPool2d(20)  # 640x640

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x1 = self.AVP(x1)
        x2 = self.AVP(x2)
        x3 = self.AVP(x3)

        if self.include_top:
            x4 = self.avgpool(x4)
            x4 = torch.flatten(x4, 1)
            x4 = self.fc(x4)
        return x1, x2, x3, x4

class ResTrainableCNN(nn.Module):
    def __init__(self,
                 block,
                 model_layers):
        super(ResTrainableCNN, self).__init__()

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)
        if model_layers == 18 or model_layers == 34:
            backbone_output_c = sum([64, 128, 256, 512]) * 3
        if model_layers == 50:
            backbone_output_c = sum([256, 512, 1024, 2048]) * 3
        trainableCNN = []

        trainableCNN.append(ConvBNAct(in_planes=backbone_output_c,
                                      out_planes=640,
                                      kernel_size=1,
                                      stride=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.ReLU))

        for i in range(2):
            trainableCNN.append(ConvBNAct(in_planes=640,
                                          out_planes=640,
                                          kernel_size=3,
                                          stride=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.ReLU))

        self.trainableCNN = nn.Sequential(*trainableCNN)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.trainableCNN(x)
        return x

class ResProjectionHead(nn.Module):
    def __init__(self,
                 num_classes: int = 2,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2):
        super(ResProjectionHead, self).__init__()

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        head_input_c = 640
        head = OrderedDict()

        head.update({"project_conv": ConvBNAct(head_input_c,
                                               num_features,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})

        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})

        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
        head.update({"classifier": nn.Linear(num_features, num_classes)})

        self.head = nn.Sequential(head)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.head(x)
        return x

class ResPredictor(nn.Module):
    def __init__(self,
                 model_layers: int = 34,
                 num_classes: int = 2,
                 dropout_rate: float = 0.2):
        super(ResPredictor, self).__init__()

        if model_layers == 18:
            self.EB = resnet18(num_classes=num_classes, include_top=False)
            self.TC = ResTrainableCNN(block=BasicBlock, model_layers=model_layers)
        if model_layers == 34:
            self.EB = resnet34(num_classes=num_classes, include_top=False)
            self.TC = ResTrainableCNN(block=BasicBlock, model_layers=model_layers)
        elif model_layers == 50:
            self.EB = resnet50(num_classes=num_classes, include_top=False)
            self.TC = ResTrainableCNN(block=Bottleneck, model_layers=model_layers)

        self.PH = ResProjectionHead(num_classes=num_classes,
                                    dropout_rate=dropout_rate)

    def forward(self, ori, dis):
        ori_x0, ori_x1, ori_x2, ori_x3 = self.EB(ori)
        dis_x0, dis_x1, dis_x2, dis_x3 = self.EB(dis)
        x = torch.cat((
            ori_x0, dis_x0, ori_x1, dis_x1, ori_x2, dis_x2, ori_x3, dis_x3, ori_x0 - dis_x0, ori_x1 - dis_x1,
            ori_x2 - dis_x2, ori_x3 - dis_x3), 1)
        x = self.TC(x)
        x = self.PH(x)
        return x

def resnet18(num_classes=2, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top, model_layers=18)

def resnet34(num_classes=2, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, model_layers=34)

def resnet50(num_classes=2, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, model_layers=50)

def CreateResModel(num_classes: int = 2, dropout_rate: float = 0, model_layers: int = 18):
    model = ResPredictor(model_layers=model_layers,
                         num_classes=num_classes,
                         dropout_rate=dropout_rate)
    return model


def CreateModel(backbone: str = 'Eff', num_classes: int = 2, dropout_rate: float = 0, model_layers: int = 34):
    if backbone == 'Eff':
        return CreateEffModel(num_classes=num_classes, dropout_rate=dropout_rate)
    elif backbone == 'Res':
        return CreateResModel(num_classes=num_classes, dropout_rate=dropout_rate, model_layers=model_layers)