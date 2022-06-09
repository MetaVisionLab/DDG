import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

from ddg.utils import load_state_dict
from ddg.utils import MODELS_REGISTRY
from .conv import Conv2dDynamic
from .mixstyle import MixStyle

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2',
           'resnet18_backbone', 'resnet34_backbone', 'resnet50_backbone', 'resnet101_backbone',
           'resnet152_backbone', 'resnext50_32x4d_backbone', 'resnext101_32x8d_backbone',
           'wide_resnet50_2_backbone', 'wide_resnet101_2_backbone',
           'resnet18_dynamic', 'resnet50_dynamic', 'resnet101_dynamic',
           'resnet18_dynamic_backbone', 'resnet50_dynamic_backbone', 'resnet101_dynamic_backbone',
           'resnet18_ms_l123', 'resnet18_ms_l12', 'resnet18_ms_l1',
           'resnet50_ms_l123', 'resnet50_ms_l12', 'resnet50_ms_l1',
           'resnet101_ms_l123', 'resnet101_ms_l12', 'resnet101_ms_l1',
           'resnet18_ms_l123_backbone', 'resnet18_ms_l12_backbone', 'resnet18_ms_l1_backbone',
           'resnet50_ms_l123_backbone', 'resnet50_ms_l12_backbone', 'resnet50_ms_l1_backbone',
           'resnet101_ms_l123_backbone', 'resnet101_ms_l12_backbone', 'resnet101_ms_l1_backbone',
           'resnet18_dynamic_ms_l123', 'resnet18_dynamic_ms_l12', 'resnet18_dynamic_ms_l1',
           'resnet50_dynamic_ms_l123', 'resnet50_dynamic_ms_l12', 'resnet50_dynamic_ms_l1',
           'resnet101_dynamic_ms_l123', 'resnet101_dynamic_ms_l12', 'resnet101_dynamic_ms_l1',
           'resnet18_dynamic_ms_l123_backbone', 'resnet18_dynamic_ms_l12_backbone', 'resnet18_dynamic_ms_l1_backbone',
           'resnet50_dynamic_ms_l123_backbone', 'resnet50_dynamic_ms_l12_backbone', 'resnet50_dynamic_ms_l1_backbone',
           'resnet101_dynamic_ms_l123_backbone', 'resnet101_dynamic_ms_l12_backbone',
           'resnet101_dynamic_ms_l1_backbone']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'resnet18_dynamic': 'https://csip.fzu.edu.cn/files/models/resnet18_dynamic-074db766.pth',
    'resnet50_dynamic': 'https://csip.fzu.edu.cn/files/models/resnet50_dynamic-2c3b0201.pth',
    'resnet101_dynamic': 'https://csip.fzu.edu.cn/files/models/resnet101_dynamic-c5f15780.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv3x3_dynamic(in_planes: int, out_planes: int,
                    stride: int = 1,
                    attention_in_channels: int = None) -> Conv2dDynamic:
    """3x3 convolution with padding"""
    return Conv2dDynamic(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=1, bias=False, attention_in_channels=attention_in_channels)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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
        out = self.relu(out)

        return out


class BasicBlockDynamic(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlockDynamic, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_dynamic(inplanes, planes, stride, attention_in_channels=inplanes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_dynamic(planes, planes, attention_in_channels=inplanes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x, attention_x=x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, attention_x=x)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckDynamic(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BottleneckDynamic, self).__init__()
        if groups != 1:
            raise ValueError('BottleneckDynamic only supports groups=1')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BottleneckDynamic")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3_dynamic(width, width, stride, attention_in_channels=inplanes)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, attention_x=x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck, BasicBlockDynamic, BottleneckDynamic]],
            layers: List[int],
            has_fc: bool = True,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            ms_class=None,
            ms_layers=None,
            ms_p=0.5,
            ms_a=0.1
    ) -> None:
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
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.has_fc = has_fc
        self.out_features = 512 * block.expansion
        if has_fc:
            self.fc = nn.Linear(self.out_features, num_classes)
            self.out_features = num_classes

        if ms_class is not None and ms_layers is not None:
            self.ms_class = ms_class(p=ms_p, alpha=ms_a)
            for layer in ms_layers:
                assert layer in ['layer1', 'layer2', 'layer3']
            self.ms_layers = ms_layers
        else:
            self.ms_class = None
            self.ms_layers = []

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
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
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

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if 'layer1' in self.ms_layers:
            x = self.ms_class(x)
        x = self.layer2(x)
        if 'layer2' in self.ms_layers:
            x = self.ms_class(x)
        x = self.layer3(x)
        if 'layer3' in self.ms_layers:
            x = self.ms_class(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.has_fc:
            x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck, BasicBlockDynamic, BottleneckDynamic]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        # remove useless keys from sate_dict 1. no fc; 2. out_features != 1000.
        removed_keys = model.has_fc is False or (model.has_fc is True and model.out_features != 1000)
        removed_keys = ['fc.weight', 'fc.bias'] if removed_keys else []
        for key in removed_keys:
            state_dict.pop(key)
        # if has fc, then allow missing key, else strict load state_dict.
        allowed_missing_keys = removed_keys if model.has_fc else None
        load_state_dict(model, state_dict, allowed_missing_keys)
    return model


@MODELS_REGISTRY.register()
def resnet18(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet34(name, args, from_name=None) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet152(name, args, from_name=None) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnext50_32x4d(name, args, from_name=None) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes,
                    groups=32,
                    width_per_group=4)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnext101_32x8d(name, args, from_name=None) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes,
                    groups=32,
                    width_per_group=8)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def wide_resnet50_2(name, args, from_name=None) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes,
                    width_per_group=64 * 2)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def wide_resnet101_2(name, args, from_name=None) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes,
                    width_per_group=64 * 2)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    has_fc=False)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet34_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet152_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnext50_32x4d_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False,
                    groups=32,
                    width_per_group=4)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnext101_32x8d_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False,
                    groups=32,
                    width_per_group=8)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def wide_resnet50_2_backbone(name, args, from_name=None) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False,
                    width_per_group=64 * 2)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def wide_resnet101_2_backbone(name, args, from_name=None) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False,
                    width_per_group=64 * 2)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_dynamic(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18_dynamic', BasicBlockDynamic, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_dynamic(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50_dynamic', BottleneckDynamic, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_dynamic(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101_dynamic', BottleneckDynamic, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_dynamic_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18_dynamic', BasicBlockDynamic, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    has_fc=False)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_dynamic_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50_dynamic', BottleneckDynamic, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_dynamic_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101_dynamic', BottleneckDynamic, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False)
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_ms_l123(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1', 'layer2', 'layer3'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_ms_l12(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1', 'layer2'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_ms_l1(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_ms_l123(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1', 'layer2', 'layer3'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_ms_l12(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1', 'layer2'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_ms_l1(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_ms_l123(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1', 'layer2', 'layer3'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_ms_l12(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1', 'layer2'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_ms_l1(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_ms_l123_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1', 'layer2', 'layer3'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_ms_l12_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1', 'layer2'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_ms_l1_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_ms_l123_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1', 'layer2', 'layer3'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_ms_l12_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1', 'layer2'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_ms_l1_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_ms_l123_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1', 'layer2', 'layer3'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_ms_l12_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1', 'layer2'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_ms_l1_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_dynamic_ms_l123(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18_dynamic', BasicBlockDynamic, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1', 'layer2', 'layer3'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_dynamic_ms_l12(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18_dynamic', BasicBlockDynamic, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1', 'layer2'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_dynamic_ms_l1(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18_dynamic', BasicBlockDynamic, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_dynamic_ms_l123(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50_dynamic', BottleneckDynamic, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1', 'layer2', 'layer3'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_dynamic_ms_l12(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50_dynamic', BottleneckDynamic, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1', 'layer2'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_dynamic_ms_l1(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50_dynamic', BottleneckDynamic, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_dynamic_ms_l123(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101_dynamic', BottleneckDynamic, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1', 'layer2', 'layer3'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_dynamic_ms_l12(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101_dynamic', BottleneckDynamic, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1', 'layer2'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_dynamic_ms_l1(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101_dynamic', BottleneckDynamic, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    num_classes=args.num_classes, ms_class=MixStyle, ms_layers=['layer1'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_dynamic_ms_l123_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18_dynamic', BasicBlockDynamic, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1', 'layer2', 'layer3'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_dynamic_ms_l12_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18_dynamic', BasicBlockDynamic, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1', 'layer2'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet18_dynamic_ms_l1_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet18_dynamic', BasicBlockDynamic, [2, 2, 2, 2], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_dynamic_ms_l123_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50_dynamic', BottleneckDynamic, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1', 'layer2', 'layer3'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_dynamic_ms_l12_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50_dynamic', BottleneckDynamic, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1', 'layer2'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet50_dynamic_ms_l1_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet50_dynamic', BottleneckDynamic, [3, 4, 6, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_dynamic_ms_l123_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101_dynamic', BottleneckDynamic, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1', 'layer2', 'layer3'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_dynamic_ms_l12_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101_dynamic', BottleneckDynamic, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1', 'layer2'])
    args.__dict__[name]['out_features'] = model.out_features
    return model


@MODELS_REGISTRY.register()
def resnet101_dynamic_ms_l1_backbone(name, args, from_name=None) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        name: model name of trainer, used for parameter exchange.
        from_name: where to get in_features.
        args: Include the necessary parameters
    """
    model = _resnet('resnet101_dynamic', BottleneckDynamic, [3, 4, 23, 3], pretrained=args.pretrained, progress=True,
                    has_fc=False, ms_class=MixStyle, ms_layers=['layer1'])
    args.__dict__[name]['out_features'] = model.out_features
    return model
