import torch.nn as nn
from ..builder import BACKBONES
from .resnet import ResNet, Bottleneck
from mmcv.cnn import build_conv_layer, build_norm_layer



@BACKBONES.register_module()
class BHResNet(ResNet):

    """Bottom-Heavy ResNet backbone.

    short description of the backbone

    Args:
        depth(int): Network depth, from {50}.
        ...
    """

    arch_settings = {
        50: (Bottleneck, (7, 6, 2, 1)),
    }

    def __init__(self,
                 depth=50,
                 strides=(2, 2, 2, 2),
                 deep_stem=False,
                 **kwargs):
        super().__init__(depth=depth, strides=strides, deep_stem=deep_stem, **kwargs)

    def _make_stem_layer(self, in_channels, stem_channels):
        # override ResNet method to modify the network structure
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

