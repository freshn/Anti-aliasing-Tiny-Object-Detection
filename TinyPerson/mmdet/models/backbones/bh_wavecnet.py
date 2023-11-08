import torch
import torch.nn as nn

from ..builder import BACKBONES
from .wavecnet_resnet import WaveCResNet, WaveCBottleneck, WaveCLayer
from mmcv.cnn import build_conv_layer, build_norm_layer

@BACKBONES.register_module()
class BHWaveCResNet(WaveCResNet):
    arch_settings = {50: (WaveCBottleneck, (7, 6, 2, 1))}

    def __init__(self,
                 depth=50,
                 strides=(2, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 wavename='haar',
                 **kwargs):
        self.wavename = wavename
        super(WaveCResNet, self).__init__(depth=depth,
                                         strides=strides,
                                         dilations=dilations,
                                         **kwargs)

    def _make_stem_layer(self, in_channels, stem_channels):
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
        self.maxpool = nn.Sequential(WaveCLayer(wavename=self.wavename))

    def train(self, mode=True):
        super(BHWaveCResNet, self).train(mode)
