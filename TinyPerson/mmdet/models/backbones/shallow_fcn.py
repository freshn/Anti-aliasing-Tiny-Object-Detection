import torch.nn as nn

from ..builder import BACKBONES
from .resnet import ResNet, BasicBlock, Bottleneck
from mmcv.cnn import build_conv_layer, build_norm_layer


@BACKBONES.register_module()
class ShallowNet(ResNet):

    """ResNet backbone for CIFAR.

    short description of the backbone

    Args:
        depth(int): Network depth, from {10, 18, 26}.
        ...
    """

    arch_settings = {
        10: (BasicBlock, (3, 1, 1, 1)),
        18: (BasicBlock, (3, 2, 2, 1)),
        26: (Bottleneck, (8, 4, 4, 1)),
        50: (Bottleneck, (7, 6, 2, 1))
    }

    def __init__(self, depth, deep_stem=False, **kwargs):
        # call ResNet init
        super().__init__(depth, deep_stem=deep_stem, **kwargs)
        # other specific initialization
        assert not self.deep_stem, 'ResNet_CIFAR do not support deep_stem'

        strides = kwargs.get('strides')
        dilations = kwargs.get('dilations')
        self.res_layers = []
        _in_channels = self.stem_channels
        _out_channels = self.base_channels
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = self.make_res_layer(
                block=self.block,
                num_blocks=num_blocks,
                inplanes=_in_channels,
                planes=_out_channels,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
            _in_channels = _out_channels * self.block.expansion
            # _in_channels = _out_channels
            _out_channels *= 2
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def _make_stem_layer(self, in_channels, base_channels):
        # override ResNet method to modify the network structure
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, base_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):  # should return a tuple
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
