import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from mmcv.runner import BaseModule
from .resnet import ResNet, BasicBlock, Bottleneck
from mmcv.cnn import build_conv_layer, build_norm_layer
from torch.nn.modules.batchnorm import _BatchNorm

# for WaveCNets
import pywt
import math
import numpy as np
from torch.autograd import Function

def get_expansion(block, expansion=None):
    
    if isinstance(expansion, int):
        assert expansion > 0
    elif expansion is None:
        if hasattr(block, 'expansion'):
            expansion = block.expansion
        elif issubclass(block, BasicBlock):
            expansion = 1
        elif issubclass(block, Bottleneck):
            expansion = 4
        else:
            raise TypeError(f'expansion is not specified for {block.__name__}')
    else:
        raise TypeError('expansion must be an integer or None')

    return expansion

class DWTFunction_2D_tiny(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        return LL
    @staticmethod
    def backward(ctx, grad_LL):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.matmul(grad_LL, matrix_Low_1.t())
        grad_input = torch.matmul(matrix_Low_0.t(), grad_L)
        return grad_input, None, None, None, None
    
class DWT_2D_tiny(nn.Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc: (N, C, H/2, W/2)
              #hfc_lh: (N, C, H/2, W/2)
              #hfc_hl: (N, C, H/2, W/2)
              #hfc_hh: (N, C, H/2, W/2)
    DWT_2D_tiny only outputs the low-frequency component, which is used in WaveCNet;
    the all four components could be get using DWT_2D, which is used in WaveUNet.
    """
    def __init__(self, wavename='haar'):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D_tiny, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \mathcal{L} * input * \mathcal{L}^T
        #input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
        #input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
        #input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency component of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D_tiny.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class WaveCLayer(nn.Module):
    
    def __init__(self, wavename='haar'):
        super(WaveCLayer, self).__init__()
        self.dwt = DWT_2D_tiny(wavename = wavename)

    def forward(self, input):
        LL = self.dwt(input)
        return LL
    
class WaveCBasicBlock(BasicBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 wavename='haar',
                 **kwargs):
        super(WaveCBasicBlock, self).__init__(inplanes=in_channels, 
                                             planes=out_channels, 
                                             stride=stride,
                                             **kwargs)

        if stride==1:
            self.conv2 = build_conv_layer(
                            conv_cfg,
                            out_channels,
                            out_channels,
                            3,
                            stride=stride,
                            padding=dilation,
                            dilation=dilation,
                            bias=False)
        else:
            self.conv2 = nn.Sequential(build_conv_layer(
                                            conv_cfg,
                                            out_channels,
                                            out_channels,
                                            3,
                                            stride=1,
                                            padding=dilation,
                                            dilation=dilation,
                                            bias=False),
                                       WaveCLayer(wavename=wavename))

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class WaveCBottleneck(Bottleneck):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 conv_cfg=None,
                 wavename='haar',
                 **kwargs):
        super(WaveCBottleneck, self).__init__(inplanes=in_channels, 
                                             planes=out_channels, 
                                             stride=stride,
                                             conv_cfg=conv_cfg,
                                             **kwargs)

        self.conv2 = build_conv_layer(
                        conv_cfg,
                        self.planes,
                        self.planes,
                        kernel_size=3,
                        stride=1,
                        padding=self.dilation,
                        dilation=self.dilation,
                        bias=False)
        if self.conv2_stride == 1:
            self.conv3 = build_conv_layer(
                            conv_cfg,
                            self.planes,
                            self.planes * self.expansion,
                            stride=1,
                            kernel_size=1,
                            bias=False)
        else:
            self.conv3 = nn.Sequential(WaveCLayer(wavename=wavename),
                                       build_conv_layer(
                                            conv_cfg,
                                            self.planes,
                                            self.planes * self.expansion,
                                            stride=1,
                                            kernel_size=1,
                                            bias=False))

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class WaveCResLayer(nn.Sequential):

    def __init__(self,
                 block,
                 num_blocks,
                 inplanes,
                 planes,
                 expansion=None,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 wavename='haar',
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)

        downsample = None
        in_channels = inplanes
        out_channels = planes
        if stride != 1 or in_channels != out_channels * self.expansion:
            downsample = [build_conv_layer(
                            conv_cfg,
                            in_channels,
                            out_channels * self.expansion,
                            kernel_size=1,
                            stride=1,
                            bias=False)]
            if stride != 1:
                downsample += [WaveCLayer(wavename=wavename)]
            downsample += [build_norm_layer(norm_cfg, out_channels*self.expansion)[1]]
            # downsample = [WaveCLayer()] if stride!=1 else []
            # downsample += [build_conv_layer(
            #                     conv_cfg,
            #                     in_channels,
            #                     out_channels * self.expansion,
            #                     kernel_size=1,
            #                     stride=1,
            #                     bias=False),
            #                build_norm_layer(norm_cfg, out_channels * self.expansion)[1]]
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                wavename=wavename,
                **kwargs))
        in_channels = out_channels * self.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    wavename=wavename,
                    **kwargs))
        super(WaveCResLayer, self).__init__(*layers)

@BACKBONES.register_module()
class WaveCResNet(ResNet):

    arch_settings = {
        18: (WaveCBasicBlock, (2, 2, 2, 2)),
        34: (WaveCBasicBlock, (3, 4, 6, 3)),
        50: (WaveCBottleneck, (3, 4, 6, 3)),
        101: (WaveCBottleneck, (3, 4, 23, 3)),
        152: (WaveCBottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 out_indices=(3, ),
                 style='pytorch',
                 wavename='haar',
                 **kwargs):
        self.wavename = wavename
        super(WaveCResNet, self).__init__(depth=depth,
                                         in_channels=in_channels,
                                         num_stages=num_stages,
                                         strides=strides,
                                         out_indices=out_indices,
                                         style='pytorch',
                                         **kwargs)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Sequential(WaveCLayer(wavename=self.wavename), WaveCLayer(wavename=self.wavename))

    def make_res_layer(self, **kwargs):
        return WaveCResLayer(wavename=self.wavename, **kwargs)

    def forward(self, x):
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

    def train(self, mode=True):
        super(WaveCResNet, self).train(mode)
