import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from mmcv.runner import BaseModule
from .resnet import ResNet, BasicBlock, Bottleneck
from mmcv.cnn import build_conv_layer, build_norm_layer
from torch.nn.modules.batchnorm import _BatchNorm

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

class PASALayer(nn.Module):
    
    def __init__(self, stride, in_channels, kernel_size=3, group=8):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        pad_size = self.kernel_size//2
        sigma = self.conv(F.pad(x,(pad_size,pad_size,pad_size,pad_size)))
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)

        n,c,h,w = sigma.shape
        sigma = sigma.reshape(n,1,c,h*w)
        n,c2,p,q = sigma.shape
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)  # (n,g*s*s//s*s,1,s*s,h*w)=(n,g,1,s*s,h*w)

        n,c,h,w = x.shape
        x = F.unfold(F.pad(x,(pad_size,pad_size,pad_size,pad_size)), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))
        n,c1,p,q = x.shape
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)

        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
        return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]
    
class PASABasicBlock(BasicBlock):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 **kwargs):
        super(PASABasicBlock, self).__init__(inplanes=in_channels, 
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
            self.conv2 = nn.Sequential(PASALayer(stride=stride, in_channels=out_channels),
                                       build_conv_layer(
                                            conv_cfg,
                                            out_channels,
                                            out_channels,
                                            3,
                                            stride=1,
                                            padding=dilation,
                                            dilation=dilation,
                                            bias=False))

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

class PASABottleneck(Bottleneck):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 conv_cfg=None,
                 **kwargs):
        super(PASABottleneck, self).__init__(inplanes=in_channels, 
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
                            kernel_size=1,
                            bias=False)
        else:
            self.conv3 = nn.Sequential(PASALayer(stride=self.conv2_stride, in_channels=self.planes),
                                                build_conv_layer(
                                                    conv_cfg,
                                                    self.planes,
                                                    self.planes * self.expansion,
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

class PASAResLayer(nn.Sequential):

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
                 **kwargs):
        self.block = block
        self.expansion = get_expansion(block, expansion)

        downsample = None
        in_channels = inplanes
        out_channels = planes
        if stride != 1 or in_channels != out_channels * self.expansion:
            downsample = [PASALayer(stride=stride, in_channels=in_channels),] if(stride !=1) else []
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels * self.expansion)[1]
            ])
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
                    **kwargs))
        super(PASAResLayer, self).__init__(*layers)

@BACKBONES.register_module()
class PASAResNet(ResNet):

    arch_settings = {
        18: (PASABasicBlock, (2, 2, 2, 2)),
        34: (PASABasicBlock, (3, 4, 6, 3)),
        32: (PASABottleneck, (3, 2, 1, 1)),
        50: (PASABottleneck, (3, 4, 6, 3)),
        101: (PASABottleneck, (3, 4, 23, 3)),
        152: (PASABottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 out_indices=(3, ),
                 style='pytorch',
                 **kwargs):
        super(PASAResNet, self).__init__(depth=depth,
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
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, stem_channels, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Sequential(*[PASALayer(stride=2, in_channels=stem_channels),
                                       nn.MaxPool2d(kernel_size=3, stride=1, padding=1)])
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def make_res_layer(self, **kwargs):
        return PASAResLayer(**kwargs)

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
        super(PASAResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
