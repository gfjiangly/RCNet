from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import logging
import numpy as np
from os.path import join

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import matplotlib.pyplot as plt

from .DCNv2.dcn_v2 import DCN
# from .dcn.modules.deform_conv import ModulatedDeformConvPack as DCN

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(bottle_planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        # self.fc = fc


def dla34(pretrained=True, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):

    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x


class DLASeg(nn.Module):
    def __init__(self, base_name, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0, fpn=False):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.fpn = fpn
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = globals()[base_name](pretrained=pretrained)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        if self.fpn:
            self.fpn_in_channels = [256, 128]
            self.fpn_out_channels = [64, 64]
            self.heads = [heads for _ in range(3)]
            for i in range(3):
                for head in self.heads[i]:
                    classes = self.heads[i][head]
                    if head_conv > 0:
                        fc = nn.Sequential(
                            nn.Conv2d(channels[self.first_level], head_conv,
                                      kernel_size=3, padding=1, bias=True),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(head_conv, classes,
                                      kernel_size=final_kernel, stride=1,
                                      padding=final_kernel // 2, bias=True))
                        if 'hm' in head:
                            fc[-1].bias.data.fill_(-2.19)
                        else:
                            fill_fc_weights(fc)
                    else:
                        fc = nn.Conv2d(channels[self.first_level], classes,
                                       kernel_size=final_kernel, stride=1,
                                       padding=final_kernel // 2, bias=True)
                        if 'hm' in head:
                            fc.bias.data.fill_(-2.19)
                        else:
                            fill_fc_weights(fc)
                    self.__setattr__(str(i)+head, fc)
        else:
            self.heads = heads
            for head in self.heads:
                classes = self.heads[head]
                if head_conv > 0:
                    fc = nn.Sequential(
                        nn.Conv2d(channels[self.first_level], head_conv,
                                  kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, classes,
                                  kernel_size=final_kernel, stride=1,
                                  padding=final_kernel // 2, bias=True))
                    if 'hm' in head:
                        fc[-1].bias.data.fill_(-2.19)
                    else:
                        fill_fc_weights(fc)
                else:
                    fc = nn.Conv2d(channels[self.first_level], classes,
                                   kernel_size=final_kernel, stride=1,
                                   padding=final_kernel // 2, bias=True)
                    if 'hm' in head:
                        fc.bias.data.fill_(-2.19)
                    else:
                        fill_fc_weights(fc)
                self.__setattr__(head, fc)

        if self.fpn:
            self.fpn_convs = nn.ModuleList()
            for i in range(2):
                fpn_conv = DeformConv(self.fpn_in_channels[i], self.fpn_out_channels[i])
                # fpn_conv = nn.Sequential(
                #     nn.Conv2d(self.fpn_in_channels[i], self.fpn_out_channels[i],
                #               kernel_size=3, padding=1, bias=True),
                #     nn.ReLU(inplace=True)
                # )
                fill_fc_weights(fpn_conv)
                self.fpn_convs.append(fpn_conv)

    def forward(self, x):
        x = self.base(x)
        
        # for i, xi in enumerate(x):
        #     self.visualize_feature_map(xi, f'base{i}', merge='sum')

        x = self.dla_up(x)

        # for i, xi in enumerate(x):
        #     self.visualize_feature_map(xi, f'dla_up{i}', merge='sum')

        # if self.fpn:
        #     y_add = [x[2].clone(), x[1].clone()]

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        rets = []

        # if self.fpn:
        #     for i in range(len(y_add)):
        #         x1 = self.fpn_convs[i](y_add[i])
        #         ret = {}
        #         for head in self.heads[i]:
        #             ret[head] = self.__getattr__(str(i)+head)(x1)
        #         rets.append(ret)
        #     ret = {}
        #     for head in self.heads[2]:
        #         ret[head] = self.__getattr__(str(2)+head)(y[-1])
        #     rets.append(ret)
        # else:
        #     ret = {}
        #     for head in self.heads:
        #         ret[head] = self.__getattr__(head)(y[-1])
        #     rets.append(ret)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(y[-1])
            # self.add_heatmap(ret[head], f'heatmap_{head}')
            # self.visualize_feature_map(ret[head], f'feature_{head}')
        rets.append(ret)
        return [rets]
    
    def add_heatmap(self, feature_maps, name):
        """
        :param feature_maps:[B, C, H, W]
        :return:
        """
        def figure_attention(activation):
            fig, ax = plt.subplots()
            im = ax.imshow(activation, cmap='jet')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im)
            return fig

        # heatmap = torch.sum(feature_maps, dim=1)
        for i in range(feature_maps.shape[1]):
            fig = figure_attention(feature_maps.detach().cpu().numpy()[0, i, :])
            fig.savefig(f'/code/CenterNet/exp/ctdet/vis/{name}_{i}.jpg', 
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True,
                    dpi=300)
    
    def visualize_feature_map(self, feature_maps, name, merge='single'):
        if merge == 'sum':
            feature_maps = torch.sum(feature_maps, dim=1, keepdim=True)
            # pass
        elif merge == 'mean':
            feature_maps = torch.mean(feature_maps, dim=1, keepdim=True)
        # feature_max = torch.max(feature_maps, dim=1)
        # feature_maps = feature_maps / feature_max
        
        plt.figure()
        for i in range(feature_maps.shape[0]):
            for j in range(feature_maps.shape[1]):
                feature = feature_maps.detach().cpu().numpy()[i, j, :]
                # feature = feature  / feature.std()
                plt.xticks([])
                plt.yticks([])
                plt.imshow(feature)
                plt.savefig(f'/code/CenterNet/exp/ctdet/vis/{name}_{merge}_{i}_{j}.jpg', 
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True,
                    dpi=300)


def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4, fpn=False):
    model = DLASeg('dla{}'.format(num_layers), heads,
                   pretrained=True,
                   down_ratio=down_ratio,
                   final_kernel=1,
                   last_level=5,
                   head_conv=head_conv,
                   fpn=fpn)
    return model
