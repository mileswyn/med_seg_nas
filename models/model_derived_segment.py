import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .operations import OPS
from tools.utils import parse_net_config


class Block(nn.Module):

    def __init__(self, in_ch, block_ch, head_op, stack_ops, stride):
        super(Block, self).__init__()
        self.head_layer = OPS[head_op](in_ch, block_ch, stride,
                                       affine=True, track_running_stats=True)

        modules = []
        for stack_op in stack_ops:
            modules.append(OPS[stack_op](block_ch, block_ch, 1,
                                         affine=True, track_running_stats=True))
        self.stack_layers = nn.Sequential(*modules)

    def forward(self, x):
        x = self.head_layer(x)
        x = self.stack_layers(x)
        return x


# class Conv1_1_Block(nn.Module):

#     def __init__(self, in_ch, block_ch):
#         super(Conv1_1_Block, self).__init__()
#         self.conv1_1 = nn.Sequential(
#             nn.Conv2d(in_channels=in_ch, out_channels=block_ch,
#                       kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(block_ch),
#             nn.ReLU6(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv1_1(x)
class Conv1_1_Block(nn.Module):

    def __init__(self, in_ch, block_ch):
        super(Conv1_1_Block, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(in_channels=in_ch, out_channels=block_ch,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(block_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=block_ch, out_channels=block_ch,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(block_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv1_1(x)

# class conv_block(nn.Module):
#     """
#     Convolution Block
#     """

#     def __init__(self, in_ch, out_ch):
#         super(conv_block, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True))

#     def forward(self, x):
#         x = self.conv(x)
#         return x
class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

# class up_conv(nn.Module):
#     """
#     Up Convolution Block
#     """

#     def __init__(self, in_ch, out_ch):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.up(x)
#         return x
class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, kernel_size=[2,2,2], stride=[2,2,2], bias=True),
        )

    def forward(self, x):
        x = self.up(x)
        return x

class MBV2_Net(nn.Module):
    def __init__(self, net_config, config=None):
        """
        net_config=[[in_ch, out_ch], head_op, [stack_ops], num_stack_layers, stride]
        """
        super(MBV2_Net, self).__init__()
        self.config = config
        self.net_config = parse_net_config(net_config)
        self.in_chs = self.net_config[0][0][0]
        self._num_classes = 2
        self.out_ch = 2
        n1 = 32
        # filters = [16, 24, 48, 128, 384]
        filters = [16, 32, 64, 128, 256]

        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.in_chs, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.in_chs),
            nn.ReLU6(inplace=True)
        )
        self.blocks = nn.ModuleList()
        for config in self.net_config:
            if config[1] == 'conv1_1':
                continue
            self.blocks.append(Block(config[0][0], config[0][1],
                                     config[1], config[2], config[-1]))

        if self.net_config[-1][1] == 'conv1_1':
            block_last_dim = self.net_config[-1][0][0]
            last_dim = self.net_config[-1][0][1]
        else:
            block_last_dim = self.net_config[-1][0][1]
        self.conv1_1_block = Conv1_1_Block(block_last_dim, last_dim)
        self.Up6 = up_conv(last_dim, filters[4])
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up4 = up_conv(filters[3], filters[2])
        self.Up3 = up_conv(filters[2], filters[0])
        # self.Up2 = up_conv(filters[1], filters[0])
        # self.Up1 = up_conv(filters[0], 16)

        self.Conv = nn.Conv2d(filters[0], self.out_ch, kernel_size=1, stride=1, padding=0)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_dim, self._num_classes)

        self.init_model()
        self.set_bn_param(0.1, 0.001)

    def forward(self, x):
        block_data = self.input_block(x)
        for i, block in enumerate(self.blocks):
            block_data = block(block_data)
        block_data = self.conv1_1_block(block_data)
        block_data = self.Up6(block_data)
        block_data = self.Up5(block_data)
        block_data = self.Up4(block_data)
        block_data = self.Up3(block_data)
        # block_data = self.Up2(block_data)
        # block_data = self.Up1(block_data)

        # block_data = self.Up_conv2(block_data)
        out = self.Conv(block_data)

        #
        # out = self.global_pooling(block_data)
        # logits = self.classifier(out.view(out.size(0), -1))

        return out

    def init_model(self, model_init='he_fout', init_div_groups=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    if init_div_groups:
                        n /= m.groups
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else:
                    raise NotImplementedError
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def set_bn_param(self, bn_momentum, bn_eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
                m.eps = bn_eps
        return

# # promise12 experiment
# class RES_Net(nn.Module):
#     def __init__(self, net_config, config=None):
#         """
#         net_config=[[in_ch, out_ch], head_op, [stack_ops], num_stack_layers, stride]
#         """
#         super(RES_Net, self).__init__()
#         self.config = config
#         self.net_config = parse_net_config(net_config)
#         self.in_chs2 = self.net_config[0][0][0]
#         self.in_chs = 64
#         self._num_classes = 2
#         self.out_ch = 2
#         n1 = 32
#         # filters = [16, 24, 48, 128, 384]
#         filters = [32, 64, 128, 256, 512, 1024]
#
#
#         self.input_block = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=self.in_chs, kernel_size=3,
#                       stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(self.in_chs),
#             nn.ReLU6(inplace=True),
#         )
#         self.input_block2 = nn.Sequential(
#             nn.Conv2d(in_channels=self.in_chs, out_channels=self.in_chs2, kernel_size=3,
#                       stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(self.in_chs2),
#             nn.ReLU6(inplace=True),
#         )
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.blocks = nn.ModuleList()
#         for config in self.net_config:
#             self.blocks.append(Block(config[0][0], config[0][1],
#                                      config[1], config[2], config[-1]))
#
#         # self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
#         if self.net_config[-1][1] == 'bottle_neck':
#             last_dim = self.net_config[-1][0][-1] * 4
#         else:
#             last_dim = self.net_config[-1][0][1]
#         # self.classifier = nn.Linear(last_dim, self._num_classes)
#         self.up7_ch = self.net_config[2][0][1]  # ch=16
#         self.up6_ch = self.net_config[1][0][1]  # ch=32
#         self.up5_ch = self.net_config[0][0][1]  # ch=64
#         self.up4_ch = self.in_chs2
#         self.up3_ch = self.in_chs
#         self.conv1_1_block = Conv1_1_Block(last_dim, last_dim)
#         # self.Up7 = up_conv(last_dim, filters[5])
#         # self.upconv7 = conv_block(self.up7_ch+filters[5], filters[5])
#         self.Up6 = up_conv(filters[5], filters[4])
#         self.upconv6 = conv_block(self.up6_ch+filters[4], filters[4])
#         self.Up5 = up_conv(filters[4], filters[3])
#         self.upconv5 = conv_block(self.up5_ch+filters[3], filters[3])
#         self.Up4 = up_conv(filters[3], filters[2])
#         self.upconv4 = conv_block(self.up4_ch+filters[2], filters[2])
#         self.Up3 = up_conv(filters[2], filters[1])
#         self.upconv3 = conv_block(self.up3_ch + filters[1], filters[1])
#         # self.Up7 = up_conv(last_dim, filters[5])
#         # self.upconv7 = conv_block(self.up7_ch + filters[5], filters[5])
#
#         # self.Up6 = up_conv(filters[4], filters[3])
#         # self.upconv6 = conv_block(self.up6_ch + filters[3], filters[3])
#         # self.Up5 = up_conv(filters[3], filters[2])
#         # self.upconv5 = conv_block(self.up5_ch + filters[2], filters[2])
#         # self.Up4 = up_conv(filters[2], filters[1])
#         # self.upconv4 = conv_block(self.up4_ch + filters[1], filters[1])
#         # self.Up3 = up_conv(filters[1], filters[0])
#         # self.upconv3 = conv_block(self.up3_ch + filters[0], filters[0])
#
#         # self.Up2 = up_conv(filters[1], filters[0])
#         # self.Up1 = up_conv(filters[0], 16)
#
#         # self.Conv = nn.Conv2d(filters[0], self.out_ch, kernel_size=1, stride=1, padding=0)
#         self.Conv = nn.Conv2d(filters[1], self.out_ch, kernel_size=1, stride=1, padding=0)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 if m.affine == True:
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
#
#
#     def forward(self, x):
#         block_data = self.input_block(x)
#         block_data0 = block_data
#         block_data = self.maxpool1(block_data)
#         block_data = self.input_block2(block_data)
#         block_data1 = block_data
#         self.downsam = []
#         for i, block in enumerate(self.blocks):
#             block_data = block(block_data)
#             self.downsam.append(block_data)
#         block_data = self.conv1_1_block(block_data)
#         # block_data = self.Up7(block_data)
#         # block_data = torch.cat((self.downsam[4],self.downsam[3],self.downsam[2], block_data), dim=1)
#         # block_data = self.upconv7(block_data)
#         block_data = self.Up6(block_data)
#         block_data = torch.cat((self.downsam[1], block_data), dim=1)
#         block_data = self.upconv6(block_data)
#         block_data = self.Up5(block_data)
#         block_data = torch.cat((self.downsam[0], block_data), dim=1)
#         block_data = self.upconv5(block_data)
#         block_data = self.Up4(block_data)
#         block_data = torch.cat((block_data1, block_data), dim=1)
#         block_data = self.upconv4(block_data)
#         block_data = self.Up3(block_data)
#         block_data = torch.cat((block_data0, block_data), dim=1)
#         block_data = self.upconv3(block_data)
#         # block_data = self.Up2(block_data)
#         # block_data = self.Up1(block_data)
#
#         # block_data = self.Up_conv2(block_data)
#         out = self.Conv(block_data)
#         # out = self.global_pooling(block_data)
#         # out = torch.flatten(out, 1)
#         # logits = self.classifier(out)
#         return out
# # promise12 experiment

# sliver07 experiment
def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        nn.BatchNorm2d(out_channels, affine=True),
        nn.ReLU(True))
    return block


class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h,w), mode='bilinear', align_corners=True)

class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP_Module, self).__init__()
        # In our re-implementation of ASPP module,
        # we follow the original paper but change the output channel
        # from 256 to 512 in all of four branches.
        out_channels = in_channels // 4

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels, affine=True),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        #self.b4 = AsppPooling(in_channels, out_channels)

    def forward(self, x):
        feat0 = self.b0(x)  # conv1X1
        feat1 = self.b1(x)  # astrous conv3X3
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        #feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3), 1)
        return y

class RES_Net(nn.Module):
    def __init__(self, net_config, config=None):
        """
        net_config=[[in_ch, out_ch], head_op, [stack_ops], num_stack_layers, stride]
        """
        super(RES_Net, self).__init__()
        self.config = config
        self.net_config = parse_net_config(net_config)
        self.in_chs2 = self.net_config[0][0][0]
        self.in_chs = 32
        # self._num_classes = 2
        self.out_ch = 9 # sliver07 use only one channel out
        n1 = 32
        # filters = [16, 24, 48, 128, 384]
        filters = [32, 64, 128, 256, 320]


        self.input_block = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=self.in_chs, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(self.in_chs),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels=self.in_chs, out_channels=self.in_chs, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(self.in_chs),
            nn.LeakyReLU(inplace=True),
        )
        # self.input_block2 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.in_chs, out_channels=self.in_chs2, kernel_size=3,
        #               stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(self.in_chs2),
        #     nn.ReLU6(inplace=True),
        # )
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.blocks = nn.ModuleList()
        for config in self.net_config:
            self.blocks.append(Block(config[0][0], config[0][1],
                                     config[1], config[2], config[-1]))

        # self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        if self.net_config[-1][1] == 'bottle_neck':
            last_dim = self.net_config[-1][0][-1] * 4
        else:
            last_dim = self.net_config[-1][0][1]
        # self.classifier = nn.Linear(last_dim, self._num_classes)
        self.up6_ch = self.net_config[3][0][1]  # ch=320
        self.up5_ch = self.net_config[2][0][1]  # ch=256
        self.up4_ch = self.net_config[1][0][1]  # ch=128
        self.up3_ch = self.net_config[0][0][1]
        self.up2_ch = self.net_config[0][0][0]
        self.conv1_1_block = Conv1_1_Block(last_dim, last_dim)
        # self.Up7 = up_conv(last_dim, filters[5])
        # self.upconv7 = conv_block(self.up7_ch+filters[5], filters[5])
        self.Up6 = up_conv(filters[4], filters[4])
        self.upconv6 = conv_block(self.up6_ch+filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[3])
        self.upconv5 = conv_block(self.up5_ch+filters[3], filters[3])
        self.Up4 = up_conv(filters[3], filters[2])
        self.upconv4 = conv_block(self.up4_ch+filters[2], filters[2])
        self.Up3 = up_conv(filters[2], filters[1])
        self.upconv3 = conv_block(self.up3_ch + filters[1], filters[1])
        self.Up2 = up_conv(filters[1], filters[0])
        self.upconv2 = conv_block(self.up2_ch + filters[0], filters[0])
        # self.Up6_Conv = nn.Sequential(
        #         self.conv1x1(filters[4], self.out_ch),
        #         nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True))
        # self.Up5_Conv = nn.Sequential(
        #         self.conv1x1(filters[3], self.out_ch),
        #         nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True))
        # self.Up4_Conv = nn.Sequential(
        #         self.conv1x1(filters[2], self.out_ch),
        #         nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
        # self.Up6_Conv_vice = nn.Sequential(
        #     self.conv1x1(filters[4], self.out_ch+1),
        #     nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True))
        # self.Up5_Conv_vice = nn.Sequential(
        #     self.conv1x1(filters[3], self.out_ch+1),
        #     nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True))
        # self.Up4_Conv_vice = nn.Sequential(
        #     self.conv1x1(filters[2], self.out_ch+1),
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
        # self.aspp = ASPP_Module(in_channels=filters[5], atrous_rates=(2,4,8))
        # self.Up7 = up_conv(last_dim, filters[5])
        # self.upconv7 = conv_block(self.up7_ch + filters[5], filters[5])

        # self.Up6 = up_conv(filters[4], filters[3])
        # self.upconv6 = conv_block(self.up6_ch + filters[3], filters[3])
        # self.Up5 = up_conv(filters[3], filters[2])
        # self.upconv5 = conv_block(self.up5_ch + filters[2], filters[2])
        # self.Up4 = up_conv(filters[2], filters[1])
        # self.upconv4 = conv_block(self.up4_ch + filters[1], filters[1])
        # self.Up3 = up_conv(filters[1], filters[0])
        # self.upconv3 = conv_block(self.up3_ch + filters[0], filters[0])

        # self.Up2 = up_conv(filters[1], filters[0])
        # self.Up1 = up_conv(filters[0], 16)

        # self.Conv = nn.Conv2d(filters[0], self.out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv = nn.Conv3d(filters[0], self.out_ch, kernel_size=1, stride=1, padding=0)
        # self.Conv_vice = nn.Conv2d(filters[1], self.out_ch+1, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.affine == True:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def conv1x1(self, in_planes, out_planes, stride=1, bias=True):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

    def forward(self, x):
        block_data = self.input_block(x)
        block_data0 = block_data
        # block_data = self.maxpool1(block_data)
        # block_data = self.input_block2(block_data)
        # block_data1 = block_data
        self.downsam = []
        for i, block in enumerate(self.blocks):
            block_data = block(block_data)
            self.downsam.append(block_data)
        block_data = self.conv1_1_block(block_data)
        # block_data = self.aspp(block_data)
        # block_data = self.Up7(block_data)
        # block_data = torch.cat((self.downsam[4],self.downsam[3],self.downsam[2], block_data), dim=1)
        # block_data = self.upconv7(block_data)
        block_data = self.Up6(block_data) # 32
        block_data = torch.cat((self.downsam[3], block_data), dim=1)
        block_data = self.upconv6(block_data)
        up6 = block_data
        block_data = self.Up5(block_data)
        block_data = torch.cat((self.downsam[2], block_data), dim=1)
        block_data = self.upconv5(block_data)
        up5 = block_data
        block_data = self.Up4(block_data)
        block_data = torch.cat((self.downsam[1], block_data), dim=1)
        block_data = self.upconv4(block_data)
        up4 = block_data
        block_data = self.Up3(block_data)
        block_data = torch.cat((self.downsam[0], block_data), dim=1)
        block_data = self.upconv3(block_data)
        block_data = self.Up2(block_data)
        block_data = torch.cat((block_data0, block_data), dim=1)
        block_data = self.upconv2(block_data)
        # block_data = self.Up2(block_data)
        # block_data = self.Up1(block_data)
        # block_data = self.aspp(block_data)
        # block_data = self.Up_conv2(block_data)
        out = self.Conv(block_data)
        # out_vice = self.Conv_vice(block_data)
        # deeps = []
        # deeps_vice = []
        # for seg,deep in zip(
        #         [up6, up5, up4],
        #         [self.Up6_Conv,self.Up5_Conv,self.Up4_Conv]):
        #     deeps.append(deep(seg))
        # for seg, deep in zip(
        #         [up6, up5, up4],
        #         [self.Up6_Conv_vice, self.Up5_Conv_vice, self.Up4_Conv_vice]):
        #     deeps_vice.append(deep(seg))
        # out = self.global_pooling(block_data)
        # out = torch.flatten(out, 1)
        # logits = self.classifier(out)
        return out #, out_vice, deeps, deeps_vice

class RES_Net_summary(nn.Module):
    def __init__(self, net_config, config=None):
        """
        net_config=[[in_ch, out_ch], head_op, [stack_ops], num_stack_layers, stride]
        """
        super(RES_Net_summary, self).__init__()
        self.config = config
        self.net_config = parse_net_config(net_config)
        self.in_chs2 = self.net_config[0][0][0]
        self.in_chs = 64
        # self._num_classes = 2
        self.out_ch = 1 # sliver07 use only one channel out
        n1 = 32
        # filters = [16, 24, 48, 128, 384]
        filters = [32, 64, 128, 256, 512, 1024]


        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.in_chs, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_chs),
            nn.ReLU6(inplace=True),
        )
        self.input_block2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_chs, out_channels=self.in_chs2, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_chs2),
            nn.ReLU6(inplace=True),
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.blocks = nn.ModuleList()
        for config in self.net_config:
            self.blocks.append(Block(config[0][0], config[0][1],
                                     config[1], config[2], config[-1]))

        # self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        if self.net_config[-1][1] == 'bottle_neck':
            last_dim = self.net_config[-1][0][-1] * 4
        else:
            last_dim = self.net_config[-1][0][1]
        # self.classifier = nn.Linear(last_dim, self._num_classes)
        self.up7_ch = self.net_config[2][0][1]  # ch=16
        self.up6_ch = self.net_config[1][0][1]  # ch=32
        self.up5_ch = self.net_config[0][0][1]  # ch=64
        self.up4_ch = self.in_chs2
        self.up3_ch = self.in_chs
        self.conv1_1_block = Conv1_1_Block(last_dim, last_dim)
        # self.Up7 = up_conv(last_dim, filters[5])
        # self.upconv7 = conv_block(self.up7_ch+filters[5], filters[5])
        self.Up6 = up_conv(filters[5], filters[4])
        self.upconv6 = conv_block(self.up6_ch+filters[4], filters[4])
        self.Up5 = up_conv(filters[4], filters[3])
        self.upconv5 = conv_block(self.up5_ch+filters[3], filters[3])
        self.Up4 = up_conv(filters[3], filters[2])
        self.upconv4 = conv_block(self.up4_ch+filters[2], filters[2])
        self.Up3 = up_conv(filters[2], filters[1])
        self.upconv3 = conv_block(self.up3_ch + filters[1], filters[1])
        self.Up6_Conv = nn.Sequential(
                self.conv1x1(filters[4], self.out_ch),
                nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True))
        self.Up5_Conv = nn.Sequential(
                self.conv1x1(filters[3], self.out_ch),
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True))
        self.Up4_Conv = nn.Sequential(
                self.conv1x1(filters[2], self.out_ch),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
        # self.Up6_Conv_vice = nn.Sequential(
        #     self.conv1x1(filters[4], self.out_ch+1),
        #     nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True))
        # self.Up5_Conv_vice = nn.Sequential(
        #     self.conv1x1(filters[3], self.out_ch+1),
        #     nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True))
        # self.Up4_Conv_vice = nn.Sequential(
        #     self.conv1x1(filters[2], self.out_ch+1),
        #     nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
        # self.Up7 = up_conv(last_dim, filters[5])
        # self.upconv7 = conv_block(self.up7_ch + filters[5], filters[5])

        # self.Up6 = up_conv(filters[4], filters[3])
        # self.upconv6 = conv_block(self.up6_ch + filters[3], filters[3])
        # self.Up5 = up_conv(filters[3], filters[2])
        # self.upconv5 = conv_block(self.up5_ch + filters[2], filters[2])
        # self.Up4 = up_conv(filters[2], filters[1])
        # self.upconv4 = conv_block(self.up4_ch + filters[1], filters[1])
        # self.Up3 = up_conv(filters[1], filters[0])
        # self.upconv3 = conv_block(self.up3_ch + filters[0], filters[0])

        # self.Up2 = up_conv(filters[1], filters[0])
        # self.Up1 = up_conv(filters[0], 16)

        # self.Conv = nn.Conv2d(filters[0], self.out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv = nn.Conv2d(filters[1], self.out_ch, kernel_size=1, stride=1, padding=0)
        self.Conv_vice = nn.Conv2d(filters[1], self.out_ch+1, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.affine == True:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def conv1x1(self, in_planes, out_planes, stride=1, bias=True):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

    def forward(self, x):
        block_data = self.input_block(x)
        block_data0 = block_data
        block_data = self.maxpool1(block_data)
        block_data = self.input_block2(block_data)
        block_data1 = block_data
        self.downsam = []
        for i, block in enumerate(self.blocks):
            block_data = block(block_data)
            self.downsam.append(block_data)
        block_data = self.conv1_1_block(block_data)
        # block_data = self.Up7(block_data)
        # block_data = torch.cat((self.downsam[4],self.downsam[3],self.downsam[2], block_data), dim=1)
        # block_data = self.upconv7(block_data)
        block_data = self.Up6(block_data) # 32
        block_data = torch.cat((self.downsam[1], block_data), dim=1)
        block_data = self.upconv6(block_data)
        up6 = block_data
        block_data = self.Up5(block_data)
        block_data = torch.cat((self.downsam[0], block_data), dim=1)
        block_data = self.upconv5(block_data)
        up5 = block_data
        block_data = self.Up4(block_data)
        block_data = torch.cat((block_data1, block_data), dim=1)
        block_data = self.upconv4(block_data)
        up4 = block_data
        block_data = self.Up3(block_data)
        block_data = torch.cat((block_data0, block_data), dim=1)
        block_data = self.upconv3(block_data)
        # block_data = self.Up2(block_data)
        # block_data = self.Up1(block_data)

        # block_data = self.Up_conv2(block_data)
        out = self.Conv(block_data)
        out_vice = self.Conv_vice(block_data)
        deeps = []
        deeps_vice = []
        for seg,deep in zip(
                [up6, up5, up4],
                [self.Up6_Conv,self.Up5_Conv,self.Up4_Conv]):
            deeps.append(deep(seg))
        # for seg, deep in zip(
        #         [up6, up5, up4],
        #         [self.Up6_Conv_vice, self.Up5_Conv_vice, self.Up4_Conv_vice]):
        #     deeps_vice.append(deep(seg))
        # out = self.global_pooling(block_data)
        # out = torch.flatten(out, 1)
        # logits = self.classifier(out)
        return out
# sliver07 experiment