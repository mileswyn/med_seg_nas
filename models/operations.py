import torch
import torch.nn as nn

OPS = {
    'mbconv_k3_t1': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 3, stride, 1, t=1, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k3_t3': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 3, stride, 1, t=3, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k3_t6': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 3, stride, 1, t=6, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k5_t3': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 5, stride, 2, t=3, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k5_t6': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 5, stride, 2, t=6, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k7_t3': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 7, stride, 3, t=3, affine=affine, track_running_stats=track_running_stats),
    'mbconv_k7_t6': lambda C_in, C_out, stride, affine, track_running_stats: MBConv(C_in, C_out, 7, stride, 3, t=6, affine=affine, track_running_stats=track_running_stats),
    'basic_block': lambda C_in, C_out, stride, affine, track_running_stats: BasicBlock(C_in, C_out, stride, affine=affine, track_running_stats=track_running_stats),
    'bottle_neck': lambda C_in, C_out, stride, affine, track_running_stats: Bottleneck(C_in, C_out, stride, affine=affine, track_running_stats=track_running_stats),
    'skip_connect': lambda C_in, C_out, stride, affine, track_running_stats: Skip(C_in, C_out, 1, affine=affine, track_running_stats=track_running_stats),
    'cweight': lambda C_in, C_out, stride, affine, track_running_stats: CWeightOp(C_in, C_out, stride=stride, affine=affine, track_running_stats=track_running_stats),
    'sep_conv_3x3': lambda C_in, C_out, stride, affine, track_running_stats: SepConv(C_in, C_out, kernel_size=3, stride=stride, padding=1, affine=affine, track_running_stats=track_running_stats),
    'dil_conv_3x3': lambda C_in, C_out, stride, affine, track_running_stats: DilConv(C_in, C_out, kernel_size=3, stride=stride, padding=2, dilation=2, affine=affine, track_running_stats=track_running_stats),
}

# class DilConv(nn.Module):

#     def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, track_running_stats=True):
#         super(DilConv, self).__init__()
#         self.op = nn.Sequential(
#             nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
#                       groups=C_in, bias=False),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
#             nn.ReLU(inplace=False),
#         )

#     def forward(self, x):
#         return self.op(x)

class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True, track_running_stats=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm3d(C_out, affine=affine, track_running_stats=track_running_stats),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

# class SepConv(nn.Module):

#     def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, track_running_stats=True):
#         super(SepConv, self).__init__()
#         self.op = nn.Sequential(
#             nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
#             nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_in, affine=affine, track_running_stats=track_running_stats),
#             nn.ReLU(inplace=False),
#             nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
#             nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
#             nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats),
#             nn.ReLU(inplace=False),
#         )

#     def forward(self, x):
#         return self.op(x)
class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True, track_running_stats=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv3d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm3d(C_in, affine=affine, track_running_stats=track_running_stats),
            nn.LeakyReLU(inplace=False),
            nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv3d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm3d(C_out, affine=affine, track_running_stats=track_running_stats),
            nn.LeakyReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class AbstractOp(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @property
    def unit_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

# class BaseOp(AbstractOp):
#     def __init__(self, in_channels, out_channels, norm_type='gn', use_norm=True, affine=True,
#                  act_func='relu', dropout_rate=0, ops_order='weight_norm_act'):
#         super(BaseOp, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         self.use_norm = use_norm  # bool
#         self.act_func = act_func  # str
#         self.dropout_rate = dropout_rate  # float
#         self.ops_order = ops_order  # str
#         self.norm_type = norm_type  # str

#         # batch norm, group norm, instance norm, layer norm
#         if self.use_norm:
#             # Ref: <Group Normalization> https://arxiv.org/abs/1803.08494
#             # 16 channels for one group is best
#             if self.norm_before_weight:
#                 group = 1 if in_channels % 16 != 0 else in_channels // 16
#                 if norm_type == 'gn':
#                     self.norm = nn.GroupNorm(group, in_channels, affine=affine)
#                 else:
#                     self.norm = nn.BatchNorm2d(in_channels, affine=affine)
#             else:
#                 group = 1 if out_channels % 16 != 0 else out_channels // 16
#                 if norm_type == 'gn':
#                     self.norm = nn.GroupNorm(group, out_channels, affine=affine)
#                 else:
#                     self.norm = nn.BatchNorm2d(out_channels, affine=affine)
#         else:
#             self.norm = None

#         if act_func == 'relu':
#             if self.ops_list[0] == 'act':
#                 # change input data(inplace=True) or not
#                 self.activation = nn.ReLU(inplace=False)
#             else:
#                 self.activation = nn.ReLU(inplace=True)
#         elif act_func == 'relu6':
#             if self.ops_list[0] == 'act':
#                 self.activation = nn.ReLU6(inplace=False)
#             else:
#                 self.activation = nn.ReLU6(inplace=True)
#         else:
#             self.activation = None

#         # dropout
#         if self.dropout_rate > 0:
#             self.dropout = nn.Dropout2d(self.dropout_rate, inplace=False)
#         else:
#             self.dropout = None

#     @property
#     def ops_list(self):
#         # return a list
#         return self.ops_order.split('_')

#     @property
#     def norm_before_weight(self):
#         # raise error if no norm or weight in ops_list
#         # return a false
#         for op in self.ops_list:
#             if op == 'norm':
#                 return True
#             elif op == 'weight':
#                 return False
#         raise ValueError()

#     @property
#     def unit_str(self):
#         raise NotImplementedError

#     @property
#     def config(self):
#         return {
#             'in_channels': self.in_channels,
#             'out_channels': self.out_channels,
#             'use_norm': self.use_norm,
#             'act_func': self.act_func,
#             'dropout_rate': self.dropout_rate,
#             'ops_order': self.ops_order
#         }

#     @staticmethod
#     def build_from_config(config):
#         raise NotImplementedError

#     @staticmethod
#     def is_zero_ops():
#         return False

#     def get_flops(self, x):
#         raise NotImplementedError

#     def weight_call(self, x):
#         raise NotImplementedError

#     def forward(self, x):
#         for op in self.ops_list:
#             if op == 'weight':
#                 if self.dropout is not None:
#                     x = self.dropout(x)
#                 x = self.weight_call(x)
#             elif op == 'norm':
#                 if self.norm is not None:
#                     x = self.norm(x)
#             elif op == 'act':
#                 if self.activation is not None:
#                     x = self.activation(x)
#             else:
#                 raise ValueError('Unrecognized op: %s' % op)
#         return x
class BaseOp(AbstractOp):
    def __init__(self, in_channels, out_channels, norm_type='gn', use_norm=True, affine=True,
                 act_func='relu', dropout_rate=0, ops_order='weight_norm_act'):
        super(BaseOp, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_norm = use_norm  # bool
        self.act_func = act_func  # str
        self.dropout_rate = dropout_rate  # float
        self.ops_order = ops_order  # str
        self.norm_type = norm_type  # str

        # batch norm, group norm, instance norm, layer norm
        if self.use_norm:
            # Ref: <Group Normalization> https://arxiv.org/abs/1803.08494
            # 16 channels for one group is best
            if self.norm_before_weight:
                group = 1 if in_channels % 16 != 0 else in_channels // 16
                if norm_type == 'gn':
                    self.norm = nn.GroupNorm(group, in_channels, affine=affine)
                else:
                    self.norm = nn.InstanceNorm3d(in_channels, affine=affine)
            else:
                group = 1 if out_channels % 16 != 0 else out_channels // 16
                if norm_type == 'gn':
                    self.norm = nn.GroupNorm(group, out_channels, affine=affine)
                else:
                    self.norm = nn.InstanceNorm3d(out_channels, affine=affine)
        else:
            self.norm = None

        if act_func == 'relu':
            if self.ops_list[0] == 'act':
                # change input data(inplace=True) or not
                self.activation = nn.LeakyReLU(inplace=False)
            else:
                self.activation = nn.LeakyReLU(inplace=True)
        elif act_func == 'relu6':
            if self.ops_list[0] == 'act':
                self.activation = nn.LeakyReLU(inplace=False)
            else:
                self.activation = nn.LeakyReLU(inplace=True)
        else:
            self.activation = None

        # dropout
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout3d(self.dropout_rate, inplace=False)
        else:
            self.dropout = None

    @property
    def ops_list(self):
        # return a list
        return self.ops_order.split('_')

    @property
    def norm_before_weight(self):
        # raise error if no norm or weight in ops_list
        # return a false
        for op in self.ops_list:
            if op == 'norm':
                return True
            elif op == 'weight':
                return False
        raise ValueError()

    @property
    def unit_str(self):
        raise NotImplementedError

    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_norm': self.use_norm,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order
        }

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    @staticmethod
    def is_zero_ops():
        return False

    def get_flops(self, x):
        raise NotImplementedError

    def weight_call(self, x):
        raise NotImplementedError

    def forward(self, x):
        for op in self.ops_list:
            if op == 'weight':
                if self.dropout is not None:
                    x = self.dropout(x)
                x = self.weight_call(x)
            elif op == 'norm':
                if self.norm is not None:
                    x = self.norm(x)
            elif op == 'act':
                if self.activation is not None:
                    x = self.activation(x)
            else:
                raise ValueError('Unrecognized op: %s' % op)
        return x

# class MBConv(nn.Module):
#     def __init__(self, C_in, C_out, kernel_size, stride, padding, t=3, affine=True,
#                     track_running_stats=True, use_se=False):
#         super(MBConv, self).__init__()
#         self.t = t
#         if self.t > 1:
#             self._expand_conv = nn.Sequential(
#                 nn.Conv2d(C_in, C_in*self.t, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
#                 nn.BatchNorm2d(C_in*self.t, affine=affine, track_running_stats=track_running_stats),
#                 nn.ReLU6(inplace=True))

#             self._depthwise_conv = nn.Sequential(
#                 nn.Conv2d(C_in*self.t, C_in*self.t, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in*self.t, bias=False),
#                 nn.BatchNorm2d(C_in*self.t, affine=affine, track_running_stats=track_running_stats),
#                 nn.ReLU6(inplace=True))

#             self._project_conv = nn.Sequential(
#                 nn.Conv2d(C_in*self.t, C_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
#                 nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats))
#         else:
#             self._expand_conv = None

#             self._depthwise_conv = nn.Sequential(
#                 nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
#                 nn.BatchNorm2d(C_in, affine=affine, track_running_stats=track_running_stats),
#                 nn.ReLU6(inplace=True))

#             self._project_conv = nn.Sequential(
#                 nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(C_out))

#     def forward(self, x):
#         input_data = x
#         if self._expand_conv is not None:
#             x = self._expand_conv(x)
#         x = self._depthwise_conv(x)
#         out_data = self._project_conv(x)

#         if out_data.shape == input_data.shape:
#             return out_data + input_data
#         else:
#             return out_data

class MBConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, t=3, affine=True,
                    track_running_stats=True, use_se=False):
        super(MBConv, self).__init__()
        self.t = t
        if self.t > 1:
            self._expand_conv = nn.Sequential(
                nn.Conv3d(C_in, C_in*self.t, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.InstanceNorm3d(C_in*self.t, affine=affine, track_running_stats=track_running_stats),
                nn.LeakyReLU(inplace=True))

            self._depthwise_conv = nn.Sequential(
                nn.Conv3d(C_in*self.t, C_in*self.t, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in*self.t, bias=False),
                nn.InstanceNorm3d(C_in*self.t, affine=affine, track_running_stats=track_running_stats),
                nn.LeakyReLU(inplace=True))

            self._project_conv = nn.Sequential(
                nn.Conv3d(C_in*self.t, C_out, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                nn.InstanceNorm3d(C_out, affine=affine, track_running_stats=track_running_stats))
        else:
            self._expand_conv = None

            self._depthwise_conv = nn.Sequential(
                nn.Conv3d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
                nn.InstanceNorm3d(C_in, affine=affine, track_running_stats=track_running_stats),
                nn.LeakyReLU(inplace=True))

            self._project_conv = nn.Sequential(
                nn.Conv3d(C_in, C_out, 1, 1, 0, bias=False),
                nn.InstanceNorm3d(C_out))

    def forward(self, x):
        input_data = x
        if self._expand_conv is not None:
            x = self._expand_conv(x)
        x = self._depthwise_conv(x)
        out_data = self._project_conv(x)

        if out_data.shape == input_data.shape:
            return out_data + input_data
        else:
            return out_data

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2

class CWeightOp(BaseOp):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,dilation=1, groups=None,
                 bias=False, has_shuffle=False, use_transpose=False,output_padding=0, norm_type='gn',
                 use_norm=False, affine=True, act_func=None, dropout_rate=0, ops_order='weight', track_running_stats=True):
        super(CWeightOp, self).__init__(in_channels, out_channels, norm_type, use_norm, affine, act_func, dropout_rate, ops_order)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.has_shuffle = has_shuffle
        self.use_transpose = use_transpose
        self.output_padding = output_padding

        padding = get_same_padding(self.kernel_size)
        if isinstance(padding, int):
            padding *= self.dilation
        else:
            padding[0] *= self.dilation
            padding[1] *= self.dilation

        # `kernel_size`, `stride`, `padding`, `dilation`
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid()
        )
        norm_layer = nn.InstanceNorm3d
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels, affine=affine, track_running_stats=track_running_stats)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels, affine=affine, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                norm_layer(out_channels, affine=affine, track_running_stats=track_running_stats),
            )
        # if stride >= 2:
        #     self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
        #                           stride=stride, padding=padding, bias=False)
        #     group = 1 if out_channels % 16 != 0 else out_channels // 16
        #     self.norm = nn.GroupNorm(group, out_channels, affine=affine)

    @property
    def unit_str(self):
        if isinstance(self.kernel_size, int):
            kernel_size = (self.kernel_size, self.kernel_size)
        else:
            kernel_size = self.kernel_size
        basic_str = 'ChannelWeight'
        basic_str = 'Tran' + basic_str if self.use_transpose else basic_str
        return basic_str

    @staticmethod
    def build_from_config(config):
        return CWeightOp(**config)

    def weight_call(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        # rst = self.norm(self.conv(x * y)) if self.stride >= 2 else x * y
        rst = x * y
        # rst += x
        identity = rst
        out = self.conv1(rst)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(rst)
        out += identity
        out = self.relu(out)
        return out

# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# class BasicBlock(nn.Module):
#     def __init__(self, inplanes, planes, stride=1, groups=1,
#                  base_width=64, dilation=1, norm_layer=None,
#                  affine=True, track_running_stats=True):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes, affine=affine, track_running_stats=track_running_stats)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes, affine=affine, track_running_stats=track_running_stats)
#         self.downsample = None
#         if stride != 1 or inplanes != planes:
#             self.downsample = nn.Sequential(
#                 conv1x1(inplanes, planes, stride),
#                 norm_layer(planes, affine=affine, track_running_stats=track_running_stats),
#             )

#     def forward(self, x):  
#         identity = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)
#         out += identity
#         out = self.relu(out)

#         return out
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 affine=True, track_running_stats=True):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, affine=affine, track_running_stats=track_running_stats)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, affine=affine, track_running_stats=track_running_stats)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                norm_layer(planes, affine=affine, track_running_stats=track_running_stats),
            )

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
        out = self.relu(out)

        return out


# class Bottleneck(nn.Module):
#     def __init__(self, inplanes, planes, stride=1, affine=True, track_running_stats=True):
#         super(Bottleneck, self).__init__()
#         if inplanes != 32:
#             inplanes *= 4
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.stride = stride
#         self.downsample = None
#         if stride != 1 or inplanes != planes*4:
#             self.downsample = nn.Sequential(
#                 conv1x1(inplanes, planes * 4, stride),
#                 nn.BatchNorm2d(planes * 4, affine=affine, track_running_stats=track_running_stats),
#             )

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, affine=True, track_running_stats=True):
        super(Bottleneck, self).__init__()
        if inplanes != 32:
            inplanes *= 4
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.InstanceNorm3d(planes * 4)
        self.relu = nn.LeakyReLU(inplace=True)
        self.stride = stride
        self.downsample = None
        if stride != 1 or inplanes != planes*4:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * 4, stride),
                nn.InstanceNorm3d(planes * 4, affine=affine, track_running_stats=track_running_stats),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# class Skip(nn.Module):
#     def __init__(self, C_in, C_out, stride, affine=True, track_running_stats=True):
#         super(Skip, self).__init__()
#         if C_in!=C_out:
#             skip_conv = nn.Sequential(
#                 nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride, padding=0, groups=1, bias=False),
#                 nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats))
#             stride = 1
#         self.op=Identity(stride)

#         if C_in!=C_out:
#             self.op=nn.Sequential(skip_conv, self.op)

#     def forward(self,x):
#         return self.op(x)
class Skip(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True, track_running_stats=True):
        super(Skip, self).__init__()
        if C_in!=C_out:
            skip_conv = nn.Sequential(
                nn.Conv3d(C_in, C_out, kernel_size=1, stride=stride, padding=0, groups=1, bias=False),
                nn.InstanceNorm3d(C_out, affine=affine, track_running_stats=track_running_stats))
            stride = 1
            stride = 2 #NOTICE:we change it
        self.op=Identity(stride)

        if C_in!=C_out:
            self.op=nn.Sequential(skip_conv, self.op)

    def forward(self,x):
        return self.op(x)

# class Identity(nn.Module):
#     def __init__(self, stride):
#         super(Identity, self).__init__()
#         self.stride = stride

#     def forward(self, x):
#         if self.stride == 1:
#             return x
#         else:
#             return x[:, :, ::self.stride, ::self.stride]
class Identity(nn.Module):
    def __init__(self, stride):
        super(Identity, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x
        else:
            return x[:, :, ::self.stride, ::self.stride, ::self.stride]