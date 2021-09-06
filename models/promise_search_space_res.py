import torch.nn as nn

from .operations import OPS
from .promise_search_space_base import Conv1_1_Block, Block, conv_block, up_conv
from .promise_search_space_base import Network as BaseSearchSpace

# ###promise12experiment
# class Network(BaseSearchSpace):
#     def __init__(self, init_ch, dataset, config, groups=1, base_width=64, dilation=1, norm_layer=None):
#         super(Network, self).__init__(init_ch, dataset, config)
#
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.input_block = nn.Sequential(
#             nn.Conv2d(1, self._C_input, kernel_size=3, stride=1, padding=1, bias=False),
#             norm_layer(self._C_input),
#             nn.ReLU(inplace=True),
#         )
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self._C_input2 = 128
#         self.input_block2 = nn.Sequential(
#             nn.Conv2d(self._C_input, self._C_input2, kernel_size=3, stride=1, padding=1, bias=False),
#             norm_layer(self._C_input2),
#             nn.ReLU(inplace=True),
#         )
#
#         self.blocks = nn.ModuleList()
#
#         for i in range(self.num_blocks):
#             input_config = self.input_configs[i]
#             self.blocks.append(Block(
#                 input_config['in_chs'],
#                 input_config['ch'],
#                 input_config['strides'],
#                 input_config['num_stack_layers'],
#                 self.config
#             ))
#
#
#         if 'bottle_neck' in self.config.search_params.PRIMITIVES_stack:
#             conv1_1_input_dim = [ch * 4 for ch in self.input_configs[-1]['in_chs']]
#             last_dim = self.config.optim.last_dim * 4
#         else:
#             conv1_1_input_dim = self.input_configs[-1]['in_chs']
#             last_dim = self.config.optim.last_dim
#         self.conv1_1_block = Conv1_1_Block(conv1_1_input_dim, last_dim)
#         self.Up1 = up_conv(self.config.optim.last_dim, self.config.optim.last_dim // 2)
#         self.upconv1 = conv_block(self.config.optim.last_dim // 2 + 512,self.config.optim.last_dim // 2)
#         self.Up2 = up_conv(self.config.optim.last_dim // 2, self.config.optim.last_dim // 4)
#         self.upconv2 = conv_block(self.config.optim.last_dim // 4 + 256, self.config.optim.last_dim // 4)
#         self.Up3 = up_conv(self.config.optim.last_dim // 4, self.config.optim.last_dim // 8)
#         self.upconv3 = conv_block(self.config.optim.last_dim // 8 + 128, self.config.optim.last_dim // 8)
#         # self.Up4 = up_conv(self.config.optim.last_dim // 8, self.config.optim.last_dim // 32)
#         # self.upconv4 = conv_block(self.config.optim.last_dim // 32 + 32, self.config.optim.last_dim // 32)
#         self.Up5 = up_conv(self.config.optim.last_dim // 8, self.config.optim.last_dim // 16)
#         self.upconv5 = conv_block(self.config.optim.last_dim // 16 + 64, self.config.optim.last_dim // 16)
#         self.last_conv = conv_block(self.config.optim.last_dim // 16, self._num_classes)
#         # self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
#         # self.classifier = nn.Linear(last_dim, self._num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 if m.affine == True:
#                     nn.init.constant_(m.weight, 1)
#                     nn.init.constant_(m.bias, 0)
# ###promise12experiment

###sliver07experiment
class Network(BaseSearchSpace):
    def __init__(self, init_ch, dataset, config, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Network, self).__init__(init_ch, dataset, config)

        if norm_layer is None:
            norm_layer = nn.InstanceNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.input_block = nn.Sequential(
            nn.Conv3d(1, self._C_input, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(self._C_input),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(self._C_input, self._C_input, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(self._C_input),
            nn.LeakyReLU(inplace=True),
        )
        # self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        # self._C_input2 = 128
        # self.input_block2 = nn.Sequential(
        #     nn.Conv2d(self._C_input, self._C_input2, kernel_size=3, stride=1, padding=1, bias=False),
        #     norm_layer(self._C_input2),
        #     nn.ReLU(inplace=True),
        # )

        self.blocks = nn.ModuleList()

        for i in range(self.num_blocks):
            input_config = self.input_configs[i]
            self.blocks.append(Block(
                input_config['in_chs'],
                input_config['ch'],
                input_config['strides'],
                input_config['num_stack_layers'],
                self.config
            ))


        if 'bottle_neck' in self.config.search_params.PRIMITIVES_stack:
            conv1_1_input_dim = [ch * 4 for ch in self.input_configs[-1]['in_chs']]
            last_dim = self.config.optim.last_dim * 4
        else:
            conv1_1_input_dim = self.input_configs[-1]['in_chs']
            last_dim = self.config.optim.last_dim
        self.conv1_1_block = Conv1_1_Block(conv1_1_input_dim, last_dim)
        self.Up1 = up_conv(self.config.optim.last_dim, self.config.optim.last_dim)
        self.upconv1 = conv_block(640, 320)
        self.Up2 = up_conv(self.config.optim.last_dim, 256)
        self.upconv2 = conv_block(512, 256)
        self.Up3 = up_conv(256, 128)
        self.upconv3 = conv_block(256, 128)
        self.Up4 = up_conv(128, 64)
        self.upconv4 = conv_block(128, 64)
        self.Up5 = up_conv(64, 32)
        self.upconv5 = conv_block(64, 32)
        self.last_conv = nn.Conv3d(32, self._num_classes, kernel_size=(1,1,1), stride=(1,1,1), bias=False)
        # self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Linear(last_dim, self._num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.affine == True:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
###sliver07experiment
