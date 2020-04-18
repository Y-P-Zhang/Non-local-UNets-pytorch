import torch
import torch.nn as nn

from utils.attention import MultiHeadAttention3d

import config

conf = config.args


# basic modules
def bn_relu(filters):
    sequence = [nn.BatchNorm3d(filters, momentum=0.997), nn.ReLU6(inplace=True)]
    return nn.Sequential(*sequence)


class Network(nn.Module):

    def __init__(self, conf):
        super(Network, self).__init__()
        self.in_channels = conf.in_channels
        self.num_classes = conf.num_classes
        self.num_filters = conf.num_filters
        # Input Block
        self.input_block = nn.Conv3d(self.in_channels, self.num_filters, kernel_size=3, padding=1)
        # down_sample_block
        self.down_sample_block1 = EncodeingBlockLayer(self.num_filters, self.num_filters * 2,
                                                      stride=2, blocks=1)
        self.down_sample_block2 = EncodeingBlockLayer(self.num_filters * 2, self.num_filters * 4,
                                                      stride=2, blocks=1)
        num_filters = self.num_filters * 4
        self.bn_relu = bn_relu(num_filters)
        # Bottom Block
        self.bottom_block = MultiHeadAttention3d(num_filters, num_filters,
                                                 num_filters, num_filters, 2, layer_type='SAME')
        # Up-Sampling Block
        self.up_sample_block1 = AttDecodeingBlockLayer(num_filters, num_filters, blocks=1, strides=2)
        self.up_sample_block2 = AttDecodeingBlockLayer(num_filters//2, num_filters, blocks=1, strides=2)

        # Output Block
        self.output_block = nn.Sequential(*[nn.BatchNorm3d(num_filters, momentum=0.997),
                                            nn.ReLU6(inplace=True),
                                            nn.Dropout(0.5),
                                            nn.Conv3d(num_filters, self.num_classes, kernel_size=1, bias=True)])

    def forward(self, x):
        skip_inputs = []
        x = self.input_block(x)
        print('1', x.shape)
        x = self.down_sample_block1(x)
        print('2', x.shape)
        skip_inputs.append(x)
        x = self.down_sample_block2(x)
        print('3', x.shape)
        skip_inputs.append(x)
        x = self.bn_relu(x)
        print('4', x.shape)
        x = self.bottom_block(x)
        x += skip_inputs[-1]
        print('5', x.shape, skip_inputs[1].shape)
        x = self.up_sample_block1(x, skip_inputs[1])
        print('6', x.shape, skip_inputs[0].shape)
        x = self.up_sample_block2(x, skip_inputs[0])
        print('7', x.shape)
        x = self.output_block(x)
        print('8', x.shape)

        return x


class EncodeingBlockLayer(nn.Module):

    def __init__(self, in_channels, out_channels, stride, blocks):
        super(EncodeingBlockLayer, self).__init__()
        self.blocks = blocks
        self.projection_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.residual_block1 = ResidualBlock(in_channels, out_channels, stride=stride,
                                             projection_shortcut=self.projection_shortcut)
        self.residual_block2 = ResidualBlock(out_channels, out_channels, stride=1)

    def forward(self, x):
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        return x


class AttDecodeingBlockLayer(nn.Module):
    def __init__(self, in_channels, filters, blocks, strides):
        super(AttDecodeingBlockLayer, self).__init__()
        self.attention_block = AttentionBlock(filters, filters, strides, projection_shortcut=True)
        self.projection_shortcut = nn.ConvTranspose3d(in_channels, filters,
                                                      kernel_size=3, stride=strides, padding=1, output_padding=1)
        self.residual_block1 = ResidualBlock(filters, filters, stride=1)

    def forward(self, inputs, skip_inputs):
        if self.projection_shortcut:
            skip_inputs = self.projection_shortcut(skip_inputs)
        inputs = self.attention_block(inputs)
        inputs = inputs + skip_inputs
        outputs = self.residual_block1(inputs)
        return outputs


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, projection_shortcut=None):
        super(ResidualBlock, self).__init__()
        self.bn_relu1 = bn_relu(in_channels)
        self.bn_relu2 = bn_relu(out_channels)
        if projection_shortcut:
            self.projection_shortcut = projection_shortcut
        else:
            self.projection_shortcut = None
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        shortcut = inputs
        if self.projection_shortcut:
            shortcut = self.projection_shortcut(shortcut)
        inputs = self.conv1(self.bn_relu1(inputs))
        inputs = self.conv2(self.bn_relu2(inputs))
        return inputs + shortcut


class AttentionBlock(nn.Module):

    def __init__(self, in_channels, filters, strides, projection_shortcut=False):
        super(AttentionBlock, self).__init__()
        self.projection_shortcut = nn.ConvTranspose3d(in_channels, filters, kernel_size=3, stride=strides, padding=1,
                                                      output_padding=1) if \
            projection_shortcut else None
        self.bn_relu = bn_relu(in_channels)
        if strides != 1:
            layer_type = 'UP'
        else:
            layer_type = 'SAME'
        self.multihead_attention_3d = MultiHeadAttention3d(in_channels, filters, filters, filters,
                                                           num_heads=1, layer_type=layer_type)

    def forward(self, inputs):
        shortcut = inputs
        inputs = self.bn_relu(inputs)
        if self.projection_shortcut:
            shortcut = self.projection_shortcut(shortcut)
        inputs = self.multihead_attention_3d(inputs)
        return inputs + shortcut


def test():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    net = Network(conf).to(device)
    x = torch.ones([2, 2, 16, 16, 12]).to(device)
    y = net(x)
    print(net)
test()
