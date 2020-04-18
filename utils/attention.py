import torch
import torch.nn as nn


class MultiHeadAttention3d(torch.nn.Module):
    def __init__(self, in_channels, total_key_filters, total_value_filters,
                 out_channels, num_heads, dropout_prob=0.5, layer_type='SAME'):
        super(MultiHeadAttention3d, self).__init__()

        if total_key_filters % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_filters, num_heads))
        if total_value_filters % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_filters, num_heads))
        if layer_type not in ['SAME', 'DOWN', 'UP']:
            raise ValueError("Layer type (%s) must be one of SAME, "
                             "DOWN, UP." % (layer_type))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.total_key_filters = total_key_filters
        self.total_value_filters = total_value_filters
        self.num_heads = num_heads
        self.layer_type = layer_type
        self.compute_qkv_3d = ComputeQkv3D(in_channels, total_key_filters, total_value_filters, layer_type)
        self.dropout = nn.Dropout(dropout_prob)
        self.outconv = nn.Conv3d(self.total_value_filters, self.out_channels, kernel_size=1, stride=1, padding=0,
                                 bias=True)

    def forward(self, inputs):
        """
        inputs: Tensor with shape [batch, channels, D, H, W]
        return: Tensor with shape [batch, channels, D, H, W]
        """
        q, k, v = self.compute_qkv_3d(inputs)
        shape = q.shape

        k = self.split_heads_3d(k, self.num_heads)
        v = self.split_heads_3d(v, self.num_heads)
        q = self.split_heads_3d(q, self.num_heads)

        # normalize
        key_filters_per_head = self.total_key_filters // self.num_heads
        q *= key_filters_per_head ** -0.5

        output = self.global_attention_3d(q, k, v, shape)
        return self.outconv(output)

    @staticmethod
    def split_heads_3d(x, num_heads):
        """
        Args:
            x: Tensor with shape  [B D H W channels]
            num_heads: an integer
        Return:
            Tensor with shape [B, num_heads, D, H, W, channels / num_heads]
        """
        channels = x.shape[-1]
        return x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], num_heads, int(channels / num_heads))

    def global_attention_3d(self, q, k, v, shape):

        k = torch.flatten(k, 0, 4)  # [B, D, H, W, N, C]
        v = torch.flatten(v, 0, 4)
        q = torch.flatten(q, 0, 4)

        attention_weight = torch.matmul(q, k.transpose(0, 1))
        attention_weight = torch.softmax(attention_weight, dim=1)
        attention_weight = self.dropout(attention_weight)

        output = torch.matmul(attention_weight, v)
        output = output.view(shape[0], shape[1], shape[2], shape[3], v.shape[-1] * self.num_heads)
        output = output.permute(0, 4, 1, 2, 3)  # [B C D H W]
        return output


class ComputeQkv3D(nn.Module):
    """Computes query, key and value.

    Args:
        inputs: a Tensor with shape [batch, channels, d, h, w] # Differnet with tensorflow
        in_channels: Conv input channels
        total_key_filters: an integer
        total_value_filters: and integer
        layer_type: String, type of this layer -- SAME, DOWN, UP

    Returns:
        q: [batch, _d, _h, _w, total_key_filters] tensor # Same with tensorflow
        k: [batch, h, w, total_key_filters] tensor
        v: [batch, h, w, total_value_filters] tensor
    """

    def __init__(self, in_channels, total_key_filters, total_value_filters, layer_type):
        super(ComputeQkv3D, self).__init__()
        self.in_channels = in_channels
        self.total_key_filters = total_key_filters
        self.total_value_filters = total_value_filters
        self.layer_type = layer_type

        if self.layer_type == 'SAME':
            self.qconv = nn.Conv3d(in_channels, total_key_filters, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        elif self.layer_type == 'DOWN':
            self.qconv = nn.Conv3d(in_channels, total_key_filters, kernel_size=3, stride=2,
                                   padding=1, bias=True)
        elif self.layer_type == 'UP':
            self.qconv = nn.ConvTranspose3d(in_channels, total_key_filters, kernel_size=3, stride=2,
                                            padding=1, output_padding=1, bias=True)

        self.kconv = nn.Conv3d(in_channels, total_key_filters, kernel_size=1, stride=1, padding=0, bias=True)
        self.vconv = nn.Conv3d(in_channels, total_value_filters, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        q = self.qconv(x)
        k = self.kconv(x)
        v = self.vconv(x)
        # [B D H W C]
        return q.permute(0, 2, 3, 4, 1), k.permute(0, 2, 3, 4, 1), v.permute(0, 2, 3, 4, 1)


def test():
    net = MultiHeadAttention3d(2, 16, 16, 4, 4, layer_type='DOWN')  # 'SAME', 'DOWN', 'UP'
    x = torch.rand(2, 2, 16, 8, 4)
    if torch.cuda.is_available():
        x = x.cuda()
        net = net.cuda()
    y = net(x)
    print(y.shape)

#test()
