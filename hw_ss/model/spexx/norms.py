import torch.nn as nn
import torch


class ChannelLN(nn.LayerNorm):
    """
    Channel-wise layer normalization based on nn.LayerNorm
    Input: 3D tensor with [batch_size(N), channel_size(C), frame_num(T)]
    Output: 3D tensor with same shape
    """

    def __init__(self, *args, **kwargs):
        super(ChannelLN, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} requires a 3D tensor input".format(
                self.__name__))
        x = x.transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2)
        return x



class GlobalLN(nn.Module):
    """
    Global layer normalization
    Input: 3D tensor with [batch_size(N), channel_size(C), frame_num(T)]
    Output: 3D tensor with same shape
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLN, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} requires a 3D tensor input".format(
                self.__name__))
        # calculate the mean, variance over the channel and time dimensions
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)



















