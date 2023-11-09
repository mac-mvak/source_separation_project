import torch 
import torch.nn as nn

from .norms import ChannelLN, GlobalLN


class TCNBlock(nn.Module):
    """
    Temporal convolutional network block,
        1x1Conv - PReLU - Norm - DConv - PReLU - Norm - SConv
    Input: 3D tensor with [N, C_in, L_in]
    Output: 3D tensor with [N, C_out, L_out]
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 causal=False):
        super(TCNBlock, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = GlobalLN(conv_channels, elementwise_affine=True) if not causal else ( 
            ChannelLN(conv_channels, elementwise_affine=True))
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.norm2 = GlobalLN(conv_channels, elementwise_affine=True) if not causal else ( 
            ChannelLN(conv_channels, elementwise_affine=True))
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.norm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.norm2(self.prelu2(y))
        y = self.sconv(y)
        y += x
        return y


class TCNBlock_Spk(nn.Module):
    """
    Temporal convolutional network block,
        1x1Conv - PReLU - Norm - DConv - PReLU - Norm - SConv
        The first tcn block takes additional speaker embedding as inputs
    Input: 3D tensor with [N, C_in, L_in]
    Input Speaker Embedding: 2D tensor with [N, D]
    Output: 3D tensor with [N, C_out, L_out]
    """

    def __init__(self,
                 in_channels=256,
                 spk_embed_dim=100,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 causal=False):
        super(TCNBlock_Spk, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels+spk_embed_dim, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = GlobalLN(conv_channels, elementwise_affine=True) if not causal else ( 
            ChannelLN(conv_channels, elementwise_affine=True))
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.norm2 = GlobalLN(conv_channels, elementwise_affine=True) if not causal else ( 
            ChannelLN(conv_channels, elementwise_affine=True))
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        self.causal = causal
        self.dconv_pad = dconv_pad
        self.dilation = dilation

    def forward(self, x, aux):
        # Repeatedly concated speaker embedding aux to each frame of the representation x
        T = x.shape[-1]
        aux = torch.unsqueeze(aux, -1)
        aux = aux.repeat(1,1,T)
        y = torch.cat([x, aux], 1)
        y = self.conv1x1(y)
        y = self.norm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.norm2(self.prelu2(y))
        y = self.sconv(y)
        y += x
        return y



class ResBlock(nn.Module):
    """
    Resnet block for speaker encoder to obtain speaker embedding
    ref to 
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
        and
        https://github.com/Jungjee/RawNet/blob/master/PyTorch/model_RawNet.py
    """
    def __init__(self, in_dims, out_dims):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_dims, out_dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_dims)
        self.batch_norm2 = nn.BatchNorm1d(out_dims)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.maxpool = nn.MaxPool1d(3)
        if in_dims != out_dims:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_dims, out_dims, kernel_size=1, bias=False)
        else:
            self.downsample = False

    def forward(self, x):
        y = self.conv1(x)
        y = self.batch_norm1(y)
        y = self.prelu1(y)
        y = self.conv2(y)
        y = self.batch_norm2(y)
        if self.downsample:
            y += self.conv_downsample(x)
        else:
            y += x
        y = self.prelu2(y)
        return self.maxpool(y)