# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/models/generator.py

import torch
import torch.nn as nn
from einops import rearrange
from .lsigmoid import LearnableSigmoid2D

def get_padding(kernel_size, dilation=1):
    """
    Calculate the padding size for a convolutional layer.
    
    Args:
    - kernel_size (int): Size of the convolutional kernel.
    - dilation (int, optional): Dilation rate of the convolution. Defaults to 1.
    
    Returns:
    - int: Calculated padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)

def get_padding_2d(kernel_size, dilation=(1, 1)):
    """
    Calculate the padding size for a 2D convolutional layer.
    
    Args:
    - kernel_size (tuple): Size of the convolutional kernel (height, width).
    - dilation (tuple, optional): Dilation rate of the convolution (height, width). Defaults to (1, 1).
    
    Returns:
    - tuple: Calculated padding size (height, width).
    """
    return (int((kernel_size[0] * dilation[0] - dilation[0]) / 2), 
            int((kernel_size[1] * dilation[1] - dilation[1]) / 2))

class DenseBlock(nn.Module):
    """
    DenseBlock module consisting of multiple convolutional layers with dilation.
    """
    def __init__(self, cfg, kernel_size=(3, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.cfg = cfg
        self.depth = depth
        self.dense_block = nn.ModuleList()
        self.hid_feature = cfg['model_cfg']['hid_feature']

        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(self.hid_feature * (i + 1), self.hid_feature, kernel_size, 
                          dilation=(dil, 1), padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(self.hid_feature, affine=True),
                nn.PReLU(self.hid_feature)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        """
        Forward pass for the DenseBlock module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Output tensor after processing through the dense block.
        """
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x
class FANDenseLayer(nn.Module):
    """A single dense-block layer with a FAN-style split activation.

    The conventional layer is  conv -> InstanceNorm -> PReLU  producing
    `hid_feature` channels. Here the same `hid_feature` output is split into:
        - a periodic part: cos(conv_p(x)) and sin(conv_p(x))   -> 2 * c_p channels
        - a gated part:    PReLU(InstanceNorm(conv_g(x)))       ->     c_g channels
    with 2 * c_p + c_g == hid_feature, so the dense-concat bookkeeping and the
    downstream dense_conv_2 are unchanged.

    Note: cos/sin are bounded in [-1, 1], so the periodic branch is left un-normed
    (normalising before a periodic nonlinearity is not what FAN does and is unneeded).
    The periodic conv is initialised small to avoid high-frequency oscillation across
    adjacent T-F bins; conv_p carries no bias (FAN's W_p has no bias term).
    """

    def __init__(
        self, in_channels, hid_feature, kernel_size=(3, 3), dilation=(1, 1), periodic_init_std=1e-2
    ):
        super(FANDenseLayer, self).__init__()
        self.c_p = hid_feature // 4
        self.c_g = hid_feature - 2 * self.c_p  # guarantees 2*c_p + c_g == hid_feature
        pad = get_padding_2d(kernel_size, dilation)

        self.conv_p = nn.Conv2d(
            in_channels, self.c_p, kernel_size, dilation=dilation, padding=pad, bias=False
        )
        self.conv_g = nn.Conv2d(in_channels, self.c_g, kernel_size, dilation=dilation, padding=pad)
        self.norm_g = nn.InstanceNorm2d(self.c_g, affine=True)
        self.act_g = nn.PReLU(self.c_g)

        nn.init.normal_(self.conv_p.weight, std=periodic_init_std)

    def forward(self, x):
        zp = self.conv_p(x)
        zg = self.act_g(self.norm_g(self.conv_g(x)))
        return torch.cat([torch.cos(zp), torch.sin(zp), zg], dim=1)  # hid_feature channels


class FANDenseBlock(nn.Module):
    """DenseBlock variant using FANDenseLayer. Drop-in shape-compatible with DenseBlock."""

    def __init__(self, cfg, kernel_size=(3, 3), depth=4):
        super(FANDenseBlock, self).__init__()
        self.cfg = cfg
        self.depth = depth
        self.hid_feature = cfg["model_cfg"]["hid_feature"]
        self.dense_block = nn.ModuleList()

        for i in range(depth):
            dil = 2**i
            self.dense_block.append(
                FANDenseLayer(
                    self.hid_feature * (i + 1),
                    self.hid_feature,
                    kernel_size=kernel_size,
                    dilation=(dil, 1),
                )
            )

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    """
    DenseEncoder module consisting of initial convolution, dense block, and a final convolution.
    """
    def __init__(self, cfg):
        super(DenseEncoder, self).__init__()
        self.cfg = cfg
        self.input_channel = cfg["model_cfg"]["input_channel"]
        self.hid_feature = cfg["model_cfg"]["hid_feature"]

        self.use_phase_fan = cfg["model_cfg"].get("use_phase_fan", False)
        self.use_fan_denseblock = cfg["model_cfg"].get("use_fan_denseblock", False)

        if self.use_phase_fan:
            assert self.input_channel == 2, (
                "use_phase_fan assumes input_channel == 2 ([magnitude, wrapped_phase])"
            )
            # learnable frequency; init 1.0 == plain (cos, sin) angle encoding
            self.phase_freq = nn.Parameter(torch.ones(1))
            conv1_in = self.input_channel + 1  # mag + cos(phi) + sin(phi)
        else:
            conv1_in = self.input_channel

        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(conv1_in, self.hid_feature, (1, 1)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature),
        )

        block_cls = FANDenseBlock if self.use_fan_denseblock else DenseBlock
        self.dense_block = block_cls(cfg, depth=4)

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature),
        )

    def _phase_lift(self, x):
        # x: [B, 2, T, F] -> [B, 3, T, F]
        mag = x[:, :1]
        phi = x[:, 1:2]
        z = self.phase_freq * phi
        return torch.cat([mag, torch.cos(z), torch.sin(z)], dim=1)

    def forward(self, x):
        if self.use_phase_fan:
            x = self._phase_lift(x)  # [batch, 3, time, freq]
        x = self.dense_conv_1(x)  # [batch, hid_feature, time, freq]
        x = self.dense_block(x)  # [batch, hid_feature, time, freq]
        x = self.dense_conv_2(x)  # [batch, hid_feature, time, freq//2]
        return x


class MagDecoder(nn.Module):
    """
    MagDecoder module for decoding magnitude information.
    """
    def __init__(self, cfg):
        super(MagDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = cfg['model_cfg']['output_channel']
        self.n_fft = cfg['stft_cfg']['n_fft']
        self.beta = cfg['model_cfg']['beta']

        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.Conv2d(self.hid_feature, self.output_channel, (1, 1)),
            nn.InstanceNorm2d(self.output_channel, affine=True),
            nn.PReLU(self.output_channel),
            nn.Conv2d(self.output_channel, self.output_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid2D(self.n_fft // 2 + 1, beta=self.beta)

    def forward(self, x):
        """
        Forward pass for the MagDecoder module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Decoded tensor with magnitude information.
        """
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = rearrange(x, 'b c t f -> b f t c').squeeze(-1)
        x = self.lsigmoid(x)
        x = rearrange(x, 'b f t -> b t f').unsqueeze(1)
        return x

class PhaseDecoder(nn.Module):
    """
    PhaseDecoder module for decoding phase information.
    """
    def __init__(self, cfg):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(cfg, depth=4)
        self.hid_feature = cfg['model_cfg']['hid_feature']
        self.output_channel = cfg['model_cfg']['output_channel']

        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

        self.phase_conv_r = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))

    def forward(self, x):
        """
        Forward pass for the PhaseDecoder module.
        
        Args:
        - x (torch.Tensor): Input tensor.
        
        Returns:
        - torch.Tensor: Decoded tensor with phase information.
        """
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x
