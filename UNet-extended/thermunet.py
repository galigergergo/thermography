#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Modified version from https://github.com/jvanvugt/pytorch-unet

import torch
from torch import nn
import torch.nn.functional as F



class ThermUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_classes=1,
        depth=5,
        wf=4,
        padding=True,
        batch_norm=False,
        up_mode='upsample',
    ):
        """
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(ThermUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.depth = depth
        self.wf = wf
        self.padding = padding
        self.batch_norm = batch_norm
        assert up_mode in ('upconv', 'upsample')
        self.up_mode = up_mode
        
        self.down_path = nn.ModuleList()

        prev_channels = self.in_channels
        for i in range(self.depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), self.padding, self.batch_norm)
            )
            prev_channels = 2 ** (self.wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(self.depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), self.up_mode, self.padding, self.batch_norm)
            )
            prev_channels = 2 ** (self.wf + i)

        self.last = nn.Conv2d(prev_channels, self.n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                UNETUpsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class UNETUpsample(nn.Module):
    """ Used in the expanding path """
    def __init__(self, scale_factor, mode="bilinear", align_corners=None):
        super(UNETUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x
        