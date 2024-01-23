# -*- coding: utf-8 -*-
"""

Reproduce ISTA-Net (DOI 10.1109/CVPR.2018.00196)

2020/11/03

@author: XIANG
"""


import torch 
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np 
import os


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)


# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))


        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb, mask):
        
        # print("lambda_step: ", self.lambda_step)
        # print("soft_thr: ", self.soft_thr)

        # convert data format from (batch_size, channel, pnum, pnum) to (circle_num, batch_size)
        pnum = x.size()[2]
        x = x.view(x.size()[0], x.size()[1], pnum*pnum, -1)   # (batch_size, channel, pnum*pnum, 1)
        x = torch.squeeze(x, 1)
        x = torch.squeeze(x, 2).t()             
        x = mask.mm(x)  
        
        # rk block in the paper
        x = x - self.lambda_step  * PhiTPhi.mm(x) + self.lambda_step * PhiTb

        # convert (circle_num, batch_size) to (batch_size, channel, pnum, pnum)
        x = torch.mm(mask.t(), x)
        x = x.view(pnum, pnum, -1)
        x = x.unsqueeze(0)
        x_input = x.permute(3, 0, 1, 2)

        x_D = F.conv2d(x_input, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_G

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]


# Define ISTA-Net
class ISTANet(torch.nn.Module):
    def __init__(self, LayerNo, Phi, mask):
        super(ISTANet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.Phi = Phi
        self.mask = mask

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Qinit, b):
        
        # convert data format from (batch_size, channel, vector_row, vector_col) to (vector_row, batch_size)
        b = torch.squeeze(b, 1)
        b = torch.squeeze(b, 2)
        b = b.t()

        PhiTPhi = self.Phi.t().mm(self.Phi)
        PhiTb = self.Phi.t().mm(b)
        x = Qinit
        layers_sym = []   # for computing symmetric loss
        xnews = [] # iteration result
        xnews.append(x)
        
        for i in range(self.LayerNo):
            # print("iteration #{}:".format(i))
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb, self.mask)
            layers_sym.append(layer_sym)
            xnews.append(x)

        x_final = x

        return [x_final, layers_sym]