# -*- coding: utf-8 -*-
"""
@created on: 11/24/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch import tensor
import cv2
import numpy as np


class SmallConvNet(nn.Module):

    def __init__(self, args):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 4, 2)
        self.fc1 = nn.Linear(2 * 2 * 64, 512)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, args.num_classes)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = tensor(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 2 * 2 * 64)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_pass_kernels(self, x):
        x = tensor(x).unsqueeze(0).unsqueeze(0).float()
        conv1 = F.relu(self.conv1(x))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        return [conv1, conv2, conv3]


class VariableConvNet(nn.Module):
    def __init__(self, args):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super().__init__()
        self.output_num = [4, 2, 1]
        self.conv1 = nn.Conv2d(1, 32, 4, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 4, 2)
        self.fc1 = nn.Linear(1344, 512)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, args.num_classes)

    def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer

        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''
        for i in range(len(out_pool_size)):
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            h_pad = int((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
            w_pad = int((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(previous_conv)
            if i == 0:
                spp = x.view(num_sample, -1)
                # print("spp size:",spp.size())
            else:
                # print("size:",spp.size())
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)
        return spp

    def forward(self, input_data):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        max_height = -1
        max_width = -1
        for image in input_data:
            h = image.shape[0]
            w = image.shape[1]
            if h > max_height:
                max_height = h
            if w > max_width:
                max_width = w

        for i, image in enumerate(input_data):
            # image = cv2.resize(image, (max_height, max_width))
            new_image = np.zeros((max_height, max_width)) + 255
            new_image[:image.shape[0], :image.shape[1]] = image
            image = new_image
            image = tensor(image).unsqueeze(0)
            if i == 0:
                x = image
            else:
                x = torch.cat((x, image), dim=0)
        x = x.float().unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        # x = self.pool2(x)
        x = F.relu(self.conv3(x))
        spp = self.spatial_pyramid_pool(x, int(x.size(0)), [int(x.size(2)), int(x.size(3))], self.output_num)
        x = F.relu(self.fc1(spp))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
