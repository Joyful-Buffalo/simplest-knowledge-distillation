#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/16
# @Author  : Joyful Buffalo
# @File    : teachers.py
from torch import nn
import torch.nn.functional as F


class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(40)
        self.conv2 = nn.Conv2d(40, 40, kernel_size=5)
        self.fc1 = nn.Linear(40 * 4 * 4, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 40 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
