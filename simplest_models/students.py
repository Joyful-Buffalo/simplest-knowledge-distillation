#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/16
# @Author  : Joyful Buffalo
# @File    : students.py
from torch import nn
import torch.nn.functional as F


class StudentCnn(nn.Module):
    def __init__(self, c=5):
        super(StudentCnn, self).__init__()
        self.c = c
        self.conv1 = nn.Conv2d(1, c, kernel_size=5)
        self.conv2 = nn.Conv2d(c, c, kernel_size=5)
        self.fc1 = nn.Linear(c * 4 * 4, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.c * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class StudentMLP(nn.Module):
    def __init__(self):
        super(StudentMLP, self).__init__()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
