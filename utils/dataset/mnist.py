#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/16
# @Author  : Joyful Buffalo
# @File    : mnist.py
import torch
import torchvision


def mnist_load_data(g):
    train_dataset = torchvision.datasets.MNIST(
        root='dataset/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root='dataset/',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1024,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=8,
        generator=g,
        persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1024,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=8,
        generator=g,
        persistent_workers=True
    )
    return train_loader, test_loader