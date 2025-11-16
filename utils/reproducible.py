#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/16
# @Author  : Joyful Buffalo
# @File    : reproducible.py
import os

import torch


def set_seed(seed=2025):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def enforce_determinism():
    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
