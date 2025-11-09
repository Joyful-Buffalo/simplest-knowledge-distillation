#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/9
# @Author  : Joyful Buffalo
# @File    : knowledge_distillation_run.py
import os
from copy import deepcopy

import torch
import torchvision
from torch import nn
from torch.nn import functional as F


def set_seed(seed=2025):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def enforce_determinism():
    """开启确定性算法并限制可能的非确定性来源。"""
    # [NEW] 打开确定性算法；若遇到无确定性实现的算子会抛错，便于排查
    torch.use_deterministic_algorithms(True)

    # [NEW] cuDNN 相关：只用确定性算法，禁用 benchmark(算法搜索会导致不确定)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # [NEW] 关闭 TF32，减少不同硬件/驱动间数值差异
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # [NEW] 可选：限制 CPU 线程，进一步减少并行归约顺序差异（如需极致位级一致）
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

def load_data(g):
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
        num_workers=0,
        generator=g
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1024,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
        generator=g
    )
    return train_loader, test_loader


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


def train_model(train_dataloader, model, criterion, optimizer, device):
    model.train()
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def test_model(model, test_dataloader, device):
    model.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(1)
            total += labels.size(0)
            correct = (predicted == labels).sum().item()
            acc += correct
    model_name = type(model).__name__
    print(f'Accuracy of the {model_name} on the test images {acc / total * 100:.2f}')
    model.train()
    return acc / total * 100


def train_main(train_dataloader, test_dataloader, teacher_model, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.01, betas=(0.9, 0.999))
    max_acc = 0
    for epoch in range(10):
        train_model(train_dataloader, teacher_model, criterion, optimizer, device)
        acc = test_model(teacher_model, test_dataloader, device)
        if acc > max_acc:
            max_acc = acc
    return max_acc


def distillation(student_model, teacher_model, train_dataloader, test_dataloader, device, temperature=2, alpha=0.5):
    hard_loss = nn.CrossEntropyLoss()
    soft_loss = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.01, betas=(0.9, 0.999))
    teacher_model.eval()
    student_model.train()
    max_acc = 0
    for epoch in range(10):
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = student_model(images)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            soft_loss_value = soft_loss(F.log_softmax(outputs / temperature, dim=1),
                                        F.softmax(teacher_outputs / temperature, dim=1)) * (temperature ** 2)
            hard_loss_value = hard_loss(outputs, labels)
            loss = alpha * soft_loss_value + (1 - alpha) * hard_loss_value
            loss.backward()
            optimizer.step()

        acc = test_model(student_model, test_dataloader, device)
        if acc > max_acc:
            max_acc = acc
    return max_acc


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    g = set_seed(2025)
    enforce_determinism()

    train_dataloader, test_dataloader = load_data(g)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('step 1: Train Teacher')
    teacher = Teacher().to(device)
    teacher_acc = train_main(train_dataloader, test_dataloader, teacher, device)
    print(f'Accuracy of the Teacher model on the test images: {teacher_acc:.2f}')

    print('step 2: Train StudentCNN')
    student_cnn = StudentCnn().to(device)
    student_cnn_acc = train_main(train_dataloader, test_dataloader, student_cnn, device)
    print(f'Accuracy of the StudentCNN model on the test images: {student_cnn_acc :.2f}')
    student_cnn_static = deepcopy(student_cnn.state_dict())

    print('step 3: Distillation Space StudentCnn')
    student_cnn = StudentCnn().to(device)
    student_cnn_space_acc = distillation(student_cnn, teacher, train_dataloader, test_dataloader, device)
    print(f'Accuracy of the Distillation Space StudentCnn on the test images: {student_cnn_space_acc:.2f}')

    print('step 4: Continue Distillation baseline StudentCnn')
    student_cnn = StudentCnn().to(device)
    student_cnn.load_state_dict(student_cnn_static)
    student_cnn_continue_acc = distillation(student_cnn, teacher, train_dataloader, test_dataloader, device)
    print(f'Accuracy of the Continue Distillation StudentCnn on the test images: {student_cnn_continue_acc:.2f}')

    print('step 5: Train StudentMLP')
    student_mlp = StudentMLP().to(device)
    student_mlp_acc = train_main(train_dataloader, test_dataloader, student_mlp, device)
    print(f'Accuracy of the StudentMLP model on the test images: {student_mlp_acc:.2f}')
    student_mlp_static = deepcopy(student_mlp.state_dict())

    print('step 6: Distillation Space StudentMLP')
    student_mlp = StudentMLP().to(device)
    student_mlp_space_acc = distillation(student_mlp, teacher, train_dataloader, test_dataloader, device)
    print(f'Accuracy of the StudentMLP model on the test images: {student_mlp_space_acc:.2f}')

    print('step 7: Continue Distillation baseline StudentMLP')
    student_mlp = StudentMLP().to(device)
    student_mlp.load_state_dict(student_mlp_static)
    student_mlp_continue_acc = distillation(student_mlp, teacher, train_dataloader, test_dataloader, device)
    print(f'Accuracy of the Continue Distillation StudentMLP on the test images: {student_mlp_continue_acc:.2f}')
    print(f'Teacher acc: {teacher_acc:.2f}')
    print(f'StudentCNN acc: {student_cnn_acc:.2f}|space acc: {student_cnn_space_acc:.2f}|continue acc: {student_cnn_continue_acc:.2f}')
    print(f'StudentMLP acc: {student_mlp_acc:.2f}|space acc: {student_mlp_space_acc:.2f}|continue acc: {student_mlp_continue_acc:.2f}')


if __name__ == '__main__':
    main()
# Teacher acc: 98.90
# StudentCNN acc: 97.86|space acc: 98.06|continue acc: 98.26
# StudentMLP acc: 94.96|space acc: 95.51|continue acc: 95.81