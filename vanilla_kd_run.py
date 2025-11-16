#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/9
# @Author  : Joyful Buffalo
# @File    : knowledge_distillation_run.py
import os
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from simplest_models.students import StudentMLP, StudentCnn
from simplest_models.teachers import Teacher
from utils.dataset.mnist import mnist_load_data
from utils.reproducible import set_seed, enforce_determinism


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

    train_dataloader, test_dataloader = mnist_load_data(g)
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
# Teacher acc: 98.86
# StudentCNN acc: 97.59|space acc: 98.22|continue acc: 98.07
# StudentMLP acc: 95.26|space acc: 95.42|continue acc: 96.08
