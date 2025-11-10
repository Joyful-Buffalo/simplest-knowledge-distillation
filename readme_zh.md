# MNIST 最小知识蒸馏示例

> 一个用于入门示例：在 MNIST 上进行 Teacher→Student 的知识蒸馏（KD）。

## 原理

## 主要公式
$$
\mathcal{L}_{\mathrm{KD}}
=(1-\alpha)\mathrm{CE}(\mathbf{z}_s, y)
+\alpha T^{2}\mathrm{KL}\left(
  \mathrm{softmax}\left(\frac{\mathbf{z}_t}{T}\right)\middle\|\mathrm{softmax}\left(\frac{\mathbf{z}_s}{T}\right)
\right)
$$



## 依赖
- `python>=3.9`
- `torch`
- `torchvision`

安装：
```bash
pip install -r requirements.txt
```

## 数据集
使用 `torchvision.datasets.MNIST`，首次运行会自动下载到 `dataset/`。

## 运行
将训练脚本为 `vanilla_kd_run.py`：
```bash
python vanilla_kd_run.py
```

## 关键可调项
- `temperature=2.0`
- `alpha=0.5`
- 优化器：Adam，`lr=1e-2`

## 预期结果（指示性）
Teacher acc: 98.90
StudentCNN acc: 97.86|space acc: 97.60|continue acc: 97.87
StudentMLP acc: 94.96|space acc: 94.43|continue acc: 95.08

> 数值会随硬件/版本/随机种子略有变化。

## 许可证与参考
- 用于学习与演示，无商业价值。
- 参考：Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network* (2015).
