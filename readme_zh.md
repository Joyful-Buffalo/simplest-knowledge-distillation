# MNIST 最小知识蒸馏示例

> 一个用于教学/入门的简洁示例：在 MNIST 上进行 Teacher→Student 的知识蒸馏（KD）。

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
将训练脚本为 `knowledge_distillation_run.py`：
```bash
python knowledge_distillation_run.py
```

## 关键可调项
- `temperature=2.0`
- `alpha=0.5`
- 优化器：Adam，`lr=1e-2`

## 预期结果（指示性）
| 模型                     | 测试准确率(%) |
|------------------------|---------:|
| Teacher (CNN)         |   ~98–99 |
| Student CNN (baseline)|   ~97–98 |
| Student CNN (KD)      |   ~97–98 |
| Student MLP (baseline)|   ~94–95 |
| Student MLP (KD)      |   ~94–95 |

> 数值会随硬件/版本/随机种子略有变化。

## 许可证与参考
- 用于学习与演示，无商业价值说明。
- 参考：Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network* (2015).