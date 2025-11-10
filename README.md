[中文说明](readme_zh.md)
# Minimal Knowledge Distillation on MNIST

A tiny educational example of Teacher→Student knowledge distillation (KD) on MNIST.

## Principle

$$
\mathcal{L}_{\mathrm{KD}}
=(1-\alpha)\mathrm{CE}(\mathbf{z}_s, y)
+\alpha T^{2}\mathrm{KL}\left(
  \mathrm{softmax}\left(\frac{\mathbf{z}_t}{T}\right)\middle\|\mathrm{softmax}\left(\frac{\mathbf{z}_s}{T}\right)
\right)
$$




## Requirements
- `python>=3.9`
- `torch`
- `torchvision`

Install:
```bash
pip install -r requirements.txt
```

## Dataset
Uses `torchvision.datasets.MNIST`. It will be downloaded automatically to `dataset/` on first run.

## Run
training script as `vanilla_kd_run.py` and execute:
```bash
python vanilla_kd_run.py
```

## Key Tunables
- `temperature = 2.0`
- `alpha = 0.5`
- Optimizer: Adam with `lr = 1e-2`

## Expected Results
Teacher acc: 98.90
StudentCNN acc: 97.86|space acc: 98.06|continue acc: 98.26
StudentMLP acc: 94.96|space acc: 95.51|continue acc: 95.81

> Numbers vary with hardware, versions, and seeds.

## License & Reference
- For learning/demo purposes only.
- Reference: Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network* (2015).
