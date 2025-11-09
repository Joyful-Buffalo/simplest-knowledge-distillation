# Minimal Knowledge Distillation on MNIST

A tiny educational example of Teacher→Student knowledge distillation (KD) on MNIST.

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
training script as `knowledge_distillation_run.py` and execute:
```bash
python knowledge_distillation_run.py
```

## Key Tunables
- `temperature = 2.0`
- `alpha = 0.5`
- Optimizer: Adam with `lr = 1e-2`

## Expected Results (indicative)
| Model                    | Test Acc (%) |
|-------------------------|-------------:|
| Teacher (CNN)           |       ~98–99 |
| Student CNN (baseline)  |       ~97–98 |
| Student CNN (KD)        |       ~97–98 |
| Student MLP (baseline)  |       ~94–95 |
| Student MLP (KD)        |       ~94–95 |

> Numbers vary with hardware, versions, and seeds.

## License & Reference
- For learning/demo purposes only.
- Reference: Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network* (2015).
