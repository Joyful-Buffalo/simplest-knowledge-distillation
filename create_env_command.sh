conda remove -n distill --all -y || true
conda create -n distill python=3.13 -y
conda activate distill

python -m pip install -U pip
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
