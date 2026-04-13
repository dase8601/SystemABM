#!/bin/bash
# setup_cloud.sh — One-shot setup for RunPod / Colab / Alpine (CUDA GPU)
#
# Usage:
#   bash setup_cloud.sh
#
# Then run:
#   DoorKey:    python abm_experiment.py --all --device auto --steps 800000
#   Crafter:    python abm_experiment.py --all --device auto --env crafter --steps 1000000
#   MiniWorld:  python abm_experiment.py --all --device auto --env miniworld --steps 500000
#
# MiniWorld + V-JEPA 2.1 benefits from A100 but works on any CUDA GPU.
#
# On Google Colab, mount Drive first and cd to your repo:
#   from google.colab import drive
#   drive.mount('/content/drive')
#   %cd /content/drive/MyDrive/jepa

set -e

echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

echo "=== Installing core dependencies ==="
pip install minigrid gymnasium numpy matplotlib crafter -q

echo "=== Installing MiniWorld + V-JEPA dependencies ==="
pip install miniworld omegaconf timm Pillow -q

echo "=== Verifying ==="
python -c "
import torch, gymnasium, minigrid
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM: {mem:.1f} GB')
print(f'gymnasium: {gymnasium.__version__}')
print('minigrid: OK')

try:
    import crafter; print('crafter: OK')
except: print('crafter: not installed')

try:
    import miniworld; print('miniworld: OK')
except: print('miniworld: FAILED')

print()
print('Run commands:')
print('  DoorKey:   python abm_experiment.py --all --device auto --steps 800000')
print('  Crafter:   python abm_experiment.py --all --device auto --env crafter --steps 1000000')
print('  MiniWorld: python abm_experiment.py --all --device auto --env miniworld --steps 500000')
"
