#!/bin/bash
# setup_cloud.sh — One-shot setup for RunPod / Colab / Alpine (CUDA GPU)
#
# Usage:
#   bash setup_cloud.sh
#
# Then run:
#   DoorKey:  python abm_experiment.py --all --device auto --steps 800000
#   Crafter:  python abm_experiment.py --all --device auto --env crafter --steps 1000000
#   Habitat:  python abm_experiment.py --all --device auto --env habitat --steps 500000
#
# Habitat requires A100 (80GB) for V-JEPA 2.1 ViT-B inference.
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

echo "=== Installing Habitat dependencies ==="
# habitat-sim with headless rendering (no display needed on cloud)
pip install habitat-sim-headless 2>/dev/null || \
    echo "NOTE: habitat-sim-headless pip install failed. Try conda:"
    echo "  conda install habitat-sim headless -c conda-forge -c aihabitat"

# habitat-lab for task configs
pip install habitat-lab 2>/dev/null || \
    echo "NOTE: habitat-lab pip install failed. Try:"
    echo "  pip install git+https://github.com/facebookresearch/habitat-lab.git"

# V-JEPA 2.1 dependencies
pip install omegaconf timm -q 2>/dev/null || true

echo "=== Verifying ==="
python -c "
import torch, gymnasium, minigrid
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'VRAM: {mem:.1f} GB')
print(f'gymnasium: {gymnasium.__version__}')
print('minigrid: OK')

try:
    import crafter; print('crafter: OK')
except: print('crafter: not installed')

try:
    import habitat_sim; print('habitat-sim: OK')
except: print('habitat-sim: not installed (needed for --env habitat)')

print()
print('Run commands:')
print('  DoorKey: python abm_experiment.py --all --device auto --steps 800000')
print('  Crafter: python abm_experiment.py --all --device auto --env crafter --steps 1000000')
print('  Habitat: python abm_experiment.py --all --device auto --env habitat --steps 500000')
"
