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

# habitat-sim: try conda first (most reliable), fall back to pip
if command -v conda &>/dev/null; then
    echo "Installing habitat-sim via conda..."
    conda install habitat-sim headless -c conda-forge -c aihabitat -y
else
    echo "conda not found — installing miniconda for habitat-sim..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/conda
    export PATH="/opt/conda/bin:$PATH"
    # Use the existing python's site-packages so torch/gymnasium are visible
    conda install habitat-sim headless -c conda-forge -c aihabitat -y
    # Symlink habitat_sim into the system python so our scripts can import it
    CONDA_SITE=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "/opt/conda/lib/python3.11/site-packages")
    SYS_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
    if [ -d "$CONDA_SITE/habitat_sim" ]; then
        ln -sf "$CONDA_SITE/habitat_sim" "$SYS_SITE/habitat_sim"
        echo "Symlinked habitat_sim into system python"
    fi
    rm /tmp/miniconda.sh
fi

# habitat-lab for task configs
pip install habitat-lab -q 2>/dev/null || \
    pip install git+https://github.com/facebookresearch/habitat-lab.git -q 2>/dev/null || \
    echo "NOTE: habitat-lab install failed — habitat env may not work with full config mode"

# V-JEPA 2.1 dependencies
pip install omegaconf timm -q 2>/dev/null || true

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
    import habitat_sim; print('habitat-sim: OK')
except: print('habitat-sim: NOT installed (needed for --env habitat)')

print()
print('Run commands:')
print('  DoorKey: python abm_experiment.py --all --device auto --steps 800000')
print('  Crafter: python abm_experiment.py --all --device auto --env crafter --steps 1000000')
print('  Habitat: python abm_experiment.py --all --device auto --env habitat --steps 500000')
"
