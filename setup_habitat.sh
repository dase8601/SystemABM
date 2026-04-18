#!/bin/bash
# setup_habitat.sh — One-shot Habitat PointNav setup for RunPod (A100/H100)
#
# Creates a Python 3.9 conda env because habitat-sim only ships conda
# packages for Python 3.9. PyTorch cu121 supports A100 (sm_80) and
# H100 (sm_90) but NOT RTX 5090 (sm_120, needs cu128 + Python 3.10+).
#
# Usage:
#   bash setup_habitat.sh
#   conda activate habitat
#   python abm_experiment.py --condition autonomous --device auto --env habitat --steps 200000 --n-envs 4

set -e

echo "=== System dependencies ==="
apt-get update -qq && apt-get install -y -qq \
    libglu1-mesa-dev libgl1-mesa-dev freeglut3-dev xvfb git-lfs \
    > /dev/null 2>&1
git lfs install 2>/dev/null || true

echo "=== Virtual display ==="
pkill Xvfb 2>/dev/null || true
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1

echo "=== Creating conda env (Python 3.9 for habitat-sim) ==="
eval "$(/opt/conda/bin/conda shell.bash hook)"
conda create -n habitat python=3.9 -y -q

echo "=== Installing habitat-sim ==="
conda install -n habitat habitat-sim -c conda-forge -c aihabitat -y -q

echo "=== Activating env ==="
conda activate habitat

echo "=== Installing PyTorch (cu121 — supports A100/H100/V100) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q

echo "=== Installing dependencies ==="
pip install gymnasium numpy matplotlib minigrid omegaconf timm Pillow einops -q

echo "=== Downloading habitat test scenes ==="
python -c "
import habitat_sim
habitat_sim.utils.datasets_download.main([
    '--uids', 'habitat_test_scenes',
    '--data-path', 'data/'
])
" 2>/dev/null || echo "  Scene download failed — will auto-detect existing scenes"

echo "=== Verification ==="
python -c "
import torch, gymnasium, habitat_sim
print(f'Python:      {__import__(\"sys\").version.split()[0]}')
print(f'PyTorch:     {torch.__version__}')
print(f'CUDA:        {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
print(f'gymnasium:   {gymnasium.__version__}')
print(f'habitat_sim: OK')

from abm.habitat_env import make_habitat_env
env = make_habitat_env(seed=0, simple=True)
obs, _ = env.reset()
print(f'Habitat env: OK (RGB {obs[\"rgb\"].shape})')
goal = env.get_goal_obs()
print(f'Goal obs:    {\"OK\" if goal is not None else \"MISSING\"}')
env.close()

print()
print('Ready! Run:')
print('  conda activate habitat')
print('  python abm_experiment.py --condition autonomous --device auto --env habitat --steps 200000 --n-envs 4')
"
