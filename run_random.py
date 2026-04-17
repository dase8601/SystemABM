"""
run_random.py — Run only the RANDOM condition in isolation.

Runs in its own process so memory is fully released after completion.
Results saved to results/miniworld/metrics_random.json

Usage:
    python run_random.py
    python run_random.py --steps 200000 --device auto
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int, default=200_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=16)
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    logger.info(f"Device: {args.device}")
    logger.info("Running condition: RANDOM (isolated process)")

    from abm.loop import run_abm_loop

    result = run_abm_loop(
        condition = "random",
        device    = args.device,
        max_steps = args.steps,
        seed      = args.seed,
        n_envs    = args.n_envs,
        env_type  = "miniworld",
    )

    save_dir = Path("results/miniworld")
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "metrics_random.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    peak = max(result["success_rate"]) if result["success_rate"] else 0
    logger.info(f"Done — peak={peak:.1%} | saved → {json_path}")


if __name__ == "__main__":
    main()
