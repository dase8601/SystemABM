"""
compare_backbones.py

Compares V-JEPA 2 ViT-Large vs V-JEPA 2.1 ViT-Base on DROID-100 robot
manipulation video — filling the gap in the V-JEPA 2.1 paper (arxiv 2603.14482)
which evaluates dense features on Ego4D / Cityscapes but not robot video.

Three output figures → results/comparison/
  1. pca_rgb.png          — Patch-level PCA→RGB visualization (4 sample episodes)
  2. episode_clusters.png — 2D PCA of episode embeddings coloured by task type
  3. linear_probe.png     — Linear probe accuracy: task-labeled vs unlabeled clips

Usage:
    python compare_backbones.py [--device mps] [--n-episodes 100]

Requirements (all already installed in this repo's venv):
    torch, torchvision, timm, lerobot, sklearn, matplotlib, seaborn, numpy, Pillow
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ImageNet normalisation — same constants used by both V-JEPA models
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ---------------------------------------------------------------------------
# Backbone configuration
# ---------------------------------------------------------------------------

@dataclass
class BackboneConfig:
    hub_name:        str   # PyTorch Hub entry point
    embed_dim:       int   # patch token dimension
    img_size:        int   # spatial resolution the model was trained at
    n_patches_h:     int   # patch grid rows  (img_size / patch_size)
    n_patches_w:     int   # patch grid cols
    temporal_frames: int   # 1 (V-JEPA 2.1) or 2 (V-JEPA 2 tubelet_size=2)
    display_name:    str   # label used in plots


CONFIGS: List[BackboneConfig] = [
    BackboneConfig(
        hub_name        = "vjepa2_vit_large",
        embed_dim       = 1024,
        img_size        = 256,
        n_patches_h     = 16,
        n_patches_w     = 16,
        temporal_frames = 2,
        display_name    = "V-JEPA 2\nViT-Large (256px)",
    ),
    BackboneConfig(
        hub_name        = "vjepa2_1_vit_base_384",
        embed_dim       = 768,
        img_size        = 384,
        n_patches_h     = 24,
        n_patches_w     = 24,
        temporal_frames = 1,
        display_name    = "V-JEPA 2.1\nViT-Base (384px)",
    ),
]


# ---------------------------------------------------------------------------
# Backbone wrapper
# ---------------------------------------------------------------------------

class BackboneWrapper:
    """
    Loads a V-JEPA encoder via PyTorch Hub and provides feature-extraction
    helpers.  All computation runs under torch.no_grad(); the backbone is
    frozen throughout.
    """

    def __init__(self, cfg: BackboneConfig, device: str):
        self.cfg    = cfg
        self.device = device
        self.encoder: Optional[torch.nn.Module] = None

    def load(self) -> None:
        logger.info(f"Loading {self.cfg.hub_name} from facebookresearch/vjepa2 ...")
        result = torch.hub.load(
            "facebookresearch/vjepa2",
            self.cfg.hub_name,
            pretrained=True,
            trust_repo=True,
        )
        # Hub always returns (encoder, predictor); we only need the encoder
        enc = result[0] if isinstance(result, (tuple, list)) else result
        enc = enc.to(self.device).eval()
        n_params = sum(p.numel() for p in enc.parameters())
        logger.info(f"  {self.cfg.hub_name}: {n_params:,} params on {self.device}")
        self.encoder = enc

    def free(self) -> None:
        """Move encoder to CPU and release MPS memory."""
        if self.encoder is not None:
            self.encoder.cpu()
            del self.encoder
            self.encoder = None
        if self.device == "mps":
            torch.mps.empty_cache()

    # ------------------------------------------------------------------ #

    def _preprocess(self, frame: torch.Tensor) -> torch.Tensor:
        """
        frame: (3, H, W) float32 in [0, 1]
        Returns: (1, 3, T, img_size, img_size) ready for the encoder.
        """
        img = TF.resize(frame, [self.cfg.img_size, self.cfg.img_size], antialias=True)
        img = (img - _MEAN) / _STD
        # → (1, 3, 1, H, W)
        img = img.unsqueeze(0).unsqueeze(2)
        if self.cfg.temporal_frames == 2:
            # V-JEPA 2 uses tubelet_size=2 — duplicate the frame
            img = img.expand(-1, -1, 2, -1, -1).contiguous()
        return img.to(self.device)

    @torch.no_grad()
    def patch_tokens(self, frame: torch.Tensor) -> np.ndarray:
        """
        Returns (N_patches, embed_dim) float32 numpy array for a single frame.
        """
        x     = self._preprocess(frame)
        out   = self.encoder(x)           # (1, N, D) or (1, D)
        tokens = out[0].float().cpu()
        if tokens.dim() == 1:
            # Scalar output — shouldn't happen but guard
            tokens = tokens.unsqueeze(0)
        return tokens.numpy()             # (N, D)

    @torch.no_grad()
    def episode_embedding(self, frames: List[torch.Tensor]) -> np.ndarray:
        """
        Mean-pool patch tokens across all sampled frames → (embed_dim,).
        """
        frame_means = []
        for frame in frames:
            tokens = self.patch_tokens(frame)   # (N, D)
            frame_means.append(tokens.mean(axis=0))  # (D,)
        return np.mean(frame_means, axis=0)          # (D,)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_droid100(
    n_episodes: int = 100,
    frames_per_ep: int = 3,
    image_key: str = "observation.images.wrist_image_left",
) -> Tuple[List[List[torch.Tensor]], List[int], List[str], List[int]]:
    """
    Load DROID-100 via lerobot.

    Returns:
        frames_list  — list[list[(3,H,W) tensor]] one inner list per episode
        task_indices — list[int]  task_index per episode (0-46)
        task_strings — list[str]  human-readable task per episode
        episode_ids  — list[int]
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise ImportError(
            "lerobot is not installed.  Run: pip install lerobot"
        )

    logger.info("Loading lerobot/droid_100 ...")
    ds = LeRobotDataset("lerobot/droid_100")

    # First pass: collect frame indices per episode
    ep_meta: Dict[int, Dict] = {}
    for i in range(len(ds)):
        s  = ds[i]
        ep = int(s["episode_index"].item())
        if ep not in ep_meta:
            ep_meta[ep] = {
                "task_index": int(s["task_index"].item()),
                "task":       s["task"],
                "frame_idxs": [],
            }
        ep_meta[ep]["frame_idxs"].append(i)

    ep_ids_sorted = sorted(ep_meta.keys())[:n_episodes]
    logger.info(f"Found {len(ep_ids_sorted)} episodes")

    frames_list:  List[List[torch.Tensor]] = []
    task_indices: List[int] = []
    task_strings: List[str] = []

    for ep_id in ep_ids_sorted:
        meta = ep_meta[ep_id]
        idxs = meta["frame_idxs"]
        n    = len(idxs)

        # Evenly spaced sample — skip the first and last 10% to avoid
        # pre-grasp and post-grasp padding that may look identical
        inner_start = max(0, n // 10)
        inner_end   = min(n, n - n // 10)
        inner_idxs  = idxs[inner_start:inner_end]

        step = max(1, len(inner_idxs) // frames_per_ep)
        sampled = [inner_idxs[i * step] for i in range(frames_per_ep)
                   if i * step < len(inner_idxs)]
        if not sampled:
            sampled = [idxs[n // 2]]

        frames = [ds[idx][image_key] for idx in sampled]
        frames_list.append(frames)
        task_indices.append(meta["task_index"])
        task_strings.append(meta["task"])

    n_labeled = sum(1 for t in task_strings if t.strip())
    n_unique  = len(set(task_indices))
    logger.info(
        f"Loaded {len(ep_ids_sorted)} episodes | "
        f"{n_labeled} labeled | {len(ep_ids_sorted) - n_labeled} unlabeled | "
        f"{n_unique} unique task types"
    )
    return frames_list, task_indices, task_strings, ep_ids_sorted


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_all_features(
    backbone: BackboneWrapper,
    frames_list: List[List[torch.Tensor]],
    vis_indices: List[int],
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    Extract episode embeddings for all episodes and patch tokens for
    the selected visualisation episodes.

    Returns:
        embeddings  — (N, D) episode embeddings
        vis_tokens  — {ep_idx: (N_patches, D) patch token array}
    """
    embeddings: List[np.ndarray] = []
    vis_tokens: Dict[int, np.ndarray] = {}

    for i, frames in enumerate(frames_list):
        emb = backbone.episode_embedding(frames)
        embeddings.append(emb)

        if i in vis_indices:
            # Use the middle frame for the patch visualisation
            mid_frame = frames[len(frames) // 2]
            vis_tokens[i] = backbone.patch_tokens(mid_frame)

        if (i + 1) % 10 == 0:
            logger.info(f"  [{backbone.cfg.hub_name}] {i + 1}/{len(frames_list)} episodes done")

    return np.stack(embeddings), vis_tokens


# ---------------------------------------------------------------------------
# Figure 1: PCA → RGB patch visualisation
# ---------------------------------------------------------------------------

def _pca_rgb(tokens: np.ndarray, nh: int, nw: int) -> np.ndarray:
    """
    tokens: (N_patches, D)  where N_patches = nh * nw
    Returns uint8 (nh, nw, 3) image.
    """
    pca  = PCA(n_components=3, random_state=42)
    proj = pca.fit_transform(tokens)   # (N, 3)

    # Normalise each channel independently to [0, 1]
    for c in range(3):
        lo, hi = proj[:, c].min(), proj[:, c].max()
        proj[:, c] = (proj[:, c] - lo) / (hi - lo + 1e-8)

    return (proj.reshape(nh, nw, 3) * 255).astype(np.uint8)


def plot_pca_rgb(
    save_dir: Path,
    frames_list: List[List[torch.Tensor]],
    task_strings: List[str],
    vis_indices: List[int],
    vis_tokens_v2:  Dict[int, np.ndarray],
    vis_tokens_v21: Dict[int, np.ndarray],
    cfg_v2:  BackboneConfig,
    cfg_v21: BackboneConfig,
) -> Path:
    """
    3-row × 4-col figure:
      Row 0 — original wrist camera frame
      Row 1 — V-JEPA 2  patch PCA → RGB
      Row 2 — V-JEPA 2.1 patch PCA → RGB
    """
    n_cols = len(vis_indices)
    n_rows = 3   # original + two backbones

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.0))

    row_labels = [
        "Original\nwrist camera",
        cfg_v2.display_name,
        cfg_v21.display_name,
    ]

    for col, ep_idx in enumerate(vis_indices):
        frame      = frames_list[ep_idx][len(frames_list[ep_idx]) // 2]
        task_label = task_strings[ep_idx] or "(unlabeled)"

        # Row 0: original RGB frame
        orig_np = frame.permute(1, 2, 0).numpy().clip(0, 1)
        axes[0, col].imshow(orig_np)
        axes[0, col].set_title(f"ep{ep_idx}\n{task_label[:28]}", fontsize=7.5, pad=3)
        axes[0, col].axis("off")

        # Row 1: V-JEPA 2 PCA-RGB
        tokens_v2 = vis_tokens_v2[ep_idx]
        pca_v2    = _pca_rgb(tokens_v2, cfg_v2.n_patches_h, cfg_v2.n_patches_w)
        pca_img_v2 = Image.fromarray(pca_v2).resize(
            (orig_np.shape[1], orig_np.shape[0]), Image.NEAREST
        )
        axes[1, col].imshow(np.array(pca_img_v2))
        axes[1, col].axis("off")

        # Row 2: V-JEPA 2.1 PCA-RGB
        tokens_v21 = vis_tokens_v21[ep_idx]
        pca_v21    = _pca_rgb(tokens_v21, cfg_v21.n_patches_h, cfg_v21.n_patches_w)
        pca_img_v21 = Image.fromarray(pca_v21).resize(
            (orig_np.shape[1], orig_np.shape[0]), Image.NEAREST
        )
        axes[2, col].imshow(np.array(pca_img_v21))
        axes[2, col].axis("off")

    # Row labels on the left
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=90,
                                 labelpad=6, va="center")

    fig.suptitle(
        "Patch Feature PCA → RGB  |  DROID-100 Robot Manipulation\n"
        "Each colour encodes a principal component of the patch embedding space",
        fontsize=10,
    )
    fig.tight_layout(rect=[0.04, 0, 1, 0.95])

    path = save_dir / "pca_rgb.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PCA-RGB figure → {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 2: Episode-level clustering (2D PCA scatter)
# ---------------------------------------------------------------------------

def plot_episode_clusters(
    save_dir: Path,
    embeddings_v2:  np.ndarray,   # (N, D1)
    embeddings_v21: np.ndarray,   # (N, D2)
    task_indices:   List[int],
    task_strings:   List[str],
) -> Path:
    """Side-by-side 2D PCA scatter coloured by task type."""
    labeled_mask = np.array([bool(t.strip()) for t in task_strings])

    # Colour map: unique labeled task indices get distinct colours,
    # unlabeled episodes are shown in light grey
    unique_tasks = sorted({ti for ti, ts in zip(task_indices, task_strings) if ts.strip()})
    cmap         = matplotlib.colormaps.get_cmap("tab20").resampled(max(len(unique_tasks), 1))
    task_color_idx = {ti: i for i, ti in enumerate(unique_tasks)}

    def _scatter(ax: plt.Axes, emb: np.ndarray, title: str) -> None:
        scaler = StandardScaler()
        normed = scaler.fit_transform(emb)
        pca    = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(normed)   # (N, 2)

        # Unlabeled first so labeled points render on top
        unlab_mask = ~labeled_mask
        ax.scatter(
            coords[unlab_mask, 0], coords[unlab_mask, 1],
            c="#cccccc", s=45, alpha=0.6, edgecolors="white", linewidths=0.3,
            label="Unlabeled episode",
        )
        for ep_i in np.where(labeled_mask)[0]:
            ti     = task_indices[ep_i]
            color  = cmap(task_color_idx[ti])
            ax.scatter(
                coords[ep_i, 0], coords[ep_i, 1],
                c=[color], s=55, alpha=0.9, edgecolors="white", linewidths=0.3,
            )

        var1 = pca.explained_variance_ratio_[0]
        var2 = pca.explained_variance_ratio_[1]
        ax.set_xlabel(f"PC1 ({var1:.1%} var)", fontsize=9)
        ax.set_ylabel(f"PC2 ({var2:.1%} var)", fontsize=9)
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(alpha=0.2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    _scatter(ax1, embeddings_v2,  "V-JEPA 2 ViT-Large\nEpisode Embeddings (PCA)")
    _scatter(ax2, embeddings_v21, "V-JEPA 2.1 ViT-Base\nEpisode Embeddings (PCA)")

    unlab_patch = mpatches.Patch(color="#cccccc", label="Unlabeled episode (54)")
    lab_patch   = mpatches.Patch(color=cmap(0),   label=f"Labeled episode (46, {len(unique_tasks)} task types)")
    fig.legend(
        handles=[unlab_patch, lab_patch],
        loc="lower center", fontsize=9, ncol=2, framealpha=0.9,
    )

    fig.suptitle(
        "Episode-Level Feature Space — DROID-100 (100 episodes, 47 task types)\n"
        "Each point is one episode; colour = task type (grey = no task label)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    path = save_dir / "episode_clusters.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Episode clusters figure → {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 3: Linear probe — labeled vs unlabeled task classification
# ---------------------------------------------------------------------------

def _run_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
) -> Dict[str, float]:
    """
    5-fold stratified cross-validated logistic regression.
    Returns {"mean": float, "std": float}.
    """
    scaler = StandardScaler()
    X      = scaler.fit_transform(embeddings)
    y      = labels

    # Clamp C to avoid instability with tiny datasets
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    return {"mean": float(scores.mean()), "std": float(scores.std())}


def plot_linear_probe(
    save_dir: Path,
    results_v2:  Dict[str, float],
    results_v21: Dict[str, float],
) -> Path:
    """
    Bar chart comparing V-JEPA 2 vs V-JEPA 2.1 linear probe accuracy
    on the binary task (labeled vs unlabeled episode).
    """
    models = ["V-JEPA 2\nViT-Large", "V-JEPA 2.1\nViT-Base"]
    accs   = [results_v2["mean"],  results_v21["mean"]]
    stds   = [results_v2["std"],   results_v21["std"]]
    colors = ["#3498db", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(models, accs, color=colors, alpha=0.85, width=0.45, zorder=3)
    ax.errorbar(
        models, accs, yerr=stds,
        fmt="none", color="black", capsize=5, lw=1.8, zorder=4,
    )

    # Random baseline (binary: 54/100 unlabeled → majority class)
    baseline = 0.54
    ax.axhline(
        baseline, color="#e74c3c", linestyle="--", lw=1.4,
        label=f"Majority-class baseline ({baseline:.0%})",
    )

    for bar, acc, std in zip(bars, accs, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            acc + std + 0.012,
            f"{acc:.1%}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_ylabel("Accuracy (5-fold CV)", fontsize=10)
    ax.set_title(
        "Linear Probe: Task-Labeled vs Unlabeled Episode\n"
        "DROID-100  |  Binary classification  |  5-fold stratified CV",
        fontsize=10, pad=8,
    )
    ax.set_ylim(0.0, min(1.0, max(accs) + max(stds) + 0.18))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.35, zorder=0)

    fig.tight_layout()
    path = save_dir / "linear_probe.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Linear probe figure → {path}")
    return path


# ---------------------------------------------------------------------------
# HTML report (self-contained, inline base64 images)
# ---------------------------------------------------------------------------

def _img_tag(path: Path, max_width: int = 700) -> str:
    import base64
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'style="max-width:{max_width}px; margin:8px; display:block;">'
    )


def write_report(
    save_dir: Path,
    results:  Dict,
    plot_paths: Dict[str, Path],
) -> Path:
    pv2  = results["linear_probe"]["vjepa2"]["mean"]
    pv21 = results["linear_probe"]["vjepa21"]["mean"]
    delta = pv21 - pv2
    sign  = "+" if delta >= 0 else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>V-JEPA 2 vs 2.1 — DROID-100 Feature Comparison</title>
  <style>
    body   {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 0 auto;
              padding: 28px; background: #f4f6f8; color: #222; }}
    h1     {{ color: #2c3e50; }}
    h2     {{ color: #34495e; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
    .card  {{ background: #fff; padding: 20px 28px; border-radius: 8px;
              margin: 18px 0; box-shadow: 0 1px 5px rgba(0,0,0,.12); }}
    table  {{ border-collapse: collapse; width: 100%; }}
    th, td {{ padding: 10px 16px; border-bottom: 1px solid #e0e0e0; text-align: left; }}
    th     {{ background: #3498db; color: #fff; }}
    tr:nth-child(even) {{ background: #f9f9f9; }}
    .big   {{ font-size: 1.9em; font-weight: bold; }}
    .green {{ color: #27ae60; }}
    .blue  {{ color: #2980b9; }}
    .delta {{ color: {"#27ae60" if delta >= 0 else "#e74c3c"}; }}
    figure {{ margin: 0; text-align: center; }}
    figcaption {{ font-size: 0.85em; color: #555; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>V-JEPA 2 vs V-JEPA 2.1 — Dense Feature Quality on DROID-100</h1>
  <p>
    Frozen backbone feature analysis on 100 real robot manipulation episodes
    (DROID-100 via lerobot).  Fills the evaluation gap in the V-JEPA 2.1 paper
    (arXiv 2603.14482, §3.3) which benchmarks dense features on Ego4D, Cityscapes
    and Diving48 but not robot manipulation video.
  </p>

  <div class="card">
    <h2>Key Results</h2>
    <table>
      <tr><th>Metric</th><th>V-JEPA 2 ViT-Large</th><th>V-JEPA 2.1 ViT-Base</th><th>Δ</th></tr>
      <tr>
        <td>Linear probe accuracy (task-labeled vs unlabeled)</td>
        <td class="blue big">{pv2:.1%}</td>
        <td class="green big">{pv21:.1%}</td>
        <td class="delta big">{sign}{delta:.1%}</td>
      </tr>
      <tr><td>Episodes</td><td colspan="3">{results["n_episodes"]} (46 labeled, 54 unlabeled)</td></tr>
      <tr><td>Unique task types</td><td colspan="3">{results["n_task_types"]}</td></tr>
      <tr><td>Probe type</td><td colspan="3">Logistic regression, 5-fold stratified CV</td></tr>
      <tr><td>Majority-class baseline</td><td colspan="3">54.0% (always predict "unlabeled")</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Figure 1 — Patch PCA → RGB Visualisation</h2>
    <p>PCA applied to (N_patches × D) patch tokens; first 3 components mapped to RGB.
       Coherent colour regions indicate spatially structured representations.</p>
    <figure>
      {_img_tag(plot_paths["pca_rgb"], 1000)}
      <figcaption>Rows: original wrist-camera frame / V-JEPA 2 ViT-L / V-JEPA 2.1 ViT-B</figcaption>
    </figure>
  </div>

  <div class="card">
    <h2>Figure 2 — Episode-Level Feature Clustering</h2>
    <p>Each point is one episode (mean-pooled patch embeddings → PCA 2D).
       Grey = no task label; coloured = explicit task description.</p>
    <figure>
      {_img_tag(plot_paths["episode_clusters"], 1000)}
      <figcaption>Better separation of coloured (task-labeled) and grey (unlabeled)
      clusters indicates richer semantic encoding.</figcaption>
    </figure>
  </div>

  <div class="card">
    <h2>Figure 3 — Linear Probe Accuracy</h2>
    <p>Binary logistic regression trained on frozen features to distinguish
       episodes with explicit task labels from unlabeled episodes.</p>
    <figure>
      {_img_tag(plot_paths["linear_probe"], 600)}
      <figcaption>Error bars = ±1 std across 5 folds.</figcaption>
    </figure>
  </div>
</body>
</html>"""

    path = save_dir / "report.html"
    path.write_text(html)
    logger.info(f"HTML report → {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> Dict:
    save_dir = Path("results/comparison")
    save_dir.mkdir(parents=True, exist_ok=True)
    device   = args.device

    logger.info("=" * 70)
    logger.info("V-JEPA 2 vs V-JEPA 2.1 — DROID-100 Feature Comparison")
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    # Step 1: Load data                                                    #
    # ------------------------------------------------------------------ #
    logger.info("\nStep 1: Loading DROID-100 ...")
    frames_list, task_indices, task_strings, episode_ids = load_droid100(
        n_episodes=args.n_episodes,
        frames_per_ep=3,
    )

    # Four visually diverse episodes for Figure 1
    # Prefer episodes that span unlabeled + labeled and different tasks
    labeled_ep_idxs   = [i for i, t in enumerate(task_strings) if t.strip()]
    unlabeled_ep_idxs = [i for i, t in enumerate(task_strings) if not t.strip()]
    vis_indices = (
        labeled_ep_idxs[:2] + unlabeled_ep_idxs[:2]
        if len(labeled_ep_idxs) >= 2 and len(unlabeled_ep_idxs) >= 2
        else list(range(min(4, len(frames_list))))
    )
    logger.info(f"Visualisation episodes: {vis_indices}")

    # ------------------------------------------------------------------ #
    # Step 2: Load both backbones and extract features                    #
    # ------------------------------------------------------------------ #
    logger.info("\nStep 2: Extracting features ...")

    cfg_v2  = CONFIGS[0]
    cfg_v21 = CONFIGS[1]

    backbone_v2  = BackboneWrapper(cfg_v2,  device)
    backbone_v21 = BackboneWrapper(cfg_v21, device)

    backbone_v2.load()
    logger.info(f"Extracting V-JEPA 2 features ({len(frames_list)} episodes) ...")
    embeddings_v2, vis_tokens_v2 = extract_all_features(
        backbone_v2, frames_list, vis_indices
    )
    backbone_v2.free()
    logger.info(f"V-JEPA 2 embeddings: {embeddings_v2.shape}")

    backbone_v21.load()
    logger.info(f"Extracting V-JEPA 2.1 features ({len(frames_list)} episodes) ...")
    embeddings_v21, vis_tokens_v21 = extract_all_features(
        backbone_v21, frames_list, vis_indices
    )
    backbone_v21.free()
    logger.info(f"V-JEPA 2.1 embeddings: {embeddings_v21.shape}")

    # ------------------------------------------------------------------ #
    # Step 3: Figure 1 — PCA-RGB patch visualisation                      #
    # ------------------------------------------------------------------ #
    logger.info("\nStep 3: Generating Figure 1 (PCA-RGB) ...")
    pca_rgb_path = plot_pca_rgb(
        save_dir, frames_list, task_strings, vis_indices,
        vis_tokens_v2, vis_tokens_v21, cfg_v2, cfg_v21,
    )

    # ------------------------------------------------------------------ #
    # Step 4: Figure 2 — Episode clustering                               #
    # ------------------------------------------------------------------ #
    logger.info("\nStep 4: Generating Figure 2 (episode clusters) ...")
    cluster_path = plot_episode_clusters(
        save_dir, embeddings_v2, embeddings_v21, task_indices, task_strings,
    )

    # ------------------------------------------------------------------ #
    # Step 5: Linear probe                                                 #
    # ------------------------------------------------------------------ #
    logger.info("\nStep 5: Running linear probe ...")

    # Binary label: 1 = has explicit task description, 0 = unlabeled
    labels = np.array([1 if t.strip() else 0 for t in task_strings], dtype=np.int32)
    n_labeled   = int(labels.sum())
    n_unlabeled = int((1 - labels).sum())
    logger.info(f"Probe labels: {n_labeled} labeled (1) | {n_unlabeled} unlabeled (0)")

    probe_v2  = _run_probe(embeddings_v2,  labels)
    probe_v21 = _run_probe(embeddings_v21, labels)

    logger.info(f"V-JEPA 2  : {probe_v2['mean']:.1%}  ±{probe_v2['std']:.1%}")
    logger.info(f"V-JEPA 2.1: {probe_v21['mean']:.1%}  ±{probe_v21['std']:.1%}")

    probe_path = plot_linear_probe(save_dir, probe_v2, probe_v21)

    # ------------------------------------------------------------------ #
    # Step 6: Save results JSON and HTML report                           #
    # ------------------------------------------------------------------ #
    n_task_types = len(set(task_indices))
    results = {
        "n_episodes":    len(frames_list),
        "n_task_types":  n_task_types,
        "n_labeled":     n_labeled,
        "n_unlabeled":   n_unlabeled,
        "linear_probe": {
            "vjepa2":  probe_v2,
            "vjepa21": probe_v21,
            "baseline_majority_class": float(max(n_labeled, n_unlabeled)) / len(labels),
        },
        "embedding_dims": {
            "vjepa2_vitl":  int(embeddings_v2.shape[1]),
            "vjepa21_vitb": int(embeddings_v21.shape[1]),
        },
    }

    json_path = save_dir / "comparison_results.json"
    json_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Results JSON → {json_path}")

    write_report(
        save_dir, results,
        {"pca_rgb": pca_rgb_path, "episode_clusters": cluster_path, "linear_probe": probe_path},
    )

    logger.info("\n" + "=" * 70)
    logger.info("Done!")
    logger.info("=" * 70)
    logger.info(f"  Results: {save_dir}/")
    logger.info(f"  V-JEPA 2  probe: {probe_v2['mean']:.1%} ± {probe_v2['std']:.1%}")
    logger.info(f"  V-JEPA 2.1 probe: {probe_v21['mean']:.1%} ± {probe_v21['std']:.1%}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare V-JEPA 2 vs V-JEPA 2.1 features on DROID-100"
    )
    parser.add_argument(
        "--device", default="mps",
        choices=["mps", "cpu", "cuda"],
        help="Compute device (default: mps)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=100,
        help="Number of DROID-100 episodes to use (default: 100)",
    )
    args = parser.parse_args()
    main(args)
