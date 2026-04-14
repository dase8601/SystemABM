"""
abm/vjepa_encoder.py — DINOv3 ViT-B/16 frozen encoder for the A-B-M loop.

DINOv3 is Meta's strongest universal vision backbone:
- 1.7B image training set (LVD-1689M), 6x larger run than DINOv2
- First SSL model to outperform weakly-supervised models on dense probing
- ViT-B/16: 768-dim CLS token, identical output shape to V-JEPA ViT-B

DINOv3 is a JEPA-class encoder: self-supervised, no pixel reconstruction,
trained on massive passive observation. Yann LeCun (AI Alliance 2026):
"[joint embedding methods are] probably the best image encoders we have."

Loaded via torch.hub from GitHub (public) + weights from dl.fbaipublicfiles.com
(public) — no HuggingFace authentication required.

Output: 768-dim L2-normalized CLS token — zero changes to VJEPAPredictor.

Usage:
    encoder = VJEPAEncoder(device="cuda")
    z = encoder.encode(obs_dict)         # (B, 768)
    z = encoder.encode_single(obs_dict)  # (1, 768)

Input:  (B, H, W, 3) uint8 RGB or (B, 3, H, W) float32 [0, 1]
Output: (B, 768) float32 — DINOv3 CLS token, L2 normalized
"""

import torch
import torch.nn.functional as F
import numpy as np


class VJEPAEncoder:
    """
    Frozen DINOv3 ViT-B/16 encoder (drop-in replacement for V-JEPA / DINOv2).

    Loads DINOv3 ViT-B/16 via torch.hub (GitHub source, public weights).
    Input images are resized to 224x224 and normalized.
    Output is the L2-normalized CLS token: (B, 768).
    """

    # ImageNet normalization (standard for DINOv2/DINOv3/V-JEPA)
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, device: str = "cuda", img_size: int = 224):
        self.device   = device
        self.img_size = img_size

        print("Loading DINOv3 ViT-B/16 (JEPA-class frozen encoder)...")
        # GitHub repo is public; weights download from dl.fbaipublicfiles.com (public)
        self.encoder = torch.hub.load(
            "facebookresearch/dinov3",
            "dinov3_vitb16",
            pretrained=True,
        )
        self.encoder = self.encoder.to(device).eval()

        for p in self.encoder.parameters():
            p.requires_grad_(False)

        self.feature_dim = 768  # ViT-B CLS token dim

        self._mean = self.MEAN.to(device)
        self._std  = self.STD.to(device)

        # Sanity check: different images must produce different features
        with torch.no_grad():
            black  = torch.zeros(1, 3, img_size, img_size, device=device)
            white  = torch.ones(1, 3, img_size, img_size, device=device)
            noise1 = torch.randn(1, 3, img_size, img_size, device=device).clamp(0, 1)
            noise2 = torch.randn(1, 3, img_size, img_size, device=device).clamp(0, 1)
            all_in = torch.cat([black, white, noise1, noise2])
            all_in = (all_in - self._mean) / self._std
            feat   = self.encoder.forward_features(all_in)["x_norm_clstoken"]
            feat   = F.normalize(feat, p=2, dim=-1)
            bw_sim = F.cosine_similarity(feat[0:1], feat[1:2]).item()
            n_sim  = F.cosine_similarity(feat[2:3], feat[3:4]).item()
            print(f"  DINOv3 loaded — feature_dim={self.feature_dim}")
            print(f"    cos_sim(black,white)={bw_sim:.4f}, cos_sim(noise1,noise2)={n_sim:.4f}")
            if bw_sim > 0.70:
                print("  WARNING: cos_sim(black,white) > 0.70 — features may not be discriminative!")
            else:
                print("  Features are discriminative — good to train.")

    def _preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Resize to 224x224 and normalize.
        Input:  (B, 3, H, W) float32 [0, 1]
        Output: (B, 3, 224, 224) float32 normalized
        """
        if imgs.shape[-2:] != (self.img_size, self.img_size):
            imgs = F.interpolate(imgs, size=(self.img_size, self.img_size),
                                 mode="bilinear", align_corners=False)
        return (imgs - self._mean) / self._std

    def _obs_to_tensor(self, obs_dict: dict) -> torch.Tensor:
        """Convert observation dict to (B, 3, H, W) float32 tensor."""
        imgs = obs_dict.get("rgb", obs_dict.get("image"))
        if isinstance(imgs, np.ndarray):
            if imgs.ndim == 3:
                imgs = imgs[np.newaxis]
            x = torch.from_numpy(imgs.astype(np.float32) / 255.0)
            x = x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
        else:
            x = imgs
        return x.to(self.device)

    @torch.no_grad()
    def encode(self, obs_dict: dict) -> torch.Tensor:
        """
        Encode a batch of observations.
        obs_dict: {"rgb": (B, H, W, 3) uint8}
        Returns: (B, 768) L2-normalized CLS token
        """
        x   = self._obs_to_tensor(obs_dict)
        x   = self._preprocess(x)
        out = self.encoder.forward_features(x)["x_norm_clstoken"]
        return F.normalize(out, p=2, dim=-1)

    @torch.no_grad()
    def encode_single(self, obs_dict: dict) -> torch.Tensor:
        """Encode a single observation. Returns (1, 768)."""
        return self.encode(obs_dict)

    @torch.no_grad()
    def encode_tensor(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Encode pre-processed image tensors directly.
        imgs: (B, 3, H, W) float32 [0, 1]
        Returns: (B, 768)
        """
        x   = self._preprocess(imgs)
        out = self.encoder.forward_features(x)["x_norm_clstoken"]
        return F.normalize(out, p=2, dim=-1)
