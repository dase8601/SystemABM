"""
explore_arcagi.py — Explore ARC-AGI-3 environment for JEPA compatibility.

Tests:
1. What does the observation space look like? (shape, type, format)
2. What actions are available?
3. Does DINOv2 produce discriminative features for game states?
4. Can our world model architecture work here?

Usage:
    pip install arc-agi
    python explore_arcagi.py
"""

import numpy as np

print("=" * 60)
print("ARC-AGI-3 Exploration — Testing JEPA compatibility")
print("=" * 60)

# ── Step 1: Load the environment ─────────────────────────────────────────────
print("\n[1] Loading ARC-AGI-3 environment...")

try:
    import arc_agi
    from arcengine import GameAction
    print(f"  arc_agi version: {arc_agi.__version__ if hasattr(arc_agi, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"  FAILED: {e}")
    print("  Run: pip install arc-agi")
    exit(1)

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode="terminal")

print("  Environment created: ls20")

# ── Step 2: Explore the environment API ──────────────────────────────────────
print("\n[2] Environment API exploration...")

# Check what methods/attributes the env has
env_attrs = [a for a in dir(env) if not a.startswith("_")]
print(f"  env attributes: {env_attrs}")

# Check GameAction
action_attrs = [a for a in dir(GameAction) if not a.startswith("_")]
print(f"  GameAction options: {action_attrs}")

# ── Step 3: Take actions and inspect observations ────────────────────────────
print("\n[3] Taking actions and inspecting observations...")

# Try to get initial observation
obs = None

# Try reset
if hasattr(env, "reset"):
    try:
        result = env.reset()
        print(f"  env.reset() returned: type={type(result)}")
        if isinstance(result, tuple):
            obs = result[0]
            print(f"    obs type: {type(obs)}, info type: {type(result[1]) if len(result) > 1 else 'N/A'}")
        else:
            obs = result
    except Exception as e:
        print(f"  env.reset() failed: {e}")

# Try step
try:
    result = env.step(GameAction.ACTION1)
    print(f"  env.step(ACTION1) returned: type={type(result)}")
    if isinstance(result, tuple):
        print(f"    tuple length: {len(result)}")
        for i, item in enumerate(result):
            t = type(item)
            if hasattr(item, "shape"):
                print(f"    [{i}] {t.__name__} shape={item.shape} dtype={item.dtype}")
            elif isinstance(item, (int, float, bool)):
                print(f"    [{i}] {t.__name__} = {item}")
            elif isinstance(item, dict):
                print(f"    [{i}] dict keys={list(item.keys())}")
            elif isinstance(item, np.ndarray):
                print(f"    [{i}] ndarray shape={item.shape}")
            else:
                print(f"    [{i}] {t.__name__}: {str(item)[:200]}")
        obs = result[0] if len(result) > 0 else None
    else:
        print(f"    result: {str(result)[:200]}")
except Exception as e:
    print(f"  env.step(ACTION1) failed: {e}")

# Try render
if hasattr(env, "render"):
    try:
        frame = env.render()
        if frame is not None:
            print(f"  env.render() returned: type={type(frame)}")
            if hasattr(frame, "shape"):
                print(f"    shape={frame.shape}, dtype={frame.dtype}")
        else:
            print("  env.render() returned None")
    except Exception as e:
        print(f"  env.render() failed: {e}")

# Try get_obs or observation
for method_name in ["get_obs", "observation", "get_observation", "get_state", "state"]:
    if hasattr(env, method_name):
        try:
            attr = getattr(env, method_name)
            if callable(attr):
                result = attr()
            else:
                result = attr
            print(f"  env.{method_name}: type={type(result)}")
            if hasattr(result, "shape"):
                print(f"    shape={result.shape}, dtype={result.dtype}")
            elif isinstance(result, dict):
                print(f"    keys={list(result.keys())}")
                for k, v in result.items():
                    if hasattr(v, "shape"):
                        print(f"      {k}: shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"      {k}: {type(v).__name__} = {str(v)[:100]}")
            else:
                print(f"    value: {str(result)[:200]}")
        except Exception as e:
            print(f"  env.{method_name} failed: {e}")

# ── Step 4: Inspect the observation deeply ───────────────────────────────────
print("\n[4] Deep observation inspection...")

if obs is not None:
    print(f"  obs type: {type(obs)}")
    if hasattr(obs, "shape"):
        print(f"  shape: {obs.shape}")
        print(f"  dtype: {obs.dtype}")
        print(f"  min: {obs.min()}, max: {obs.max()}")
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            print("  Looks like an RGB image!")
        elif len(obs.shape) == 2:
            print(f"  Looks like a 2D grid! size={obs.shape}")
            print(f"  Unique values: {np.unique(obs)}")
    elif isinstance(obs, dict):
        print(f"  dict keys: {list(obs.keys())}")
        for k, v in obs.items():
            if hasattr(v, "shape"):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}, min={v.min()}, max={v.max()}")
            else:
                print(f"    {k}: {type(v).__name__} = {str(v)[:100]}")
    elif isinstance(obs, (list, tuple)):
        print(f"  length: {len(obs)}")
        for i, item in enumerate(obs[:5]):
            print(f"    [{i}]: {type(item).__name__} = {str(item)[:100]}")
    else:
        print(f"  value: {str(obs)[:500]}")
else:
    print("  No observation captured — check API above for the right method")

# ── Step 5: Take several actions and collect states ──────────────────────────
print("\n[5] Collecting states across multiple actions...")

states = []
actions_taken = [
    GameAction.ACTION1,
    GameAction.ACTION2 if hasattr(GameAction, "ACTION2") else GameAction.ACTION1,
    GameAction.ACTION3 if hasattr(GameAction, "ACTION3") else GameAction.ACTION1,
    GameAction.ACTION4 if hasattr(GameAction, "ACTION4") else GameAction.ACTION1,
    GameAction.ACTION1,
]

for i, action in enumerate(actions_taken):
    try:
        result = env.step(action)
        if isinstance(result, tuple) and len(result) > 0:
            s = result[0]
            states.append(s)
            if hasattr(s, "shape"):
                print(f"  Step {i} ({action}): shape={s.shape}")
            else:
                print(f"  Step {i} ({action}): type={type(s).__name__}")
        else:
            print(f"  Step {i}: result={str(result)[:100]}")
    except Exception as e:
        print(f"  Step {i} failed: {e}")

# ── Step 6: Test DINOv2 feature discrimination ──────────────────────────────
print("\n[6] Testing DINOv2 feature discrimination on game states...")

if len(states) >= 2:
    try:
        import torch
        from PIL import Image
        from torchvision import transforms

        # Load DINOv2
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        dinov2.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dinov2 = dinov2.to(device)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def encode_state(state):
            """Convert game state to DINOv2 features."""
            if hasattr(state, "shape") and len(state.shape) == 3 and state.shape[2] == 3:
                # RGB image
                img = Image.fromarray(state.astype(np.uint8))
            elif hasattr(state, "shape") and len(state.shape) == 2:
                # 2D grid — render as colored image
                grid = state
                # Map grid values to colors
                colors = {
                    0: [0, 0, 0],       # black
                    1: [0, 0, 255],     # blue
                    2: [255, 0, 0],     # red
                    3: [0, 255, 0],     # green
                    4: [255, 255, 0],   # yellow
                    5: [128, 128, 128], # gray
                    6: [255, 0, 255],   # magenta
                    7: [255, 128, 0],   # orange
                    8: [0, 255, 255],   # cyan
                    9: [128, 0, 0],     # maroon
                    10: [255, 255, 255], # white
                }
                h, w = grid.shape
                img_arr = np.zeros((h, w, 3), dtype=np.uint8)
                for val, color in colors.items():
                    img_arr[grid == val] = color
                # Scale up for DINOv2
                img = Image.fromarray(img_arr).resize((224, 224), Image.NEAREST)
            else:
                print(f"  Can't encode state type: {type(state)}")
                return None

            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                features = dinov2.forward_features(tensor)
                cls_token = features["x_norm_clstoken"]  # (1, 768)
                cls_token = torch.nn.functional.normalize(cls_token, dim=-1)
            return cls_token

        # Encode all collected states
        features = []
        for i, s in enumerate(states):
            f = encode_state(s)
            if f is not None:
                features.append(f)
                print(f"  State {i} encoded: feature_dim={f.shape[-1]}")

        # Compute pairwise cosine similarities
        if len(features) >= 2:
            print("\n  Pairwise cosine similarities:")
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    cos_sim = torch.nn.functional.cosine_similarity(
                        features[i], features[j]
                    ).item()
                    print(f"    state_{i} vs state_{j}: cos_sim = {cos_sim:.4f}")

            # The key question: are different states discriminative?
            sims = []
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    sims.append(
                        torch.nn.functional.cosine_similarity(
                            features[i], features[j]
                        ).item()
                    )
            avg_sim = np.mean(sims)
            print(f"\n  Average cos_sim between states: {avg_sim:.4f}")
            if avg_sim < 0.85:
                print("  DISCRIMINATIVE — DINOv2 can distinguish game states!")
                print("  This architecture should work for ARC-AGI-3.")
            elif avg_sim < 0.95:
                print("  MODERATE — some discrimination, may work with tuning")
            else:
                print("  POOR — states too similar, may need different encoder")
                print("  (Same issue as MiniWorld — consider patch tokens or custom encoder)")

    except Exception as e:
        print(f"  DINOv2 test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  Not enough states collected — check API above")

# ── Step 7: Summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
Next steps based on results above:
- If obs is an RGB image → our architecture works directly
- If obs is a 2D grid → we render it to RGB and encode with DINOv2
- If cos_sim < 0.85 → DINOv2 features are discriminative, proceed
- If cos_sim > 0.95 → need different encoder (patch tokens or custom)
- Check action space to see if discrete → our MPC works as-is
""")

# Get scorecard
try:
    scorecard = arc.get_scorecard()
    print(f"Scorecard: {scorecard}")
except Exception as e:
    print(f"Scorecard failed: {e}")
