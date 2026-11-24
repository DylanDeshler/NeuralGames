"""Loss / conditioning masks for training on the packed WebDataset samples.

Each sample's JSON carries, in output-pixel coordinates (width x height = the clip's aspect bucket):
  pad = [px, py], content_wh = [w, h]    letterbox geometry (game content rectangle)
  inpaint_boxes = [[x, y, w, h], ...]     regions inpainted in stage 03 (controller overlay, webcams, ...),
                                          already dilated by --mask-dilate px

Recommended use (see README "Overlay masking"):
  * the model never sees overlay pixels: they were removed before encoding (stage 03 inpainting);
  * don't supervise inpainted pixels or letterbox bars: multiply the per-element loss by `valid`;
  * optionally feed `1 - valid` (latent resolution) as an extra input channel, add random fake boxes to
    ~10-20% of samples with `augment_fake_boxes`, and pass an all-valid mask at inference.

numpy-only, and works with any framework (torch.from_numpy on the outputs).
"""
from __future__ import annotations

import numpy as np


def pixel_valid_mask(info: dict, mask_letterbox: bool = True, mask_inpainted: bool = True,
                     extra_dilate: int = 0) -> np.ndarray:
    """(H, W) float32, 1 = supervise this pixel, 0 = ignore (inpainted overlay / pad bars in 'fit' clips)."""
    W, H = int(info["width"]), int(info["height"])
    valid = np.ones((H, W), np.float32)
    if mask_letterbox:
        px, py = info.get("pad", [0, 0])
        cw, ch = info.get("content_wh", [W, H])
        valid[:] = 0
        valid[py:py + ch, px:px + cw] = 1
    if mask_inpainted:
        for x, y, w, h in info.get("inpaint_boxes") or []:
            x0, y0 = max(0, x - extra_dilate), max(0, y - extra_dilate)
            x1, y1 = min(W, x + w + extra_dilate), min(H, y + h + extra_dilate)
            valid[y0:y1, x0:x1] = 0
    return valid


def grid_valid_mask(valid_px: np.ndarray, stride: int, halo: int = 0) -> np.ndarray:
    """Downsample a pixel mask to a grid of `stride`-px cells (latent cells, or patches for pixel-space
    models like JiT: stride = patch size gives a per-token mask, e.g. to drop fully-masked tokens).

    A latent cell is valid only if ALL of its pixels are valid (conservative: any touched cell is
    masked). `halo` additionally masks that many neighbouring cells, which covers the VAE encoder's
    receptive-field bleed if you ever encode frames that still contain overlay pixels.
    """
    H, W = valid_px.shape
    gh, gw = H // stride, W // stride
    cells = valid_px[: gh * stride, : gw * stride].reshape(gh, stride, gw, stride).min(axis=(1, 3))
    if halo > 0:
        inv = cells < 0.5
        grown = inv.copy()
        for dy in range(-halo, halo + 1):
            for dx in range(-halo, halo + 1):
                grown |= np.roll(np.roll(inv, dy, 0), dx, 1) & _shift_ok(gh, gw, dy, dx)
        cells = (~grown).astype(np.float32)
    return cells.astype(np.float32)


def _shift_ok(gh: int, gw: int, dy: int, dx: int) -> np.ndarray:
    """Mask out wrap-around from np.roll."""
    ok = np.ones((gh, gw), bool)
    if dy > 0:
        ok[:dy] = False
    elif dy < 0:
        ok[dy:] = False
    if dx > 0:
        ok[:, :dx] = False
    elif dx < 0:
        ok[:, dx:] = False
    return ok


def augment_fake_boxes(valid: np.ndarray, rng: np.random.Generator, p: float = 0.15, max_boxes: int = 2,
                       min_frac: float = 0.03, max_frac: float = 0.12) -> np.ndarray:
    """With prob p, zero out 1..max_boxes random rectangles (each min_frac..max_frac of the area).

    Use on the *conditioning* mask channel (and blank the same region in the context frames) so the model
    learns "masked = unknown" in general rather than "masked = where overlays live".
    """
    if rng.random() >= p:
        return valid
    v = valid.copy()
    H, W = v.shape
    for _ in range(rng.integers(1, max_boxes + 1)):
        area = rng.uniform(min_frac, max_frac) * H * W
        ar = rng.uniform(0.6, 1.6)
        h = int(np.clip(np.sqrt(area / ar), 2, H))
        w = int(np.clip(area / max(h, 1), 2, W))
        y, x = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
        v[y:y + h, x:x + w] = 0
    return v


def fit_to_canvas(frames: np.ndarray, valid: np.ndarray, W: int, H: int, max_crop: float = 0.05):
    """Map a clip stored at its own bucket onto a training canvas chosen later (e.g. one 16:9 canvas).

    frames: (T, h, w, 3) uint8; valid: (h, w) loss mask from pixel_valid_mask(). Keeps the aspect ratio:
    if it is within `max_crop` of the canvas, scale to cover and centre-crop; otherwise scale to fit and pad
    with black. Returns (frames (T, H, W, 3), valid (H, W)) with padded pixels marked invalid."""
    import cv2
    T, h, w = frames.shape[:3]
    a, t = w / h, W / H
    cover = 1 - min(a / t, t / a) <= max_crop
    s = max(W / w, H / h) if cover else min(W / w, H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    rf = np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR) for f in frames])
    rv = cv2.resize(valid.astype(np.float32), (nw, nh), interpolation=cv2.INTER_NEAREST)
    out_f = np.zeros((T, H, W, 3), np.uint8)
    out_v = np.zeros((H, W), np.float32)
    if cover:  # centre-crop the small excess
        x0, y0 = (nw - W) // 2, (nh - H) // 2
        out_f[:] = rf[:, y0:y0 + H, x0:x0 + W]
        out_v[:] = rv[y0:y0 + H, x0:x0 + W]
    else:      # pad (letter/pillarbox); padding is excluded from the loss
        x0, y0 = (W - nw) // 2, (H - nh) // 2
        out_f[:, y0:y0 + nh, x0:x0 + nw] = rf
        out_v[y0:y0 + nh, x0:x0 + nw] = rv
    return out_f, out_v


# ----------------------------------------------------------------------------- action alignment (training side)
# The NitroGen action labels are read from the on-screen controller overlay, which often leads or lags the
# rendered game frame by 1-3 native (60 fps) frames (capture-card / OBS latency, constant per recording setup).
# That is below one 15 fps output frame, so it can only be corrected from the native-rate actions stored in
# each sample (`actions_native.npy`, (T * frame_stride, 21) float16). Two complementary fixes:
#   1. realign_actions(native, stride, shift): re-pool with a shift of `shift` native frames;
#   2. action_history(acts, k): condition step t on [a_{t-k+1}, ..., a_t] instead of a_t alone, so the model
#      can absorb a small residual lag (k = 2-3 output frames covers +-1-3 native frames).

def pool_native(native: np.ndarray, stride: int) -> np.ndarray:
    """Same pooling as stage 03: sticks (first 4 dims) averaged, buttons (17) OR'd over each output frame."""
    T = len(native) // stride
    g = native[: T * stride].astype(np.float32).reshape(T, stride, native.shape[1])
    return np.concatenate([g[:, :, :4].mean(1), g[:, :, 4:].max(1)], axis=1)


def realign_actions(native: np.ndarray, stride: int, shift: int = 0) -> np.ndarray:
    """(T, 21) actions re-pooled after shifting by `shift` native frames.

    shift > 0 means the overlay LAGS the picture (an input shows up on the overlay `shift` frames after its
    effect is visible): actions are moved earlier. Edges are padded by repeating the first/last frame, so the
    output has the same length as the stored actions."""
    a = native.astype(np.float32)
    if shift > 0:
        a = np.concatenate([a[shift:], np.repeat(a[-1:], shift, 0)])
    elif shift < 0:
        a = np.concatenate([np.repeat(a[:1], -shift, 0), a[:shift]])
    return pool_native(a, stride)


def action_history(acts: np.ndarray, k: int = 3) -> np.ndarray:
    """(T, k * D): row t = [a_{t-k+1}, ..., a_t] (zeros before the clip start = "no input")."""
    T, D = acts.shape
    pad = np.concatenate([np.zeros((k - 1, D), acts.dtype), acts])
    return np.concatenate([pad[i:i + T] for i in range(k)], axis=1)
