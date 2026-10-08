"""Overlay GT (green), predictions (red), missed GT (yellow fill) on an ex-vivo crop with most misses."""
import os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cellmatch import rle_to_labels, read_image  # noqa: E402
from common import load_truth, ROOT  # noqa: E402

sid = sys.argv[1]
mod = sys.argv[2] if len(sys.argv) > 2 else "exvivo"
size = int(sys.argv[3]) if len(sys.argv) > 3 else 300
truth = load_truth()
data = np.load(os.path.join(os.path.dirname(__file__), "data", "heldout_labels.npz"))
img = read_image(os.path.join(ROOT, "training", *sid.split("__"), f"{mod}.tif")).astype(np.float32)
gt, _ = rle_to_labels(truth[sid][f"{mod}_instances"], img.shape)
pred = data[f"{sid}|{mod}"].astype(np.int32)
overlap = np.bincount(gt[pred > 0], minlength=gt.max() + 1)
missed = np.isin(gt, np.flatnonzero(overlap == 0)) & (gt > 0)
yy, xx = np.nonzero(missed)
dens = np.zeros(img.shape, np.float32)
dens[yy, xx] = 1
dens = cv2.blur(dens, (size, size))
cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
y0, x0 = max(cy - size // 2, 0), max(cx - size // 2, 0)
sl = (slice(y0, y0 + size), slice(x0, x0 + size))
lo, hi = np.percentile(img, [1, 99.7])
base = (np.clip((img[sl] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
raw = rgb.copy()
rgb[missed[sl]] = (0.5 * rgb[missed[sl]] + [0, 110, 110]).astype(np.uint8)
for lab, color in ((gt[sl], (0, 255, 0)), (pred[sl], (0, 0, 255))):
    edge = (cv2.dilate(lab.astype(np.float32), np.ones((3, 3))) != cv2.erode(lab.astype(np.float32), np.ones((3, 3)))) & (lab > 0)
    rgb[edge] = color
out = np.hstack([cv2.resize(raw, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST),
                 cv2.resize(rgb, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)])
path = f"/private/tmp/cm/miss_{sid[-6:]}_{mod}.png"
cv2.imwrite(path, out)
print(path, "crop", y0, x0, "missed in crop", len(np.unique(gt[sl][missed[sl]])))
