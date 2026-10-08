"""Align serial ex-vivo sections of one mouse to a reference section by intensity (ECC)."""
import numpy as np, cv2, tifffile

DOWN = 4


def prep(image):
    im = image.astype(np.float32)
    lo, hi = np.percentile(im, [1, 99.5])
    im = np.clip((im - lo) / max(hi - lo, 1), 0, 1)
    im = cv2.resize(im, (im.shape[1] // DOWN, im.shape[0] // DOWN), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(im, (0, 0), 2)


def align(moving, ref, motion=cv2.MOTION_AFFINE):
    """2x3 matrix mapping full-res moving-canvas coords to full-res reference coords, and the ECC value."""
    best = (None, -1)
    h = max(moving.shape[0], ref.shape[0])
    w = max(moving.shape[1], ref.shape[1])
    pad = lambda a: cv2.copyMakeBorder(a, 0, h - a.shape[0], 0, w - a.shape[1], cv2.BORDER_CONSTANT)
    m, r = pad(moving), pad(ref)
    for angle in (-6, -3, 0, 3, 6):
        warp = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0).astype(np.float32)
        try:
            cc, warp = cv2.findTransformECC(r, m, warp, motion,
                                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
        except cv2.error:
            continue
        if cc > best[1]:
            best = (warp, cc)
    warp, cc = best
    if warp is None:
        return None, 0.0
    # findTransformECC warp maps reference coords -> moving coords; invert and rescale to full res
    inv = cv2.invertAffineTransform(warp)
    inv[:, 2] *= DOWN
    return inv.astype(np.float64), float(cc)


def compose(A, B):
    """A after B for 2x3 affine matrices."""
    return np.c_[A[:, :2] @ B[:, :2], A[:, :2] @ B[:, 2] + A[:, 2]]
