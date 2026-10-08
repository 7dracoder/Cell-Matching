"""Per-mouse consensus by voting over each region's top-K refined Hough candidates."""
import numpy as np
from hough import hough_candidates
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from registration import refine  # noqa: E402

P0 = np.array([300.0, 300.0])


def region_candidates(iv_c, ex_c, offset, top=30, scales=np.arange(0.85, 1.13, 0.02)):
    """[(score, angle, landing, M)] refined candidates, deduplicated."""
    out = []
    for votes, angle, scale, M in hough_candidates(iv_c, ex_c, scales=scales, per_pose=2, keep=top * 2)[:top * 2]:
        R, score = refine(iv_c, ex_c, M)
        a = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        land = R[:, :2] @ (P0 - offset) + R[:, 2]
        if any(abs(a - b) < 1 and np.linalg.norm(land - l) < 10 for _, b, l, _ in out):
            continue
        out.append((score, a, land, R))
    out.sort(key=lambda c: c[0], reverse=True)
    return out[:top]


def vote_modes(cands_by_region, ang_tol=3.0, land_tol=70.0, max_modes=2):
    """Cluster centre maximising the sum over regions of the best normalised score within tolerance."""
    norm = {}
    for s, cands in cands_by_region.items():
        sc = np.array([c[0] for c in cands])
        base, spread = np.median(sc), sc.std() + 1e-6
        norm[s] = [(max((c[0] - base) / spread, 0), c[1], c[2]) for c in cands]
    pool = [(s, z, a, l) for s, cs in norm.items() for z, a, l in cs]
    modes, taken = [], set()
    for _ in range(max_modes):
        best = None
        for s0, z0, a0, l0 in pool:
            support, members = 0.0, []
            for s, cs in norm.items():
                if s in taken:
                    continue
                inside = [(z, a, l) for z, a, l in cs if abs(a - a0) < ang_tol and np.linalg.norm(l - l0) < land_tol]
                if inside:
                    z, a, l = max(inside, key=lambda t: t[0])
                    support += z
                    if z > 0:
                        members.append((s, a, l))
            if best is None or support > best[0]:
                best = (support, members)
        if best is None or len(best[1]) < 2 or (modes and best[0] < 0.45 * modes[0]["support"]):
            break
        modes.append({"angle": float(np.median([a for _, a, _ in best[1]])),
                      "landing": np.median([l for _, _, l in best[1]], axis=0),
                      "support": best[0], "n": len(best[1])})
        taken |= {s for s, _, _ in best[1]}
    return modes
