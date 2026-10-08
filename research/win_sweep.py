"""Sweep window radius / angle_win / second-mode cutoff on train registration."""
import os
import pickle
from multiprocessing import Pool

import numpy as np

from hough import window_register
from margin_lab import R, C, margin
from reg_lab import err
from vote import P0, vote_modes

ANGLE_WINS = (5.0, 8.0, 12.0)
RADII = (120, 180, 250)
MODE_CUTS = (0.45, 0.30, 0.20, 0.10)


def run(args):
    sid, modes, aw, rad = args
    r = R[sid]
    M, score = window_register(r["iv_c"], r["ex_c"], r["offset"], modes, P0, angle_win=aw, radius=rad)
    return sid, M, score


if __name__ == "__main__":
    base_cands = C
    for mcut in MODE_CUTS:
        # patch vote_modes second-mode threshold via wrapping
        def modes_for(subj, cut=mcut):
            sids = [s for s in R if R[s]["subject"] == subj]
            # copy vote_modes with custom cut
            from vote import vote_modes as vm
            import vote
            # temporarily monkeypatch by inlining
            cands_by = {s: base_cands[s] for s in sids}
            norm = {}
            for s, cands in cands_by.items():
                sc = np.array([c[0] for c in cands])
                base, spread = np.median(sc), sc.std() + 1e-6
                norm[s] = [(max((c[0] - base) / spread, 0), c[1], c[2]) for c in cands]
            pool = [(s, z, a, l) for s, cs in norm.items() for z, a, l in cs]
            modes, taken = [], set()
            for _ in range(2):
                best = None
                for s0, z0, a0, l0 in pool:
                    support, members = 0.0, []
                    for s, cs in norm.items():
                        if s in taken:
                            continue
                        inside = [(z, a, l) for z, a, l in cs
                                  if abs(a - a0) < 3 and np.linalg.norm(l - l0) < 70]
                        if inside:
                            z, a, l = max(inside, key=lambda t: t[0])
                            support += z
                            if z > 0:
                                members.append((s, a, l))
                    if best is None or support > best[0]:
                        best = (support, members)
                if best is None or len(best[1]) < 2 or (modes and best[0] < cut * modes[0]["support"]):
                    break
                modes.append({"angle": float(np.median([a for _, a, _ in best[1]])),
                              "landing": np.median([l for _, _, l in best[1]], axis=0),
                              "support": best[0], "n": len(best[1])})
                taken |= {s for s, _, _ in best[1]}
            return {s: modes for s in sids}

        all_modes = {}
        for subj in sorted({r["subject"] for r in R.values()}):
            all_modes.update(modes_for(subj))

        for aw in ANGLE_WINS:
            for rad in RADII:
                jobs = [(s, all_modes[s], aw, rad) for s in R]
                with Pool(8) as pool:
                    out = dict((s, (M, sc)) for s, M, sc in pool.map(run, jobs))
                ok = sum(out[s][0] is not None and err(R[s], out[s][0]) < 5 for s in R)
                mg = {s: margin(base_cands[s], out[s][0], out[s][1], R[s]) for s in R}
                g = [s for s in R if mg[s] >= 3]
                gok = sum(out[s][0] is not None and err(R[s], out[s][0]) < 5 for s in g)
                print(f"mcut {mcut:.2f} aw {aw:4.1f} rad {rad:3d} | correct {ok}/47 "
                      f"gated {len(g)} ok {gok} GT {sum(R[s]['n_gt_pairs'] for s in g if err(R[s], out[s][0]) < 5)}",
                      flush=True)
