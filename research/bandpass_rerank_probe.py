"""Rerank existing mouse-held-out registration hypotheses by image evidence."""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from bandpass_registration_probe import HERE, pose_score
from common import ROOT
from reg_lab import err
from margin_lab import margin
from registration import transform
import pair_clf


def main():
    with (HERE / "data/lab.pkl").open("rb") as stream:
        records = pickle.load(stream)
    with (HERE / "data/vote_cands.pkl").open("rb") as stream:
        candidates = pickle.load(stream)
    with (HERE / "data/reg_window_vote5.pkl").open("rb") as stream:
        previous = pickle.load(stream)
    results = {}
    selected_regs = {}
    image_gates = {}
    for sid, record in records.items():
        subject, region = sid.split("__")
        iv = cv2.imread(str(Path(ROOT) / "training" / subject / region / "invivo.tif"),
                        cv2.IMREAD_UNCHANGED)
        ex = cv2.imread(str(Path(ROOT) / "training" / subject / region / "exvivo.tif"),
                        cv2.IMREAD_UNCHANGED)
        poses = [("previous", previous[sid][0], 0.0)] + [
            (f"vote_{j}", item[3], float(item[0]))
            for j, item in enumerate(candidates[sid])]
        ranked = []
        for source, M, hough in poses:
            if M is None:
                continue
            bp = pose_score(iv, ex, M, 2, 8)
            if np.isfinite(bp):
                ranked.append((bp, source, M, hough))
        if not ranked:
            continue
        ranked.sort(key=lambda x: x[0], reverse=True)
        best = ranked[0]
        h, w = record["iv_shape"]
        anchors = np.array([[0, 0], [w / 2, h / 2], [w, h]], dtype=np.float32)
        mapped = transform(anchors, best[2])
        runner = next((z for z in ranked[1:] if np.max(np.linalg.norm(
            transform(anchors, z[2]) - mapped, axis=1)) > 50), None)
        image_gap = best[0] - runner[0] if runner else np.inf
        vote_gap = margin(candidates[sid], previous[sid][0], previous[sid][1], record)
        prior_score = next((z[0] for z in ranked if z[1] == "previous"), -np.inf)
        # Preserve the existing high-margin poses and let independent image
        # evidence rescue only the ambiguous ones.
        use_best = vote_gap < 3 and image_gap >= 0.03
        choice = best if use_best else next(z for z in ranked if z[1] == "previous")
        selected_regs[sid] = (choice[2], previous[sid][1] if choice[1] == "previous" else choice[3])
        image_gates[sid] = bool(use_best and image_gap >= 0.03)
        old_error = err(record, previous[sid][0])
        new_error = err(record, best[2])
        results[sid] = (old_error, new_error, best[1], best[0] - prior_score,
                        image_gap, vote_gap)
        print(sid, "old", round(old_error, 1), "new", round(new_error, 1),
              best[1], "score", round(best[0], 4),
              "margin", round(best[0] - prior_score, 4),
              "image_gap", round(image_gap, 4), flush=True)
    print("TOTAL", len(results), "old correct", sum(x[0] < 5 for x in results.values()),
          "new correct", sum(x[1] < 5 for x in results.values()),
          "rescued", sum(x[0] >= 5 and x[1] < 5 for x in results.values()),
          "lost", sum(x[0] < 5 and x[1] >= 5 for x in results.values()))
    for threshold in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08):
        chosen = [x[1] if x[3] >= threshold else x[0] for x in results.values()]
        print("MARGIN", threshold, "correct", sum(e < 5 for e in chosen),
              "rescued", sum(x[0] >= 5 and x[1] < 5 and x[3] >= threshold
                            for x in results.values()),
              "lost", sum(x[0] < 5 and x[1] >= 5 and x[3] >= threshold
                         for x in results.values()))
    for cut in (0.005, 0.01, 0.02, 0.03, 0.05):
        gated = [x for x in results.values() if x[4] >= cut]
        extra = [x for x in gated if x[5] < 3]
        print("IMAGE_GAP", cut, "regions", len(gated), "correct", sum(x[1] < 5 for x in gated),
              "low_vote_margin", len(extra), "correct_extra", sum(x[1] < 5 for x in extra))
    print("SELECTED_CORRECT", sum(err(records[s], selected_regs[s][0]) < 5 for s in selected_regs))
    pair_runs = {}
    for name, regs in (("previous", previous), ("bandpass", selected_regs)):
        rows = pair_clf.dataset({s: (regs[s][0], regs[s][1]) for s in records})
        probabilities = pair_clf.loo_predict(rows)
        pair_runs[name] = (rows, probabilities)
        for gate in ("vote", "vote_or_image"):
            if name == "previous" and gate != "vote":
                continue
            good = {}
            for sid, record in records.items():
                vote_ok = margin(candidates[sid], regs[sid][0], regs[sid][1], record) >= 3
                good[sid] = vote_ok or (gate == "vote_or_image" and image_gates[sid])
            best_f1 = (0.0, 0.0, 0, 0)
            for threshold in np.arange(0.0, 0.3, 0.025):
                tp = sum(int(y[probabilities[sid] >= threshold].sum())
                         for sid, _, _, y in rows if good[sid])
                pred = sum(int((probabilities[sid] >= threshold).sum())
                           for sid, _, _, y in rows if good[sid])
                f1 = 2 * tp / (pred + pair_clf.TOTAL)
                best_f1 = max(best_f1, (f1, float(threshold), tp, pred))
            print("PAIR_F1", name, gate, best_f1, "regions", sum(good.values()))
    old_rows, old_probs = pair_runs["previous"]
    new_rows, new_probs = pair_runs["bandpass"]
    old_gate = {sid: margin(candidates[sid], previous[sid][0], previous[sid][1], records[sid]) >= 3
                for sid in records}
    tp_old = sum(int(y[old_probs[sid] >= .025].sum()) for sid, _, _, y in old_rows if old_gate[sid])
    pr_old = sum(int((old_probs[sid] >= .025).sum()) for sid, _, _, y in old_rows if old_gate[sid])
    for extra_threshold in (0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4):
        extra = [(sid, y) for sid, _, _, y in new_rows if image_gates[sid] and not old_gate[sid]]
        tp_extra = sum(int(y[new_probs[sid] >= extra_threshold].sum()) for sid, y in extra)
        pr_extra = sum(int((new_probs[sid] >= extra_threshold).sum()) for sid, y in extra)
        f1 = 2 * (tp_old + tp_extra) / (pr_old + pr_extra + pair_clf.TOTAL)
        print("BLEND_F1", extra_threshold, round(f1, 5), "extra", tp_extra, pr_extra)


if __name__ == "__main__":
    main()
