"""gpu_scan peak selection and Stage ``compute`` (Req 5.5-5.9, 5.11, 5.12, 3.5, 3.6, 13.2).

CPU torch only: the Stage runs in smoke mode (5 x 3 x 1 grid) on synthetic regions.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hpc_unlock import checkpoint, gpu_scan as gs
from hpc_unlock.config import UnlockConfig

REPO = Path(__file__).resolve().parents[2]


class FakeCtx:
    """Minimal StageContext stand-in: ``load("prep")`` returns the given scan inputs."""

    def __init__(self, scan_inputs, smoke=True, run_dir=None):
        self.name, self.smoke, self.run_dir, self.workers = "gpu_scan", smoke, run_dir, 1
        self.loads: list[str] = []
        self._prep = {"scan_inputs": scan_inputs}

    def load(self, dep):
        self.loads.append(dep)
        assert dep == "prep"
        return self._prep


def _spaced(rng, n, lo, hi, min_d):
    pts = []
    while len(pts) < n:
        p = rng.uniform(lo, hi, 2)
        if all(np.hypot(*(p - q)) >= min_d for q in pts):
            pts.append(p)
    return np.array(pts)


def _planted(seed=5, angle=17.5, scale=1.0, tau=(140.3, 95.6), shape=(360, 420)):
    rng = np.random.default_rng(seed)
    iv = _spaced(rng, 30, 0, 160, 11.0)
    A = scale * gs.rotation(angle)
    ex = iv @ A.T + np.asarray(tau)
    ex = np.r_[ex, rng.uniform([0, 0], [shape[1] - 1, shape[0] - 1], (15, 2))]   # clutter
    assert (ex >= 0).all() and (ex[:, 0] < shape[1]).all() and (ex[:, 1] < shape[0]).all()
    return {"iv_c": iv.astype(np.float32), "ex_c": ex.astype(np.float32),
            "ex_shape": shape, "offset": np.array([40.0, 25.0], np.float32)}


# ------------------------------------------------------------ selection
def test_select_peaks_separation_rule():
    score = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]
    angle = [0.0, 2.0, 0.0, 10.0, 3.0, -3.0]
    land = [(0, 0), (5, 0), (30, 0), (0, 0), (0, 20), (12, 16)]
    # 1: |da|=2, |dl|=5 -> skip; 2: |dl|=30 -> keep; 3: |da|=10 -> keep;
    # 4: |da|=3 and |dl|=20 (both at the bound) -> skip; 5: |da|=3, |dl|=20 vs 0 -> skip
    assert gs.select_peaks(score, angle, land, 50).tolist() == [0, 2, 3]


def test_select_peaks_descending_and_k_cap():
    rng = np.random.default_rng(0)
    n = 40
    score = rng.permutation(n).astype(float)
    angle = np.arange(n) * 10.0                                   # all separated
    land = np.zeros((n, 2))
    kept = gs.select_peaks(score, angle, land, 7)
    assert len(kept) == 7
    assert np.all(np.diff(score[kept]) < 0)
    assert set(kept) == set(np.argsort(-score)[:7])


def test_select_peaks_fewer_than_k_and_empty():
    kept = gs.select_peaks([3.0, 2.0, 1.0], [0.0, 0.5, 30.0], [(0, 0), (1, 1), (0, 0)], 50)
    assert kept.tolist() == [0, 2]
    assert gs.select_peaks([], [], np.zeros((0, 2)), 50).tolist() == []


def test_select_peaks_chunked_equals_sequential():
    rng = np.random.default_rng(1)
    n = 600
    score = rng.normal(size=n)
    angle = rng.choice(np.arange(-35, 35.5, 0.5), n)
    land = rng.uniform(0, 120, (n, 2))

    def naive(k):
        out = []
        for i in np.argsort(-score, kind="stable"):
            if any(abs(angle[i] - angle[j]) <= 3 and np.hypot(*(land[i] - land[j])) <= 20 for j in out):
                continue
            out.append(i)
            if len(out) == k:
                break
        return out

    for k in (5, 60, 500):
        assert gs.select_peaks(score, angle, land, k, chunk=17).tolist() == naive(k)


def test_batch_from_free_clamped():
    assert gs.batch_from_free(24e9, 1350, 1350) == int(0.6 * 24e9 // (12 * 1350 * 1350))
    assert gs.batch_from_free(1e6, 1350, 1350) == 16
    assert gs.batch_from_free(1e15, 256, 256) == 1024


def test_batch_peaks_are_valid_local_maxima():
    inp = _planted()
    grid = gs.smoke_grid()
    spec = gs.region_spectrum(inp, grid, 2.5, 2.0, "cpu")
    res = gs.correlate_batch(spec, torch.from_numpy(inp["iv_c"]), torch.from_numpy(grid.A))
    pk = gs.batch_peaks(spec, res)
    valid = gs.valid_mask(spec, res.o, res.centre).numpy()
    corr = res.corr.numpy()
    ky = np.mod(pk.ty, spec.Hf)
    kx = np.mod(pk.tx, spec.Wf)
    assert len(pk.score) > 0 and np.bincount(pk.b, minlength=len(grid)).max() <= 4
    for s, b, y, x in zip(pk.score, pk.b, ky, kx):
        assert valid[b, y, x] and s > 0 and np.isclose(s, corr[b, y, x])
        nb = corr[b, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        nv = valid[b, max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        assert s >= nb[nv].max()


# --------------------------------------------------------------- Stage
def test_smoke_grid():
    g = gs.smoke_grid()
    assert len(g) == 15
    assert 0.0 in g.angles.tolist() and g.angles[0] == -35.0 and g.angles[-1] == 35.0
    assert 1.0 in g.scales.tolist() and g.stretches == ((1.0, None),)


def test_compute_smoke_recovers_planted_pose_and_copies_duplicates(capsys):
    tau = np.array([140.3, 95.6])
    planted = _planted(tau=tuple(tau))
    other = _planted(seed=9, angle=-17.5, scale=1.13, tau=(120.0, 150.0))
    inputs = {"k1": {**planted, "sids": ["s__a", "s__b"]}, "k2": {**other, "sids": ["t__c"]}}
    cfg = UnlockConfig(scan_k=10)
    out = gs.compute(cfg, FakeCtx(inputs))

    assert set(out["by_sid"]) == {"s__a", "s__b", "t__c"}
    assert out["grid"]["n_poses"] == 15 and out["grid"]["smoke"] is True
    assert out["torch"]["version"] == torch.__version__
    a, b = out["by_sid"]["s__a"], out["by_sid"]["s__b"]
    assert a is not b and a["kept"] == b["kept"] == len(a["cands"])
    for ca, cb in zip(a["cands"], b["cands"]):
        assert ca["M"] is not cb["M"] and np.array_equal(ca["M"], cb["M"])

    off = planted["offset"].astype(float)
    for entry in out["by_sid"].values():
        cands = entry["cands"]
        assert 0 < entry["kept"] <= 10
        assert all(cands[i]["score"] >= cands[i + 1]["score"] for i in range(len(cands) - 1))
        for c in cands:
            M = c["M"]
            assert M.shape == (2, 3) and M.dtype == np.float64
            assert c["stretch_k"] == 1.0 and c["stretch_dir"] is None
            assert np.allclose(M[:, :2], c["scale"] * gs.rotation(c["angle"]))
            assert np.allclose(M[:, 2], c["translation"])
        for i, ci in enumerate(cands):                       # separation holds among kept
            for cj in cands[:i]:
                assert (abs(ci["angle"] - cj["angle"]) > 3
                        or np.hypot(*np.subtract(ci["landing"], cj["landing"])) > 20)

    for c in a["cands"]:
        assert np.allclose(c["landing"], c["M"][:, :2] @ (gs.P0 - off) + c["M"][:, 2])
    hit = [i for i, c in enumerate(a["cands"][:3])
           if c["angle"] == 17.5 and c["scale"] == 1.0
           and np.linalg.norm(np.subtract(c["translation"], tau)) <= 2.0]
    assert hit, a["cands"][:3]
    o = out["by_sid"]["t__c"]["cands"][0]
    assert o["angle"] == -17.5 and o["scale"] == 1.13

    log = capsys.readouterr().out
    assert log.startswith("TORCH_INFO version=")
    assert "cuda=" in log.splitlines()[0] and "available=" in log and "device=" in log
    assert log.count("\nSCAN sid=") == 2 and "sid=s__a,s__b " in log


def test_no_gpu_visible_exits_before_any_region(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    ctx = FakeCtx({"k": {**_planted(), "sids": ["s__a"]}}, smoke=False)
    with pytest.raises(SystemExit) as e:
        gs.compute(UnlockConfig(), ctx)
    assert e.value.code == 1
    assert ctx.loads == []
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("TORCH_INFO ") and "available=False" in lines[0]
    assert lines[1].startswith("NO_GPU_VISIBLE")


def _oom_injector(monkeypatch, fail_calls):
    real = gs.correlate_batch
    calls: list[int] = []

    def fake(spec, iv_c, A_batch):
        calls.append(len(A_batch))
        if len(calls) in fail_calls:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected)")
        return real(spec, iv_c, A_batch)

    monkeypatch.setattr(gs, "correlate_batch", fake)
    monkeypatch.setattr(gs, "CPU_BATCH", 8)
    return calls


def test_oom_retries_once_at_half_batch_then_fails_without_marker(monkeypatch, tmp_path, capsys):
    calls = _oom_injector(monkeypatch, fail_calls={1, 2})
    ctx = FakeCtx({"k": {**_planted(), "sids": ["s__a"]}}, run_dir=tmp_path)
    with pytest.raises(SystemExit) as e:
        checkpoint.run_stage("gpu_scan", lambda: gs.compute(UnlockConfig(), ctx), tmp_path, "fp")
    assert e.value.code == 1
    assert calls == [8, 4]
    assert not checkpoint.marker_path(tmp_path, "gpu_scan").exists()
    assert not checkpoint.checkpoint_path(tmp_path, "gpu_scan").exists()
    log = capsys.readouterr().out
    assert "SCAN_OOM_RETRY sid=s__a batch=8 -> 4" in log
    assert "REGION_FAILED s__a: OutOfMemoryError: CUDA out of memory (injected)" in log


def test_oom_retry_succeeds_at_half_batch(monkeypatch):
    calls = _oom_injector(monkeypatch, fail_calls={1})
    out = gs.compute(UnlockConfig(scan_k=5), FakeCtx({"k": {**_planted(), "sids": ["s__a"]}}))
    assert calls == [8, 4, 4, 4, 3]
    assert out["by_sid"]["s__a"]["kept"] > 0


def test_region_exception_logged_and_exit_1(monkeypatch, capsys):
    def boom(*a, **k):
        raise ValueError("bad region")

    monkeypatch.setattr(gs, "correlate_batch", boom)
    with pytest.raises(SystemExit) as e:
        gs.compute(UnlockConfig(), FakeCtx({"k": {**_planted(), "sids": ["s__a"]}}))
    assert e.value.code == 1
    assert "REGION_FAILED s__a: ValueError: bad region" in capsys.readouterr().out


def test_import_graph_excludes_cpu_only_modules():
    code = ("import sys; import hpc_unlock.gpu_scan; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('registration', 'sklearn') "
            "or m in ('hpc_unlock.validate', 'hpc_unlock.assemble', 'hpc_unlock.prep', "
            "'hpc_unlock.soft', 'validate', 'assemble')); print(bad)")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"
