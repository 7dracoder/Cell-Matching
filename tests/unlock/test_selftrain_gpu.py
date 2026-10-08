"""Unit tests for hpc_unlock.selftrain_gpu (Req 12.4, 12.5, 12.9, 12.14, 3.5, 3.6).

Cellpose is replaced through the module seams (``_new_model``, ``_train_seg``,
``_flows``, ``_decode``, ``_torch_info``); the Stage logic, the side files and
the Checkpoint / done-marker handling are real.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hpc_unlock import checkpoint
from hpc_unlock import selftrain_gpu as S
from hpc_unlock.config import UnlockConfig
from hpc_unlock.stage import StageContext

REPO = Path(__file__).resolve().parents[2]
GPU = {"version": "2.x", "cuda": "12.1", "available": True, "device": "FakeGPU"}
NOGPU = {"version": "2.x", "cuda": None, "available": False, "device": "cpu"}
JOB_REGIONS = {"m0": [f"m0__r{k}" for k in range(3)], "m1": [f"m1__r{k}" for k in range(2)],
               "m2": [f"m2__r{k}" for k in range(2)], "test": [f"t__r{k:02d}" for k in range(29)]}
N_TILES = {"m0": 4, "m1": 0, "m2": 3, "test": 5}


# --------------------------------------------------------------- synthetic prep
def write_prep(run_dir: Path, seed: int = 0) -> dict:
    """A ``selftrain_prep`` Checkpoint + side file with a done-marker; returns the arrays."""
    rng = np.random.default_rng(seed)
    arrays, jobs = {}, {}
    for j_i, (job, regions) in enumerate(JOB_REGIONS.items()):
        for sid in regions:
            arrays[f"image|{sid}"] = rng.random((24, 20)).astype(np.float32)
        n = N_TILES[job]
        arrays[f"tiles|{job}|image"] = np.full((n, 96, 96), j_i + 1.0, np.float32) \
            + rng.random((n, 96, 96)).astype(np.float32) * 0.01
        arrays[f"tiles|{job}|label"] = rng.integers(0, 4, (n, 96, 96)).astype(np.int32)
        arrays[f"tiles|{job}|sid"] = np.array([regions[0]] * n, dtype=str) if n \
            else np.zeros(0, "<U1")
        nc = job == "m1"
        jobs[job] = {"split": "test" if job == "test" else "heldout", "regions": regions,
                     "train_regions": [] if nc else regions[:1], "n_tiles": n,
                     "no_confident": nc, "reason": "no confident regions" if nc else None}

    def compute():
        ref = checkpoint.save_npz_atomic(run_dir / "selftrain_tiles.npz", **arrays)
        return {"selection": "joint", "jobs": jobs, "tiles_side": ref,
                "keys": {"image": "image|<sid>", "tiles": "tiles|<job>|image",
                         "labels": "tiles|<job>|label", "tile_sid": "tiles|<job>|sid"}}

    checkpoint.run_stage("selftrain_prep", compute, run_dir)
    return arrays


class FakeNet:
    pass


class FakeModel:
    def __init__(self, path):
        self.path = None if path is None else Path(path)
        self.net = FakeNet()


class Fakes:
    """Records every call to the Cellpose seams."""

    def __init__(self, fail_on: str | None = None):
        self.models: list[FakeModel] = []
        self.trains: list[dict] = []
        self.flows: list[tuple] = []
        self.decodes: list[tuple] = []
        self.fail_on = fail_on
        self.fail_on_after = 0     # number of successful flow calls before the failure

    def new_model(self, path=None):
        m = FakeModel(path)
        self.models.append(m)
        return m

    def train_seg(self, net, train_data, train_labels, **kw):
        self.trains.append({"net": net, "data": train_data, "labels": train_labels, **kw})
        out = Path(kw["save_path"]) / "models" / kw["model_name"]
        out.parent.mkdir(exist_ok=True)   # like cellpose: save_path must exist
        out.write_bytes(b"weights")
        return out, [], []

    def flows_fn(self, model, image):
        self.flows.append((model, image.shape))
        if self.fail_on and len(self.flows) > self.fail_on_after:
            raise RuntimeError("CUDA out of memory")
        return (np.stack([image, -image]).astype(np.float16), image.astype(np.float16), 7)

    def decode(self, flows, cellprob, flow):
        dp, cp, n = flows
        self.decodes.append((cellprob, flow, n))
        return (cp.astype(np.float32) > 0.5 + 0.1 * cellprob).astype(np.int32)

    def install(self, monkeypatch, info=GPU):
        monkeypatch.setattr(S, "_torch_info", lambda: dict(info))
        monkeypatch.setattr(S, "_new_model", self.new_model)
        monkeypatch.setattr(S, "_train_seg", lambda: self.train_seg)
        monkeypatch.setattr(S, "_flows", self.flows_fn)
        monkeypatch.setattr(S, "_decode", self.decode)
        return self


def _ctx(run_dir: Path, smoke: bool = False) -> StageContext:
    return StageContext(name="selftrain_gpu", run_dir=run_dir, smoke=smoke, workers=1)


def _run(run_dir: Path, cfg=None, smoke=False):
    cfg = cfg or UnlockConfig()
    return checkpoint.run_stage("selftrain_gpu", lambda: S.compute(cfg, _ctx(run_dir, smoke)),
                                run_dir)


# --------------------------------------------------------------- full run
def test_full_run_trains_fresh_models_on_exvivo_tiles_and_writes_flows(tmp_path, monkeypatch,
                                                                        capsys):
    arrays = write_prep(tmp_path)
    fk = Fakes().install(monkeypatch)
    ck = _run(tmp_path)
    lines = capsys.readouterr().out.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("TORCH_INFO "))
    assert lines[i] == "TORCH_INFO version=2.x cuda=12.1 available=True device=FakeGPU"
    assert lines[i - 1].startswith("COMPUTE selftrain_gpu")   # first line of the Stage

    assert ck["order"] == ["m0", "m1", "m2", "test"]
    st = {j: v["status"] for j, v in ck["jobs"].items()}
    assert st == {"m0": "done", "m1": "skipped_no_confident", "m2": "done", "test": "done"}
    assert ck["jobs"]["m1"]["flows_side"] is None and ck["jobs"]["m1"]["reason"]

    # one fresh pretrained model per trained job; no-confident job never trained
    assert [t["model_name"] for t in fk.trains] == [
        "exvivo_cpsam_selftrain_m0", "exvivo_cpsam_selftrain_m2", "exvivo_cpsam_selftrain_test"]
    nets = [t["net"] for t in fk.trains]
    assert len({id(n) for n in nets}) == 3
    pre = {id(m.net): m for m in fk.models}
    assert all(pre[id(n)].path is None for n in nets)       # cpsam_v2 pretrained weights
    for t, job in zip(fk.trains, ["m0", "m2", "test"]):
        # only this job's ex-vivo tiles and pseudo-label tiles
        assert len(t["data"]) == len(t["labels"]) == N_TILES[job]
        np.testing.assert_array_equal(np.stack(t["data"]), arrays[f"tiles|{job}|image"])
        np.testing.assert_array_equal(np.stack(t["labels"]), arrays[f"tiles|{job}|label"])
        assert t["normalize"] is False and t["rescale"] is True
        assert (t["batch_size"], t["n_epochs"], t["nimg_per_epoch"]) == (8, 60, 128)
        assert (t["learning_rate"], t["weight_decay"], t["min_train_masks"]) == (1e-5, 0.1, 1)
        assert Path(t["save_path"]) == tmp_path / "selftrain_models" / job

    # inference with the fine-tuned model on every ex-vivo image of the job
    infer_models = [m for m, _ in fk.flows]
    assert all(m.path is not None and m.path.name.startswith("exvivo_cpsam_selftrain_")
               for m in infer_models)
    by_job = {}
    for m, _ in fk.flows:
        by_job[m.path.name.rsplit("_", 1)[1]] = by_job.get(m.path.name.rsplit("_", 1)[1], 0) + 1
    assert by_job == {"m0": 3, "m2": 2, "test": 29}
    assert len(fk.decodes) == 6 * (3 + 2 + 29)
    assert {(c, f) for c, f, _ in fk.decodes} == set(UnlockConfig().st_decode_grid)

    # Checkpoint reloads with verified side refs; npz keys exactly as specified
    assert checkpoint.marker_path(tmp_path, "selftrain_gpu").is_file()
    ck2 = checkpoint.load(checkpoint.checkpoint_path(tmp_path, "selftrain_gpu"))
    for job in ("m0", "m2", "test"):
        ref = ck2["jobs"][job]["flows_side"]
        assert ref[checkpoint.SIDE_KEY] == f"selftrain_flows/{job}.npz"
        z = checkpoint.load_npz(ref, tmp_path)
        assert set(z) == S.expected_keys(JOB_REGIONS[job], 6)
        sid = JOB_REGIONS[job][0]
        assert z[f"{sid}|dp"].dtype == np.float16 and z[f"{sid}|dp"].shape == (2, 24, 20)
        assert z[f"{sid}|cp"].dtype == np.float16
        assert z[f"{sid}|n"].dtype == np.int32 and int(z[f"{sid}|n"]) == 7
        assert all(z[f"{sid}|labels|{k}"].dtype == np.uint16 for k in range(6))
    assert ck2["jobs"]["test"]["regions"] == JOB_REGIONS["test"]
    assert ck2["torch"] == GPU


def test_failure_logs_cause_exits_1_without_marker(tmp_path, monkeypatch, capsys):
    write_prep(tmp_path)
    fk = Fakes(fail_on="test").install(monkeypatch)
    fk.fail_on_after = 3 + 2          # m0 and m2 regions succeed, first test region fails
    with pytest.raises(SystemExit) as e:
        _run(tmp_path)
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "SELFTRAIN_FAILED test: RuntimeError: CUDA out of memory" in out
    assert not checkpoint.marker_path(tmp_path, "selftrain_gpu").exists()
    assert not checkpoint.checkpoint_path(tmp_path, "selftrain_gpu").exists()
    assert not (tmp_path / "selftrain_flows" / "test.npz").exists()
    assert (tmp_path / "selftrain_flows" / "m0.npz").is_file()


def test_resume_skips_finished_jobs_and_reuses_trained_model(tmp_path, monkeypatch, capsys):
    write_prep(tmp_path)
    fk = Fakes(fail_on="test").install(monkeypatch)
    fk.fail_on_after = 3 + 2
    with pytest.raises(SystemExit):
        _run(tmp_path)
    fk2 = Fakes().install(monkeypatch)
    ck = _run(tmp_path)
    assert {j: v["resumed"] for j, v in ck["jobs"].items() if v["status"] == "done"} == \
        {"m0": True, "m2": True, "test": False}
    # the test model finished training before the failure: reused, not retrained
    assert fk2.trains == []
    assert ck["jobs"]["test"]["model_reused"] is True
    assert len(fk2.flows) == 29
    assert "SELFTRAIN_RESUME m0" in capsys.readouterr().out
    z = checkpoint.load_npz(ck["jobs"]["m0"]["flows_side"], tmp_path)
    assert set(z) == S.expected_keys(JOB_REGIONS["m0"], 6)


def test_finished_rejects_truncated_or_incomplete_npz(tmp_path):
    p = tmp_path / "x.npz"
    np.savez(p, **{"a|dp": np.zeros(2), "a|cp": np.zeros(2), "a|n": np.int32(1),
                   "a|labels|0": np.zeros(2, np.uint16)})
    assert S.finished(p, ["a"], 1)
    assert not S.finished(p, ["a"], 2)
    assert not S.finished(p, ["a", "b"], 1)
    p.write_bytes(p.read_bytes()[:40])
    assert not S.finished(p, ["a"], 1)


# --------------------------------------------------------------- no GPU / smoke
class NoLoadCtx:
    smoke = False
    run_dir = Path("/nonexistent")

    def load(self, name):
        raise AssertionError(f"loaded {name} before the GPU check")


def test_no_gpu_exits_1_before_loading(monkeypatch, capsys):
    Fakes().install(monkeypatch, info=NOGPU)
    with pytest.raises(SystemExit) as e:
        S.compute(UnlockConfig(), NoLoadCtx())
    assert e.value.code == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("TORCH_INFO ") and "available=False" in lines[0]
    assert lines[1].startswith("NO_GPU_VISIBLE selftrain_gpu")


def test_smoke_runs_no_fine_tune(tmp_path, monkeypatch, capsys):
    write_prep(tmp_path)
    fk = Fakes().install(monkeypatch, info=NOGPU)
    ck = _run(tmp_path, smoke=True)
    assert {v["status"] for v in ck["jobs"].values()} == {"skipped_smoke"}
    assert fk.models == fk.trains == fk.flows == []
    assert not (tmp_path / "selftrain_flows").exists()
    assert ck["params"]["smoke"] is True
    assert "SKIPPED (no fine-tune in smoke)" in capsys.readouterr().out


# --------------------------------------------------------------- seams / helpers
def test_default_seams_delegate_to_pipeline(monkeypatch):
    import pipeline

    calls = {}
    monkeypatch.setattr(pipeline, "cellpose_model", lambda path=None: calls.setdefault("m", path))
    monkeypatch.setattr(pipeline, "cellpose_flows", lambda m, im: calls.setdefault("f", (m, im)))
    monkeypatch.setattr(pipeline, "cellpose_labels",
                        lambda fl, mod, cfg: calls.setdefault("l", (mod, cfg)))
    S._new_model(None)
    assert calls["m"] is None
    S._flows("model", "img")
    assert calls["f"] == ("model", "img")
    S._decode(("dp", "cp", 3), -0.5, 0.4)
    assert calls["l"] == ("exvivo", {"cellprob": -0.5, "flow": 0.4})


def test_to_uint16_range():
    assert S.to_uint16(np.array([[0, 65535]], np.int32)).dtype == np.uint16
    with pytest.raises(ValueError):
        S.to_uint16(np.array([65536], np.int32))


def test_import_graph_excludes_cpu_only_modules():
    code = ("import sys; import hpc_unlock.selftrain_gpu; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
            "('registration', 'sklearn', 'cellpose', 'pipeline') "
            "or m in ('hpc_unlock.validate', 'hpc_unlock.assemble', 'hpc_unlock.selftrain_pairs', "
            "'hpc_unlock.pairs', 'hpc_unlock.verifier', 'validate', 'assemble')); print(bad)")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"
