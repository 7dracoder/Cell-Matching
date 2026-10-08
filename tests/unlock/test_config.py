"""Unit tests for hpc_unlock/config.py (Req 2.2, 2.3, 4.8, 5.11, 12.13)."""
from __future__ import annotations

import ast
import dataclasses
import json
import re
from pathlib import Path

import pytest

from hpc_unlock import config as C
from hpc_unlock.config import UnlockConfig

SRC = Path(C.__file__)


# ---------------------------------------------------------------- defaults
def test_defaults_match_design():
    cfg = UnlockConfig()
    assert cfg.account == "cs_gy_6923-2026fa"
    assert cfg.cpu_partition == "n2c48m24"
    assert cfg.gpu_partition == "g2-standard-12"
    assert cfg.env_mode == "singularity"
    assert cfg.sif_name is None
    assert cfg.disable_selftrain is False
    assert cfg.sigma == 2.5
    assert cfg.scan_k == 50
    assert cfg.scan_angle_step == 0.5
    assert cfg.scan_scale_step == 0.01
    assert cfg.scan_cell_px == 2.0
    assert cfg.joint_lambdas == (0.0, 2.0, 5.0, 10.0, 20.0)
    assert cfg.joint_top_l == 10
    assert cfg.match_radius == 6.0 and cfg.seed_radius == 6.0
    assert (cfg.st_epochs, cfg.st_tiles_per_epoch, cfg.st_batch) == (60, 128, 8)
    assert len(cfg.st_decode_grid) == 6
    assert cfg.st_grow15 is True
    assert cfg.validate() == []


def test_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        UnlockConfig().sigma = 2.0  # type: ignore[misc]


def test_resources_cover_every_stage_and_are_independent():
    a, b = UnlockConfig(), UnlockConfig()
    stages = {"setup", "prep", "gpu_scan", "pose_search", "joint", "pairs",
              "verifier", "validate", "selftrain_prep", "selftrain_gpu",
              "selftrain_pairs", "assemble"}
    assert set(a.resources) == stages
    for r in a.resources.values():
        assert {"cpus", "mem", "time"} <= set(r)
    a.resources["prep"]["cpus"] = 1
    assert b.resources["prep"]["cpus"] == 32  # no shared mutable default


def test_stage_resources_gpu_memory_by_partition():
    l4 = UnlockConfig()
    a100 = UnlockConfig(gpu_partition="c12m85-a100-1")
    assert l4.stage_resources("gpu_scan")["mem"] == "40G"
    assert a100.stage_resources("gpu_scan")["mem"] == "64G"
    assert "mem_a100" not in a100.stage_resources("selftrain_gpu")
    assert a100.stage_resources("prep") == {"cpus": 32, "mem": "20G", "time": "01:00:00"}
    with pytest.raises(KeyError):
        l4.stage_resources("nope")


# -------------------------------------------------------------- validation
def test_a100_and_venv_are_valid():
    assert UnlockConfig(gpu_partition="c12m85-a100-1", env_mode="venv").validate() == []


@pytest.mark.parametrize("kw, field, value", [
    ({"gpu_partition": "gpu-x"}, "gpu_partition", "'gpu-x'"),
    ({"env_mode": "conda"}, "env_mode", "'conda'"),
    ({"sigma": 1.49}, "sigma", "1.49"),
    ({"sigma": 2.51}, "sigma", "2.51"),
    ({"sigma": float("nan")}, "sigma", "nan"),
    ({"scan_k": 0}, "scan_k", "0"),
    ({"scan_k": 501}, "scan_k", "501"),
    ({"scan_k": 50.0}, "scan_k", "50.0"),
    ({"scan_k": True}, "scan_k", "True"),
    ({"match_radius": 0.5}, "match_radius", "0.5"),
    ({"seed_radius": 20.5}, "seed_radius", "20.5"),
])
def test_single_invalid_field_names_field_and_value(kw, field, value):
    errs = UnlockConfig(**kw).validate()
    assert len(errs) == 1
    assert errs[0].startswith(f"{field}=")
    assert value in errs[0]


@pytest.mark.parametrize("kw", [
    {"sigma": 1.5}, {"sigma": 2.5}, {"scan_k": 1}, {"scan_k": 500},
    {"match_radius": 1}, {"seed_radius": 20.0},
])
def test_bounds_are_inclusive(kw):
    assert UnlockConfig(**kw).validate() == []


def test_all_errors_returned_not_only_first():
    cfg = UnlockConfig(gpu_partition="bad", env_mode="bad", sigma=9.0,
                       scan_k=-1, match_radius=0.0, seed_radius=99.0)
    errs = cfg.validate()
    assert [e.split("=", 1)[0] for e in errs] == [
        "gpu_partition", "env_mode", "sigma", "scan_k", "match_radius", "seed_radius"]


# ------------------------------------------------------------- fingerprint
def test_fingerprint_format_and_stability():
    fp = UnlockConfig().fingerprint()
    assert re.fullmatch(r"[0-9a-f]{10}", fp)
    assert fp == UnlockConfig().fingerprint()


def test_fingerprint_ignores_non_result_fields():
    base = UnlockConfig().fingerprint()
    res = C.default_resources()
    res["prep"]["cpus"] = 4
    other = UnlockConfig(account="x", cpu_partition="y",
                         gpu_partition="c12m85-a100-1", env_mode="venv",
                         sif_name="z.sif", disable_selftrain=True, resources=res)
    assert other.fingerprint() == base


@pytest.mark.parametrize("kw", [
    {"sigma": 2.0}, {"scan_k": 51}, {"scan_angle_step": 0.25},
    {"scan_scale_step": 0.005}, {"scan_cell_px": 1.0},
    {"joint_lambdas": (0.0, 5.0)}, {"joint_top_l": 5},
    {"match_radius": 5.0}, {"seed_radius": 7.0}, {"st_epochs": 30},
    {"st_tiles_per_epoch": 64}, {"st_batch": 4},
    {"st_decode_grid": ((0.0, 0.4),)}, {"st_grow15": False},
])
def test_fingerprint_changes_with_result_fields(kw):
    assert UnlockConfig(**kw).fingerprint() != UnlockConfig().fingerprint()


def test_fingerprint_int_float_equivalent():
    assert UnlockConfig(match_radius=6).fingerprint() == UnlockConfig().fingerprint()


def test_run_dir(tmp_path):
    cfg = UnlockConfig()
    assert cfg.run_dir(tmp_path) == tmp_path / cfg.fingerprint()
    from hpc_unlock import paths
    assert cfg.run_dir() == paths.HPC / cfg.fingerprint()


# ------------------------------------------------------------- save / load
def test_save_load_round_trip(tmp_path):
    res = C.default_resources()
    res["pose_search"]["mem"] = "24G"
    cfg = UnlockConfig(sigma=2.0, scan_k=100, gpu_partition="c12m85-a100-1",
                       env_mode="venv", sif_name="img.sif",
                       joint_lambdas=(0.0, 1.0), resources=res)
    path = cfg.save(tmp_path / cfg.fingerprint())
    assert path.name == "run_config.json"
    assert not list(path.parent.glob("*.tmp-*"))
    loaded = UnlockConfig.load(path)
    assert loaded == cfg
    assert isinstance(loaded.joint_lambdas, tuple)
    assert isinstance(loaded.st_decode_grid[0], tuple)
    assert UnlockConfig.load(path.parent) == cfg  # directory form
    assert json.loads(path.read_text())["fingerprint"] == cfg.fingerprint()


def test_load_rejects_fingerprint_mismatch(tmp_path):
    path = UnlockConfig().save(tmp_path)
    payload = json.loads(path.read_text())
    payload["config"]["sigma"] = 2.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint"):
        UnlockConfig.load(path)


def test_load_rejects_unknown_field(tmp_path):
    path = UnlockConfig().save(tmp_path)
    payload = json.loads(path.read_text())
    payload["config"]["netid"] = "x"
    del payload["fingerprint"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="netid"):
        UnlockConfig.load(path)


# ------------------------------------------------------------ constraints
def test_imports_standard_library_only():
    import sys
    tree = ast.parse(SRC.read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            mods.add(node.module.split(".")[0])
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}


def test_no_other_account_or_user_path():
    text = SRC.read_text()
    assert set(re.findall(r"cs_gy_\w+-\w+", text)) == {"cs_gy_6923-2026fa"}
    assert "/scratch/" not in text and "/home/" not in text
