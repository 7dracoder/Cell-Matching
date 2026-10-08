"""Build cellmatch_v11_bundle.zip: everything CellMatch_v11_Colab.ipynb needs, in one upload.

Layout inside the zip (extracted to /content on Colab):
  work/            pipeline modules, research/ scripts, Project_2_Dataset/
  *.py, *.csv, *.npz at the root: Colab drivers and the v7 baseline inputs
"""
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "cellmatch_v11_bundle.zip"

MODULES = ["cellmatch.py", "learned.py", "registration.py", "pipeline.py", "validate_submission.py"]
ROOT_FILES = {
    "submission_v7_grow15.csv": HERE / "submission_v7_grow15.csv",
    "iv_baseline_cache.npz": HERE / "research/data/iv_baseline_cache.npz",
    "v8_colab.py": HERE / "research/v8_colab.py",
    "pseudo_cv_colab.py": HERE / "research/pseudo_cv_colab.py",
    "pseudo_candidate_colab.py": HERE / "research/pseudo_candidate_colab.py",
}

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for name in MODULES:
        z.write(HERE / name, f"work/{name}")
    for path in sorted((HERE / "research").glob("*.py")):
        z.write(path, f"work/research/{path.name}")
    z.write(HERE / "research/data/reg_hough.pkl", "work/research/data/reg_hough.pkl")
    for path in sorted((HERE / "Project_2_Dataset").rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            z.write(path, f"work/{path.relative_to(HERE)}")
    for arc, path in ROOT_FILES.items():
        z.write(path, arc)
print(OUT, f"{OUT.stat().st_size / 1e6:.0f} MB")
