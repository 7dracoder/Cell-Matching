"""Pack only the held-out in-vivo masks/probabilities needed by Colab CV."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
source = ROOT / 'data/v7/heldout_labels.npz'
target = ROOT / 'data/iv_baseline_cache.npz'
with np.load(source) as old:
    values = {key: old[key] for key in old.files
              if key.endswith('|invivo') or key.endswith('|invivo|prob')}
np.savez_compressed(target, **values)
print(target, target.stat().st_size, 'bytes', len(values), 'arrays')
