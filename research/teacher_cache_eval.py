"""Evaluate already-cached, untrained Cellpose-SAM teacher masks by mouse."""
from pathlib import Path
import sys

import numpy as np

WORK = Path('/content/work')
sys.path.insert(0, str(WORK))
import pipeline as P  # noqa: E402
from cellmatch import pq_score  # noqa: E402


def main():
    truth = P.load_truth()
    by_subject = {}
    for sid, row in truth.items():
        path = WORK / 'pseudo_teacher' / f'{sid}.npz'
        if not path.exists():
            continue
        with np.load(path) as packed:
            labels = packed['labels'].astype(np.int32)
        pq = pq_score(labels, row['exvivo'][0])[0]
        by_subject.setdefault(row['subject'], []).append(pq)
    for subject, values in by_subject.items():
        print(subject, 'regions', len(values), 'teacher_PQ',
              float(np.mean(values)), flush=True)


if __name__ == '__main__':
    main()
