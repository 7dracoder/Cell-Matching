"""Train the supervised cell mask model.

Examples:
  python train_model.py --modality invivo --steps 1400 --output models/invivo.pt
  python train_model.py --modality exvivo --steps 1800 --output models/exvivo.pt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cellmatch import ROOT, pq_score, read_image, rle_to_labels
from learned import RandomPatches, UNet, load_training, loss_function, predict_probabilities, probabilities_to_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["invivo", "exvivo"], required=True)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-subject", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    subjects = sorted({p.name for p in (ROOT / "training").glob("subject_*")})
    train_subjects = [s for s in subjects if s != args.val_subject]
    print(f"device={device} training_subjects={train_subjects} modality={args.modality}", flush=True)
    records = load_training(args.modality, train_subjects)
    print(f"loaded {len(records)} training regions", flush=True)
    dataset = RandomPatches(records, length=args.steps * args.batch)
    workers = 2 if device == "cuda" else 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=workers,
                                         pin_memory=device == "cuda", persistent_workers=bool(workers))
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=3e-5)
    started = time.monotonic()
    model.train()
    for step, (images, masks) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, masks, args.modality)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step}/{args.steps}: loss={loss.item():.4f}, elapsed={time.monotonic()-started:.0f}s", flush=True)
        if step >= args.steps:
            break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), args.output)
    print(f"saved {args.output}", flush=True)
    if args.val_subject:
        model = model.to(device).eval()
        frame = pd.read_csv(ROOT / "training/train_ground_truth.csv")
        frame = frame[frame.sample_id.str.startswith(args.val_subject)]
        for row in list(frame.itertuples(index=False))[:3]:
            path = ROOT / "training" / row.sample_id.replace("__", "/") / f"{args.modality}.tif"
            image = read_image(path)
            truth, _ = rle_to_labels(json.loads(getattr(row, f"{args.modality}_instances")), image.shape)
            probs = predict_probabilities(model, image, args.modality, device)
            for threshold in (0.4, 0.5, 0.6):
                labels = probabilities_to_labels(probs, args.modality, threshold, 0.5)
                print(row.sample_id, threshold, pq_score(labels, truth), flush=True)


if __name__ == "__main__":
    main()
