"""Small supervised U-Net for the two microscopy modalities."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from cellmatch import ROOT, normalize, read_image, rle_to_labels


class ConvBlock(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, 3, padding=1),
            torch.nn.GroupNorm(4, output_channels),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(output_channels, output_channels, 3, padding=1),
            torch.nn.GroupNorm(4, output_channels),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(torch.nn.Module):
    def __init__(self, width: int = 16):
        super().__init__()
        self.down1 = ConvBlock(2, width)
        self.down2 = ConvBlock(width, width * 2)
        self.down3 = ConvBlock(width * 2, width * 4)
        self.down4 = ConvBlock(width * 4, width * 8)
        self.middle = ConvBlock(width * 8, width * 12)
        self.up4 = ConvBlock(width * 20, width * 8)
        self.up3 = ConvBlock(width * 12, width * 4)
        self.up2 = ConvBlock(width * 6, width * 2)
        self.up1 = ConvBlock(width * 3, width)
        self.output = torch.nn.Conv2d(width, 2, 1)
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in (self.down1, self.down2, self.down3, self.down4):
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.middle(x)
        for block, skip in zip((self.up4, self.up3, self.up2, self.up1), reversed(skips)):
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return self.output(x)


def core_target(labels: np.ndarray) -> np.ndarray:
    """Erode instances separately, leaving a seed region for every cell."""
    positive = labels > 0
    result = positive.copy()
    for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        result &= labels == np.roll(labels, shift, axis=(0, 1))
    result[:2] = False
    result[-2:] = False
    result[:, :2] = False
    result[:, -2:] = False
    return result


def resize_features(features: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.stack([cv2.resize(channel, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
                     for channel in features.astype(np.float32)])


def load_training(modality: str, subjects: list[str] | None = None, scale: float = 1.0):
    frame = pd.read_csv(ROOT / "training/train_ground_truth.csv")
    records = []
    field = f"{modality}_instances"
    for row in frame.itertuples(index=False):
        subject = row.sample_id.split("__")[0]
        if subjects is not None and subject not in subjects:
            continue
        region = ROOT / "training" / row.sample_id.replace("__", "/")
        image = read_image(region / f"{modality}.tif")
        labels, _ = rle_to_labels(json.loads(getattr(row, field)), image.shape)
        features = normalize(image, modality)
        if scale != 1.0:
            # Small somata get more pixels; the network sees them at a friendlier size.
            shape = (round(image.shape[0] * scale), round(image.shape[1] * scale))
            features = resize_features(features, shape)
            labels = cv2.resize(labels, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        features = features.astype(np.float16)
        core = core_target(labels)
        targets = np.stack((labels > 0, core), axis=0).astype(np.uint8)
        centers = ndi.center_of_mass(np.ones(labels.shape, np.uint8), labels, np.arange(1, labels.max() + 1))
        centers = np.asarray(centers, dtype=np.float32)
        records.append((row.sample_id, features, targets, centers))
    return records


class RandomPatches(torch.utils.data.Dataset):
    def __init__(self, records, patch: int = 128, length: int = 10000):
        self.records, self.patch, self.length = records, patch, length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        _, features, targets, centers = self.records[np.random.randint(len(self.records))]
        height, width = features.shape[-2:]
        if len(centers) and np.random.random() < 0.7:
            cy, cx = centers[np.random.randint(len(centers))]
            cy += np.random.randint(-self.patch // 3, self.patch // 3 + 1)
            cx += np.random.randint(-self.patch // 3, self.patch // 3 + 1)
        else:
            cy, cx = np.random.randint(height), np.random.randint(width)
        y = int(np.clip(cy - self.patch // 2, 0, height - self.patch))
        x = int(np.clip(cx - self.patch // 2, 0, width - self.patch))
        image = features[:, y : y + self.patch, x : x + self.patch].astype(np.float32)
        target = targets[:, y : y + self.patch, x : x + self.patch].astype(np.float32)
        k = np.random.randint(4)
        image = np.rot90(image, k, axes=(1, 2))
        target = np.rot90(target, k, axes=(1, 2))
        if np.random.random() < 0.5:
            image = image[:, :, ::-1]
            target = target[:, :, ::-1]
        if np.random.random() < 0.5:
            image = image[:, ::-1]
            target = target[:, ::-1]
        gain = np.random.uniform(0.75, 1.35)
        gamma = np.random.uniform(0.75, 1.3)
        image = np.clip(image * gain, 0, 1) ** gamma
        if np.random.random() < 0.25:
            sigma = np.random.uniform(0.3, 0.8)
            image = np.stack([ndi.gaussian_filter(ch, sigma) for ch in image])
        image = np.clip(image + np.random.normal(0, np.random.uniform(0, 0.025), image.shape), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)), torch.from_numpy(
            np.ascontiguousarray(target, dtype=np.float32)
        )


def loss_function(logits: torch.Tensor, targets: torch.Tensor, modality: str) -> torch.Tensor:
    pos_weight = torch.tensor([3.0, 5.0] if modality == "invivo" else [6.0, 8.0], device=logits.device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight[None, :, None, None]
    )
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum((0, 2, 3))
    dice = 1 - (2 * intersection + 1) / (probs.sum((0, 2, 3)) + targets.sum((0, 2, 3)) + 1)
    return bce + dice.mean()


@torch.inference_mode()
def predict_probabilities(model: UNet, image: np.ndarray, modality: str, device: str = "cpu",
                          tile: int = 256, overlap: int = 32, tta: bool = False,
                          scale: float = 1.0) -> np.ndarray:
    features = normalize(image, modality)
    if scale != 1.0:
        features = resize_features(features, (round(image.shape[0] * scale),
                                              round(image.shape[1] * scale)))
    height, width = features.shape[1:]
    output = np.zeros((2, height, width), np.float32)
    weights = np.zeros((height, width), np.float32)
    stride = tile - overlap
    ys = list(range(0, max(height - tile, 0), stride)) + [max(height - tile, 0)]
    xs = list(range(0, max(width - tile, 0), stride)) + [max(width - tile, 0)]
    window = np.ones((tile, tile), np.float32)
    edge = np.linspace(0.15, 1, overlap, dtype=np.float32)
    window[:overlap] *= edge[:, None]
    window[-overlap:] *= edge[::-1, None]
    window[:, :overlap] *= edge[None, :]
    window[:, -overlap:] *= edge[None, ::-1]
    model.eval()
    for y in ys:
        for x in xs:
            crop = features[:, y : y + tile, x : x + tile]
            crop = np.pad(crop, ((0, 0), (0, tile - crop.shape[1]), (0, tile - crop.shape[2])), mode="reflect")
            tensor = torch.from_numpy(crop[None]).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                if tta:
                    batch = torch.cat((tensor, tensor.flip(-1), tensor.flip(-2), tensor.transpose(-1, -2)))
                    prediction = torch.sigmoid(model(batch).float())
                    prob = torch.stack((prediction[0], prediction[1].flip(-1),
                                        prediction[2].flip(-2), prediction[3].transpose(-1, -2))).mean(0).cpu().numpy()
                else:
                    prob = torch.sigmoid(model(tensor).float())[0].cpu().numpy()
            h, w = min(tile, height - y), min(tile, width - x)
            output[:, y : y + h, x : x + w] += prob[:, :h, :w] * window[:h, :w]
            weights[y : y + h, x : x + w] += window[:h, :w]
    output /= weights[None]
    if scale != 1.0:
        output = resize_features(output, image.shape)
    return output


def decode_instances(probs: np.ndarray, modality: str, body_threshold: float = 0.5,
                     core_threshold: float = 0.5) -> np.ndarray:
    """Marker watershed on the body/core maps, with the hard size gates."""
    body = probs[0] >= body_threshold
    core = (probs[1] >= core_threshold) & body
    core = ndi.binary_opening(core, iterations=1)
    markers, count = ndi.label(core)
    sizes = np.bincount(markers.ravel())
    markers[sizes[markers] < 4] = 0
    # Guarantee a marker for a confidently detected cell with no core peak.
    components, component_count = ndi.label(body)
    covered = np.bincount(components[markers > 0], minlength=component_count + 1)
    candidates = np.flatnonzero((covered == 0) & (np.bincount(components.ravel()) >= 15))
    for component in candidates[candidates > 0]:
        location = np.argmax(np.where(components == component, probs[1], -1))
        markers.flat[location] = count + component
    labels = watershed(-probs[1], markers, mask=body).astype(np.int32)
    counts = np.bincount(labels.ravel())
    low, high = (20, 330) if modality == "invivo" else (17, 350)
    keep = (counts >= low) & (counts <= high)
    keep[0] = False
    labels[~keep[labels]] = 0
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32)


def instance_features(labels: np.ndarray, probs: np.ndarray | None = None) -> np.ndarray:
    """Per-instance (core mean, circularity, eccentricity, bbox fill), rows in label order.

    Without probabilities the core-mean column is zero.
    """
    from skimage.measure import regionprops

    features = np.zeros((int(labels.max()), 4), np.float32)
    for prop in regionprops(labels, intensity_image=None if probs is None else probs[1]):
        perimeter = max(prop.perimeter, 1.0)
        y0, x0, y1, x1 = prop.bbox
        features[prop.label - 1] = (0.0 if probs is None else prop.intensity_mean,
                                    4 * np.pi * prop.area / perimeter ** 2,
                                    prop.eccentricity, prop.area / max((y1 - y0) * (x1 - x0), 1))
    return features


def filter_instances(labels: np.ndarray, features: np.ndarray, min_core: float = 0.0,
                     min_circularity: float = 0.0, max_eccentricity: float = 1.0,
                     min_fill: float = 0.0) -> np.ndarray:
    """Drop non-compact blobs (bright axons/dendrites) and weak detections."""
    keep = np.r_[False, (features[:, 0] >= min_core) & (features[:, 1] >= min_circularity)
                 & (features[:, 2] <= max_eccentricity) & (features[:, 3] >= min_fill)]
    labels = np.where(keep[labels], labels, 0)
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(labels.shape).astype(np.int32)


def probabilities_to_labels(probs: np.ndarray, modality: str, body_threshold: float = 0.5,
                            core_threshold: float = 0.5, **filters) -> np.ndarray:
    labels = decode_instances(probs, modality, body_threshold, core_threshold)
    if not filters or not labels.max():
        return labels
    return filter_instances(labels, instance_features(labels, probs), **filters)


def load_model(path: Path, device: str = "cpu") -> UNet:
    state = torch.load(path, map_location=device, weights_only=True)
    model = UNet(width=state["down1.layers.0.weight"].shape[0])
    model.load_state_dict(state)
    return model.to(device).eval()


