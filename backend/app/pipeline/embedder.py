from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CLIP_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "clip"
CLIP_NAME = "ViT-B/32"


class Embedder:
    def __init__(self, device: str = "cpu", root: Path = CLIP_DIR):
        import clip
        import torch

        self.torch = torch
        self.device = device if device not in ("", None) else "cpu"
        self.model, self.preprocess = clip.load(CLIP_NAME, device=self.device, download_root=str(root))
        self.model.eval()

    def __call__(self, images: list[np.ndarray], batch: int = 16) -> np.ndarray:
        from PIL import Image

        out = []
        for i in range(0, len(images), batch):
            tensors = [self.preprocess(Image.fromarray(cv2.cvtColor(square(im), cv2.COLOR_BGR2RGB)))
                       for im in images[i:i + batch]]
            with self.torch.no_grad():
                emb = self.model.encode_image(self.torch.stack(tensors).to(self.device)).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 512), np.float32)


def square(im: np.ndarray) -> np.ndarray:
    h, w = im.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s, 3), np.uint8)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = im
    return out


def available(root: Path = CLIP_DIR) -> bool:
    return (root / "ViT-B-32.pt").exists()
