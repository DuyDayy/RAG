"""
Fine-tune CLIP embedding on image-text pairs (contrastive learning).
Input: data/raw/train_pairs.json
Output: checkpoints/embedding_finetuned/
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.setting import DEVICE, EMBEDDING_FINETUNED_DIR, EMBEDDING_MODEL, RAW_DATA_DIR
from src.embedder import MultimodalEmbedder
from src.utils import load_image
from src.video_utils import extract_keyframes


class ImageTextPairDataset(Dataset):
    def __init__(self, pairs: list[dict], root: Path):
        self.pairs = pairs
        self.root = root

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[str, Image.Image]:
        item = self.pairs[idx]
        if item.get("image_path"):
            img_path = (
                self.root / item["image_path"]
                if not Path(item["image_path"]).is_absolute()
                else Path(item["image_path"])
            )
            return item["text"], load_image(img_path)
        if item.get("video_path"):
            vp = (
                self.root / item["video_path"]
                if not Path(item["video_path"]).is_absolute()
                else Path(item["video_path"])
            )
            frame = extract_keyframes(vp, num_frames=1)[0]
            return item["text"], load_image(frame)
        raise KeyError("Pair needs image_path or video_path")


def contrastive_loss(text_emb, image_emb, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE between batch of text and image embeddings."""
    logits = (text_emb @ image_emb.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_t = torch.nn.functional.cross_entropy(logits, labels)
    loss_i = torch.nn.functional.cross_entropy(logits.T, labels)
    return (loss_t + loss_i) / 2


def train(
    pairs_path: Path,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-5,
    output_dir: Path | None = None,
) -> None:
    output_dir = output_dir or Path(EMBEDDING_FINETUNED_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pairs_path, encoding="utf-8") as f:
        pairs = json.load(f)

    dataset = ImageTextPairDataset(pairs, ROOT)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    embedder = MultimodalEmbedder()
    model = embedder.model
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for texts, images in loader:
            text_features = model.encode(
                list(texts),
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            image_features = model.encode(
                list(images),
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            loss = contrastive_loss(text_features, image_features)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs} loss={total_loss / max(len(loader), 1):.4f}")

    embedder.save(output_dir)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    print(f"Saved fine-tuned embedder to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune multimodal embedding (CLIP)")
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path(RAW_DATA_DIR) / "train_pairs.json",
        help="JSON list of {image_path, text}",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.pairs.exists():
        print(f"Missing {args.pairs}. Create train_pairs.json before training.")
        sys.exit(1)
    train(args.pairs, args.epochs, args.batch_size, args.lr, args.output)
