"""Trích keyframe từ mọi video trong manifest (cache trước khi index)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.setting import RAW_DATA_DIR
from src.video_utils import extract_keyframes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(RAW_DATA_DIR) / "manifest.json")
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as f:
        docs = json.load(f)

    for doc in docs:
        vp = doc.get("video_path")
        if not vp or not Path(vp).exists():
            continue
        frames = extract_keyframes(vp)
        doc["frame_paths"] = [str(p) for p in frames]
        print(f"{doc.get('id')}: {len(frames)} frames")

    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print(f"Updated {args.manifest}")


if __name__ == "__main__":
    main()
