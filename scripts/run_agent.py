"""CLI: multimodal RAG agent (text + image + video)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import RAGAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal RAG Agent (ảnh + text + video)")
    parser.add_argument("--question", "-q", type=str, required=True)
    parser.add_argument("--image", "-i", type=Path, default=None)
    parser.add_argument("--video", "-v", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    agent = RAGAgent()
    if args.rebuild_index:
        n = agent.rebuild_index()
        print(f"Rebuilt index with {n} documents.")

    out = agent.ask(
        args.question,
        image_path=args.image,
        video_path=args.video,
        top_k=args.top_k,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
