"""Build FAISS vector index from multimodal corpus."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.setting import EMBEDDING_FINETUNED_DIR
from src.embedder import MultimodalEmbedder
from src.indexer import MultimodalIndexer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--use-finetuned", action="store_true")
    args = parser.parse_args()

    embedder = MultimodalEmbedder()
    finetuned = Path(EMBEDDING_FINETUNED_DIR)
    if args.use_finetuned and finetuned.exists():
        from sentence_transformers import SentenceTransformer
        embedder.model = SentenceTransformer(str(finetuned), device=embedder.device)

    indexer = MultimodalIndexer(embedder=embedder)
    docs = indexer.load_corpus_manifest(args.manifest)
    n = indexer.build_and_save(docs)
    print(f"Indexed {n} documents.")


if __name__ == "__main__":
    main()
