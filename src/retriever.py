"""Retrieve multimodal documents (text / image / video) from FAISS."""

from pathlib import Path

import faiss
import numpy as np

from config.setting import METADATA_PATH, TOP_K, VECTOR_DB_PATH, VIDEO_QUERY_FRAMES
from src.embedder import MultimodalEmbedder
from src.utils import load_pickle
from src.video_utils import extract_keyframes, resolve_visual_inputs


class MultimodalRetriever:
    def __init__(
        self,
        embedder: MultimodalEmbedder | None = None,
        index_path: str | Path = VECTOR_DB_PATH,
        metadata_path: str | Path = METADATA_PATH,
        top_k: int = TOP_K,
    ):
        self.embedder = embedder or MultimodalEmbedder()
        self.top_k = top_k
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Vector index not found: {self.index_path}. Run scripts/build_index.py first."
            )
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = load_pickle(self.metadata_path)

    def retrieve(
        self,
        text: str | None = None,
        image: str | Path | None = None,
        video: str | Path | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        if self.index is None:
            self.load()
        k = top_k or self.top_k

        frame_paths: list[Path] = []
        if video and Path(video).exists():
            _, frame_paths = resolve_visual_inputs(video_path=video, num_video_frames=VIDEO_QUERY_FRAMES)

        query_vec = self.embedder.encode_multimodal(
            text=text,
            image=image,
            video=video if not frame_paths else None,
            frame_paths=frame_paths or None,
        )
        query_vec = np.expand_dims(query_vec.astype(np.float32), axis=0)
        scores, indices = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = dict(self.metadata[idx])
            doc["score"] = float(score)
            results.append(doc)
        return results
