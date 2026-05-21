"""Build FAISS index from multimodal corpus (image + text + video)."""

import json
from pathlib import Path

import faiss
import numpy as np

from config.setting import METADATA_PATH, RAW_DATA_DIR, VECTOR_DB_PATH, VIDEO_EXTENSIONS
from src.embedder import MultimodalEmbedder
from src.utils import ensure_dir, save_pickle
from src.video_utils import extract_keyframes, is_video_file


class MultimodalIndexer:
    """
    Mỗi document có thể gồm:
    - text, image_path, video_path (bất kỳ tổ hợp)
    - media_type: image | video | multimodal
    - frame_paths: keyframe đã trích (lưu trong metadata cho LVLM)
    """

    def __init__(self, embedder: MultimodalEmbedder | None = None):
        self.embedder = embedder or MultimodalEmbedder()

    def load_corpus_manifest(self, manifest_path: str | Path | None = None) -> list[dict]:
        manifest_path = manifest_path or Path(RAW_DATA_DIR) / "manifest.json"
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            return self._scan_raw_folder()
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)

    def _scan_raw_folder(self) -> list[dict]:
        raw = Path(RAW_DATA_DIR)
        docs = []
        idx = 0
        for path in sorted(raw.rglob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                docs.append(
                    {
                        "id": f"auto_img_{idx}",
                        "image_path": str(path),
                        "text": "",
                        "media_type": "image",
                    }
                )
                idx += 1
            elif path.suffix.lower() in VIDEO_EXTENSIONS:
                docs.append(
                    {
                        "id": f"auto_vid_{idx}",
                        "video_path": str(path),
                        "text": "",
                        "media_type": "video",
                    }
                )
                idx += 1
        return docs

    def _prepare_doc_visuals(self, doc: dict) -> tuple[list[Path], str | None]:
        """Trích frame nếu có video; trả frame_paths + media hint."""
        frame_paths: list[Path] = []
        image_path = doc.get("image_path")
        video_path = doc.get("video_path")

        if video_path and Path(video_path).exists():
            frame_paths = extract_keyframes(video_path)
            doc["frame_paths"] = [str(p) for p in frame_paths]
            if not doc.get("media_type"):
                doc["media_type"] = "video"
        if image_path and Path(image_path).exists():
            frame_paths.insert(0, Path(image_path))
            if doc.get("media_type") == "video" and image_path:
                doc["media_type"] = "multimodal"
            elif not doc.get("media_type"):
                doc["media_type"] = "image"

        return frame_paths, doc.get("media_type")

    def build_document_vectors(self, documents: list[dict]) -> tuple[np.ndarray, list[dict]]:
        vectors = []
        metadata = []
        for doc in documents:
            doc = dict(doc)
            text = doc.get("text") or ""
            image_path = doc.get("image_path")
            video_path = doc.get("video_path")
            frame_paths, _ = self._prepare_doc_visuals(doc)

            try:
                use_image = image_path if image_path and Path(image_path).exists() else None
                if frame_paths and use_image:
                    if str(use_image) in [str(p) for p in frame_paths]:
                        use_image = None
                if video_path and Path(video_path).exists() and is_video_file(video_path):
                    vec = self.embedder.encode_multimodal(
                        text=text or None,
                        image=use_image,
                        frame_paths=frame_paths or None,
                        video=video_path if not frame_paths else None,
                    )
                elif image_path and Path(image_path).exists():
                    vec = self.embedder.encode_multimodal(
                        text=text or None,
                        image=image_path,
                    )
                elif text.strip():
                    vec = self.embedder.encode_text([text])[0]
                else:
                    continue
            except (FileNotFoundError, RuntimeError, ImportError) as e:
                print(f"Skip doc {doc.get('id')}: {e}")
                continue

            vectors.append(vec)
            metadata.append(doc)

        if not vectors:
            raise ValueError("No valid documents to index.")
        return np.vstack(vectors).astype(np.float32), metadata

    def build_and_save(
        self,
        documents: list[dict] | None = None,
        index_path: str | Path = VECTOR_DB_PATH,
        metadata_path: str | Path = METADATA_PATH,
    ) -> int:
        documents = documents or self.load_corpus_manifest()
        vectors, metadata = self.build_document_vectors(documents)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        ensure_dir(Path(index_path).parent)
        faiss.write_index(index, str(index_path))
        save_pickle(metadata, metadata_path)
        return len(metadata)
