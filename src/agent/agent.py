"""
RAG Agent: ảnh + văn bản + video.
Input: question + optional image_path + optional video_path.
Output: text + sources (image, video, frame_paths, text).
"""

from pathlib import Path

from src.rag import MultimodalRAG


class RAGAgent:
    def __init__(self, rag: MultimodalRAG | None = None):
        self.rag = rag or MultimodalRAG()

    def ask(
        self,
        question: str,
        image_path: str | Path | None = None,
        video_path: str | Path | None = None,
        top_k: int | None = None,
    ) -> dict:
        result = self.rag.generate(
            question=question,
            query_image=image_path,
            query_video=video_path,
            top_k=top_k,
        )
        return {
            "response": result["answer"],
            "sources": [
                {
                    "id": d.get("id"),
                    "score": d.get("score"),
                    "media_type": d.get("media_type", "image"),
                    "image_path": d.get("image_path"),
                    "video_path": d.get("video_path"),
                    "frame_paths": d.get("frame_paths"),
                    "text": d.get("text"),
                }
                for d in result["retrieved"]
            ],
            "context": result["context_text"],
            "lvlm_images": result.get("lvlm_images", []),
        }

    def describe_image(self, image_path: str | Path, question: str | None = None) -> dict:
        q = question or "Mô tả chi tiết ảnh và liên hệ với dữ liệu trong kho."
        return self.ask(q, image_path=image_path)

    def describe_video(self, video_path: str | Path, question: str | None = None) -> dict:
        q = question or (
            "Mô tả nội dung video (hành động, bối cảnh, đối tượng) "
            "và so khớp với tài liệu trong kho tri thức."
        )
        return self.ask(q, video_path=video_path)

    def rebuild_index(self, manifest_path: str | Path | None = None) -> int:
        from src.indexer import MultimodalIndexer

        indexer = MultimodalIndexer(embedder=self.rag.retriever.embedder)
        docs = indexer.load_corpus_manifest(manifest_path)
        return indexer.build_and_save(docs)
