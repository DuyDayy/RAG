"""Multimodal RAG: text + image + video -> retrieve -> LVLM (frames as images)."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.setting import (
    LVLM_MAX_FRAMES_PER_DOC,
    LVLM_MODEL_NAME,
    SYSTEM_PROMPT,
    VIDEO_QUERY_FRAMES,
)
from src.retriever import MultimodalRetriever
from src.video_utils import extract_keyframes, resolve_visual_inputs


class MultimodalRAG:
    """
    LVLM (Qwen-VL) nhận ảnh/frame; video được biểu diễn bằng keyframe.
    Input: text, image, video (bất kỳ tổ hợp). Output: text + nguồn ảnh/video/frame.
    """

    def __init__(
        self,
        model_name: str = LVLM_MODEL_NAME,
        retriever: MultimodalRetriever | None = None,
    ):
        self.retriever = retriever or MultimodalRetriever()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).eval()

    def _doc_visual_paths(self, doc: dict, max_frames: int = LVLM_MAX_FRAMES_PER_DOC) -> list[str]:
        paths: list[str] = []
        if doc.get("image_path") and Path(doc["image_path"]).exists():
            paths.append(str(doc["image_path"]))
        cached = doc.get("frame_paths") or []
        for fp in cached:
            if Path(fp).exists() and str(fp) not in paths:
                paths.append(str(fp))
        if len(paths) < max_frames and doc.get("video_path") and Path(doc["video_path"]).exists():
            for fp in extract_keyframes(doc["video_path"], num_frames=max_frames):
                s = str(fp)
                if s not in paths:
                    paths.append(s)
        return paths[:max_frames]

    def _format_context(self, retrieved: list[dict]) -> tuple[list[dict], str]:
        context_lines = []
        for i, doc in enumerate(retrieved, 1):
            line = f"[Ngữ cảnh {i}] (điểm={doc.get('score', 0):.3f}, loại={doc.get('media_type', '?')})"
            if doc.get("text"):
                line += f"\nVăn bản: {doc['text']}"
            if doc.get("image_path"):
                line += f"\nẢnh: {doc['image_path']}"
            if doc.get("video_path"):
                line += f"\nVideo: {doc['video_path']}"
            if doc.get("frame_paths"):
                line += f"\nKhung hình ({len(doc['frame_paths'])}): " + ", ".join(doc["frame_paths"][:3])
                if len(doc["frame_paths"]) > 3:
                    line += " ..."
            context_lines.append(line)
        return retrieved, "\n\n".join(context_lines)

    def _collect_lvlm_images(
        self,
        retrieved: list[dict],
        user_image: str | Path | None,
        user_video: str | Path | None,
    ) -> list[str]:
        images: list[str] = []
        seen: set[str] = set()

        for doc in retrieved:
            for p in self._doc_visual_paths(doc):
                if p not in seen:
                    images.append(p)
                    seen.add(p)

        _, query_frames = resolve_visual_inputs(
            image_path=user_image,
            video_path=user_video,
            num_video_frames=VIDEO_QUERY_FRAMES,
        )
        for p in query_frames:
            s = str(p)
            if s not in seen:
                images.append(s)
                seen.add(s)

        return images

    def _build_lvlm_query(
        self,
        user_text: str,
        user_image: str | Path | None,
        user_video: str | Path | None,
        retrieved: list[dict],
        context_text: str,
        lvlm_images: list[str],
    ) -> list[dict]:
        parts: list[dict] = []
        for img in lvlm_images:
            if Path(img).exists():
                parts.append({"image": img})
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"--- Ngữ cảnh đã truy xuất (ảnh / văn bản / video qua khung hình) ---\n"
            f"{context_text}\n\n"
            f"--- Câu hỏi ---\n{user_text}\n\n"
            "Trả lời dựa trên ngữ cảnh và các khung hình đính kèm. "
            "Với nguồn video, suy luận từ chuỗi frame. Nếu thiếu thông tin, nói rõ."
        )
        parts.append({"text": prompt})
        return parts

    @torch.no_grad()
    def generate(
        self,
        question: str,
        query_image: str | Path | None = None,
        query_video: str | Path | None = None,
        top_k: int | None = None,
    ) -> dict:
        retrieved = self.retriever.retrieve(
            text=question,
            image=query_image,
            video=query_video,
            top_k=top_k,
        )
        retrieved, context_text = self._format_context(retrieved)
        lvlm_images = self._collect_lvlm_images(retrieved, query_image, query_video)
        query = self._build_lvlm_query(
            question, query_image, query_video, retrieved, context_text, lvlm_images
        )
        formatted = self.tokenizer.from_list_format(query)
        response, _ = self.model.chat(self.tokenizer, query=formatted, history=None)
        return {
            "answer": response,
            "retrieved": retrieved,
            "context_text": context_text,
            "lvlm_images": lvlm_images,
        }
