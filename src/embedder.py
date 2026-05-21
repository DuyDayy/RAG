"""Multimodal embedder: text + image + video(frames) -> shared vector space (CLIP)."""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

from config.setting import DEVICE, EMBEDDING_MODEL, VIDEO_SAMPLE_FRAMES
from src.utils import load_image
from src.video_utils import extract_keyframes, is_video_file


class MultimodalEmbedder:
    """
    Encode text, images, and video (via keyframes) into the same embedding space.
    Video -> average of frame embeddings, then fused with text/image.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str | None = None,
    ):
        self.device = device or DEVICE
        self.model = SentenceTransformer(model_name, device=self.device)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode_text(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        emb = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_image(
        self,
        images: list[str | Path | Image.Image],
        normalize: bool = True,
    ) -> np.ndarray:
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_images.append(load_image(img))
            else:
                pil_images.append(img.convert("RGB"))
        emb = self.model.encode(
            pil_images,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_video(
        self,
        video_path: str | Path,
        num_frames: int = VIDEO_SAMPLE_FRAMES,
        normalize: bool = True,
    ) -> np.ndarray:
        """Một vector cho cả video = trung bình embedding các keyframe."""
        frames = extract_keyframes(video_path, num_frames=num_frames)
        frame_emb = self.encode_image(frames, normalize=False)
        v = frame_emb.mean(axis=0)
        if normalize:
            v = v / (np.linalg.norm(v) + 1e-8)
        return v.astype(np.float32)

    def _fuse_vectors(
        self,
        parts: list[np.ndarray],
        fusion: Literal["average", "concat"] = "average",
        normalize: bool = True,
    ) -> np.ndarray:
        if not parts:
            raise ValueError("No vectors to fuse.")
        if len(parts) == 1:
            v = parts[0]
        elif fusion == "concat":
            v = np.concatenate(parts)
        else:
            v = np.mean(parts, axis=0)
        if normalize:
            v = v / (np.linalg.norm(v) + 1e-8)
        return v.astype(np.float32)

    def encode_multimodal(
        self,
        text: str | None = None,
        image: str | Path | Image.Image | None = None,
        video: str | Path | None = None,
        frame_paths: list[str | Path] | None = None,
        fusion: Literal["text", "image", "video", "concat", "average"] = "average",
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Fuse text + image + video (ảnh/văn bản/video — có thể kết hợp bất kỳ tập con).
        frame_paths: dùng khi đã trích frame sẵn (tránh trích lại).
        """
        if text is None and image is None and video is None and not frame_paths:
            raise ValueError("Cần ít nhất một trong: text, image, video, frame_paths.")

        if fusion in ("text", "image", "video"):
            if fusion == "text" and text:
                return self.encode_text([text], normalize=normalize)[0]
            if fusion == "image" and image:
                return self.encode_image([image], normalize=normalize)[0]
            if fusion == "video" and video:
                return self.encode_video(video, normalize=normalize)

        parts: list[np.ndarray] = []
        if text and text.strip():
            parts.append(self.encode_text([text], normalize=False)[0])
        if image is not None:
            parts.append(self.encode_image([image], normalize=False)[0])
        if frame_paths:
            parts.append(self.encode_image(frame_paths, normalize=False).mean(axis=0))
        elif video is not None:
            if is_video_file(video):
                parts.append(self.encode_video(video, normalize=False))
            elif Path(video).exists():
                parts.append(self.encode_image([video], normalize=False)[0])

        return self._fuse_vectors(parts, fusion="concat" if fusion == "concat" else "average", normalize=normalize)

    def encode_query(
        self,
        text: str | None = None,
        image: str | Path | Image.Image | None = None,
        video: str | Path | None = None,
        fusion: Literal["text", "image", "video", "concat", "average"] = "average",
        normalize: bool = True,
    ) -> np.ndarray:
        return self.encode_multimodal(text=text, image=image, video=video, fusion=fusion, normalize=normalize)

    def load_finetuned(self, checkpoint_path: str | Path) -> None:
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=False)

    def save(self, path: str | Path) -> None:
        self.model.save(str(path))
