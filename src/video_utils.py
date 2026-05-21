"""Trích keyframe từ video phục vụ embedding (CLIP) và LVLM (Qwen-VL nhận ảnh)."""

from pathlib import Path

import numpy as np
from PIL import Image

from config.setting import FRAMES_CACHE_DIR, VIDEO_EXTENSIONS, VIDEO_QUERY_FRAMES, VIDEO_SAMPLE_FRAMES
from src.utils import ensure_dir, load_image

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def extract_keyframes(
    video_path: str | Path,
    num_frames: int = VIDEO_SAMPLE_FRAMES,
    cache_dir: str | Path | None = FRAMES_CACHE_DIR,
) -> list[Path]:
    """
    Lấy num_frames frame đều theo thời gian (OpenCV).
    Cache ra disk để tái dùng khi index / RAG.
    """
    if not HAS_CV2:
        raise ImportError("Cần opencv-python: pip install opencv-python-headless")

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    cache_root = ensure_dir(cache_dir or FRAMES_CACHE_DIR)
    out_dir = cache_root / f"{video_path.stem}_{abs(hash(str(video_path.resolve()))) % 10**8}"
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        raise RuntimeError(f"No frames in video: {video_path}")

    indices = np.linspace(0, max(total - 1, 0), num=min(num_frames, total), dtype=int)
    frame_paths: list[Path] = []

    for i, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_path = out_dir / f"frame_{i:03d}.jpg"
        Image.fromarray(frame_rgb).save(out_path, quality=90)
        frame_paths.append(out_path)

    cap.release()
    if not frame_paths:
        raise RuntimeError(f"Failed to extract frames from {video_path}")
    return frame_paths


def load_frame_images(frame_paths: list[str | Path]) -> list[Image.Image]:
    return [load_image(p) for p in frame_paths]


def resolve_visual_inputs(
    image_path: str | Path | None = None,
    video_path: str | Path | None = None,
    num_video_frames: int = VIDEO_SAMPLE_FRAMES,
) -> tuple[list[Path], list[Path]]:
    """
    Trả về (images_for_lvlm, frame_paths_used_for_embedding).
    image_path: ảnh tĩnh; video_path: trích frame.
    """
    images: list[Path] = []
    frames: list[Path] = []

    if image_path and Path(image_path).exists():
        p = Path(image_path)
        images.append(p)
        frames.append(p)

    if video_path and Path(video_path).exists():
        vframes = extract_keyframes(video_path, num_frames=num_video_frames)
        frames.extend(vframes)
        images.extend(vframes[:VIDEO_QUERY_FRAMES])

    return images, frames
