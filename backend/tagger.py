"""打标推理。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image, ImageOps

# 保留下划线
PRESERVE_UNDERSCORE_PREFIXES = ("score_",)


def normalize_tag(tag: str) -> str:
    """规范标签。"""
    tag = tag.strip().lower()
    if tag.startswith(PRESERVE_UNDERSCORE_PREFIXES):
        return tag
    return tag.replace("_", " ")


class Tagger:
    """ONNX 打标器。"""

    def __init__(self, model_dir: str | Path, use_cuda: bool = False) -> None:
        model_dir = Path(model_dir)
        model_path = model_dir / "model.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        tags_path = model_dir / "selected_tags.csv"
        if not tags_path.exists():
            raise FileNotFoundError(f"标签文件未找到: {tags_path}")

        providers: list[str] = []
        if use_cuda and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.providers = providers
        self.session = ort.InferenceSession(str(model_path), providers=providers)

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.size = int(shape[1] if isinstance(shape[1], int) else 448)

        tags_df = pd.read_csv(tags_path)
        required_columns = {"name", "category"}
        missing_columns = required_columns - set(tags_df.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"标签文件缺少必要列: {missing}")
        self.names = tags_df["name"].tolist()
        self.categories = [int(c) for c in tags_df["category"].tolist()]

    def infer(self, image_path: str | Path) -> np.ndarray:
        """推理。"""
        with Image.open(image_path) as src:
            image = src.convert("RGBA")
        # 透明转白底
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        image = background.convert("RGB")

        # 方图缩放
        max_side = max(image.size)
        padded = ImageOps.pad(
            image,
            (max_side, max_side),
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255),
        )
        resized = padded.resize((self.size, self.size), Image.Resampling.LANCZOS)

        array = np.asarray(resized, dtype=np.float32)
        # NHWC + BGR
        array = array[:, :, ::-1]
        array = np.expand_dims(array, axis=0)
        probs = self.session.run(None, {self.input_name: array})[0][0]
        probs = probs.astype(float)
        if len(probs) != len(self.names):
            raise ValueError(
                f"模型输出标签数量不匹配: 输出 {len(probs)}，标签表 {len(self.names)}"
            )
        return probs

    def tag_image(
        self,
        image_path: str | Path,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        include_character: bool = False,
        normalize: bool = True,
    ) -> list[dict]:
        """输出标签。"""
        probs = self.infer(image_path)
        results: list[dict] = []
        for name, category, prob in zip(self.names, self.categories, probs):
            if category == 9:
                continue
            if category == 4:
                if include_character and prob >= character_threshold:
                    results.append({"name": name, "category": category, "prob": float(prob)})
            elif prob >= general_threshold:
                results.append({"name": name, "category": category, "prob": float(prob)})
        results.sort(key=lambda x: x["prob"], reverse=True)
        if normalize:
            for r in results:
                r["name"] = normalize_tag(r["name"])
        return results

    def tag_candidates(self, image_path: str | Path, top_n: int = 60) -> list[dict]:
        """候选标签。"""
        probs = self.infer(image_path)
        results: list[dict] = []
        for name, category, prob in zip(self.names, self.categories, probs):
            if category == 9:
                continue
            results.append({
                "name": normalize_tag(name),
                "category": category,
                "prob": float(prob),
            })
        results.sort(key=lambda x: x["prob"], reverse=True)
        return results[:top_n]
