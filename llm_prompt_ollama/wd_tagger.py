# ================================================================================
# SmilingWolf WD Tagger (ONNX V3) で画像からタグを生成する
# ================================================================================
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .download import get_models_dir

WD_V3_MODELS: list[tuple[str, str]] = [
    ("wd-swinv2-tagger-v3", "SmilingWolf/wd-swinv2-tagger-v3"),
    ("wd-convnext-tagger-v3", "SmilingWolf/wd-convnext-tagger-v3"),
    ("wd-vit-tagger-v3", "SmilingWolf/wd-vit-tagger-v3"),
    ("wd-vit-large-tagger-v3", "SmilingWolf/wd-vit-large-tagger-v3"),
    ("wd-eva02-large-tagger-v3", "SmilingWolf/wd-eva02-large-tagger-v3"),
]
WD_V3_REPO_IDS = {repo for _, repo in WD_V3_MODELS}
DEFAULT_WD_REPO = "SmilingWolf/wd-swinv2-tagger-v3"

_MODEL_FILENAME = "model.onnx"
_LABEL_FILENAME = "selected_tags.csv"

_session_cache: dict[str, Any] = {}
_tags_cache: dict[str, list[dict[str, Any]]] = {}


# ================================================================================
# WD Tagger キャッシュディレクトリを返す
# ================================================================================
def wd_cache_dir(repo_id: str) -> Path:
    safe = repo_id.replace("/", "__")
    return get_models_dir() / "wd-tagger" / safe


# ================================================================================
# Gradio Dropdown 用の (label, repo_id) を返す
# ================================================================================
def wd_model_choices() -> list[tuple[str, str]]:
    return list(WD_V3_MODELS)


# ================================================================================
# repo_id を V3 許可リストに対して検証する
# ================================================================================
def _require_v3_repo(repo_id: str) -> str:
    rid = (repo_id or "").strip()
    if rid not in WD_V3_REPO_IDS:
        raise ValueError(
            f"Unsupported WD Tagger model: {rid!r}. Only V3 models are allowed."
        )
    return rid


# ================================================================================
# HF から ONNX とタグ CSV を取得する
# ================================================================================
def ensure_wd_model(repo_id: str) -> tuple[Path, Path]:
    rid = _require_v3_repo(repo_id)
    dest = wd_cache_dir(rid)
    dest.mkdir(parents=True, exist_ok=True)
    model_path = dest / _MODEL_FILENAME
    csv_path = dest / _LABEL_FILENAME
    if model_path.is_file() and csv_path.is_file():
        return model_path, csv_path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to download WD Tagger models."
        ) from e

    model_path = Path(
        hf_hub_download(repo_id=rid, filename=_MODEL_FILENAME, local_dir=str(dest))
    )
    csv_path = Path(
        hf_hub_download(repo_id=rid, filename=_LABEL_FILENAME, local_dir=str(dest))
    )
    return model_path.resolve(), csv_path.resolve()


# ================================================================================
# タグ CSV を読み込む
# ================================================================================
def _load_tags(csv_path: Path) -> list[dict[str, Any]]:
    key = str(csv_path)
    if key in _tags_cache:
        return _tags_cache[key]
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "name": str(row.get("name") or ""),
                    "category": int(float(row.get("category") or 0)),
                }
            )
    _tags_cache[key] = rows
    return rows


# ================================================================================
# ONNX セッションを取得する
# ================================================================================
def _get_session(model_path: Path):
    key = str(model_path)
    if key in _session_cache:
        return _session_cache[key]
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime is required for WD Tagger. "
            "Install with: pip install onnxruntime"
        ) from e

    providers = ort.get_available_providers()
    prefer = []
    if "CUDAExecutionProvider" in providers:
        prefer.append("CUDAExecutionProvider")
    prefer.append("CPUExecutionProvider")
    sess = ort.InferenceSession(str(model_path), providers=prefer)
    _session_cache[key] = sess
    return sess


# ================================================================================
# タグ名を SD 向け文字列に整形する
# ================================================================================
def format_tag_names(
    names: list[str],
    *,
    underscore_to_space: bool = True,
    escape_parentheses: bool = True,
) -> str:
    out: list[str] = []
    for raw in names:
        s = raw
        if underscore_to_space:
            s = s.replace("_", " ")
        if escape_parentheses:
            s = s.replace("(", r"\(").replace(")", r"\)")
        out.append(s)
    return ", ".join(out)


# ================================================================================
# PIL 画像を WD Tagger 入力テンソルへ変換する
# ================================================================================
def _prepare_image(image, target_size: int):
    import numpy as np
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGBA")
    canvas = Image.new("RGBA", image.size, (255, 255, 255))
    canvas.alpha_composite(image)
    image = canvas.convert("RGB")

    # Pad to square then resize (SmilingWolf / wd-tagger style).
    max_dim = max(image.size)
    pad = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    pad.paste(image, ((max_dim - image.size[0]) // 2, (max_dim - image.size[1]) // 2))
    image = pad.resize((target_size, target_size), Image.BICUBIC)

    arr = np.asarray(image, dtype=np.float32)
    # BGR order expected by WD models
    arr = arr[:, :, ::-1]
    return np.expand_dims(arr, axis=0)


# ================================================================================
# 画像から Danbooru 風タグ列を生成する
# ================================================================================
def tag_image(
    image,
    *,
    repo_id: str = DEFAULT_WD_REPO,
    general_thresh: float = 0.35,
    character_thresh: float = 0.85,
    exclude_rating: bool = True,
    underscore_to_space: bool = True,
    escape_parentheses: bool = True,
) -> str:
    if image is None:
        raise ValueError("No image provided for WD Tagger.")

    model_path, csv_path = ensure_wd_model(repo_id)
    sess = _get_session(model_path)
    tags = _load_tags(csv_path)

    input_info = sess.get_inputs()[0]
    height = int(input_info.shape[1] or 448)
    tensor = _prepare_image(image, height)
    input_name = input_info.name
    preds = sess.run(None, {input_name: tensor})[0][0]

    general: list[tuple[str, float]] = []
    character: list[tuple[str, float]] = []
    for tag, score in zip(tags, preds):
        name = tag["name"]
        cat = tag["category"]
        conf = float(score)
        if exclude_rating and cat == 9:
            continue
        if cat == 4 and conf >= character_thresh:
            character.append((name, conf))
        elif cat == 0 and conf >= general_thresh:
            general.append((name, conf))

    character.sort(key=lambda x: x[1], reverse=True)
    general.sort(key=lambda x: x[1], reverse=True)
    names = [n for n, _ in character] + [n for n, _ in general]
    return format_tag_names(
        names,
        underscore_to_space=underscore_to_space,
        escape_parentheses=escape_parentheses,
    )
