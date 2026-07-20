# ================================================================================
# 拡張ルートの models.json から GGUF モデルカタログを読み込む
# ================================================================================
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_REQUIRED_KEYS = ("id", "label", "ollama_name", "hf_repo", "hf_file")


# ================================================================================
# 拡張機能のルートディレクトリを返す
# ================================================================================
def _extension_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ================================================================================
# models.json のパスを返す
# ================================================================================
def models_json_path() -> Path:
    return _extension_root() / "models.json"


# ================================================================================
# 生の辞書エントリを正規化する（必須キー欠落時は None）
# ================================================================================
def _normalize_entry(raw: dict[str, Any]) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    out: dict[str, str] = {}
    for key in _REQUIRED_KEYS:
        val = raw.get(key)
        if val is None or str(val).strip() == "":
            return None
        out[key] = str(val).strip()
    # Optional multimodal projector file in the same hf_repo.
    mmproj = str(raw.get("hf_mmproj") or "").strip()
    if mmproj:
        out["hf_mmproj"] = mmproj
    modality = str(raw.get("modality") or "").strip().lower()
    if modality in ("text", "vision"):
        out["modality"] = modality
    elif mmproj:
        out["modality"] = "vision"
    else:
        out["modality"] = "text"
    return out


# ================================================================================
# models.json 欠落・不正時の空カタログを返す
# ================================================================================
def _empty_catalog() -> dict[str, Any]:
    return {"default": "", "models": []}


# ================================================================================
# models.json を読み込みキャッシュする
# ================================================================================
@lru_cache(maxsize=1)
def load_models_catalog() -> dict[str, Any]:
    path = models_json_path()
    if not path.is_file():
        return _empty_catalog()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_catalog()

    if not isinstance(data, dict):
        return _empty_catalog()

    models_raw = data.get("models")
    if not isinstance(models_raw, list):
        return _empty_catalog()

    models: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in models_raw:
        entry = _normalize_entry(item)
        if not entry:
            continue
        if entry["id"] in seen:
            continue
        seen.add(entry["id"])
        models.append(entry)

    if not models:
        return _empty_catalog()

    default_id = str(data.get("default") or "").strip()
    if default_id not in seen:
        default_id = models[0]["id"]

    return {"default": default_id, "models": models}


# ================================================================================
# カタログキャッシュを破棄して再読み込みする
# ================================================================================
def reload_models_catalog() -> dict[str, Any]:
    load_models_catalog.cache_clear()
    return load_models_catalog()


# ================================================================================
# カタログ内の全モデル一覧を返す
# ================================================================================
def list_models() -> list[dict[str, str]]:
    return list(load_models_catalog()["models"])


# ================================================================================
# models.json に定義された Ollama モデル名の一覧を返す
# ================================================================================
def catalog_ollama_names() -> list[str]:
    return [m["ollama_name"] for m in list_models()]


# ================================================================================
# 有効なカタログモデルが1件以上あるか返す
# ================================================================================
def has_catalog_models() -> bool:
    return bool(list_models())


# ================================================================================
# ID でモデルエントリを取得する（見つからなければ None）
# ================================================================================
def get_model(model_id: str | None) -> dict[str, str] | None:
    mid = (model_id or "").strip()
    if not mid:
        return None
    for m in list_models():
        if m["id"] == mid:
            return m
    return None


# ================================================================================
# デフォルトモデルの ID を返す
# ================================================================================
def default_model_id() -> str:
    return str(load_models_catalog()["default"] or "")


# ================================================================================
# デフォルトモデルのエントリを返す（無ければ None）
# ================================================================================
def default_model() -> dict[str, str] | None:
    mid = default_model_id()
    found = get_model(mid)
    if found:
        return found
    models = list_models()
    return models[0] if models else None


# ================================================================================
# モデル ID を解決する（無ければ既定。カタログ空なら例外）
# ================================================================================
def require_model(model_id: str | None) -> dict[str, str]:
    mid = (model_id or "").strip() or default_model_id()
    found = get_model(mid)
    if found:
        return found
    dm = default_model()
    if dm:
        return dm
    raise ValueError(
        "No models in models.json. Add at least one model entry under \"models\"."
    )


# ================================================================================
# Gradio Dropdown 用の (label, id) 選択肢を返す
# ================================================================================
def choices_for_ui() -> list[tuple[str, str]]:
    return [(m["label"], m["id"]) for m in list_models()]


# ================================================================================
# カタログ内の全モデル ID 一覧を返す
# ================================================================================
def catalog_ids() -> list[str]:
    return [m["id"] for m in list_models()]
