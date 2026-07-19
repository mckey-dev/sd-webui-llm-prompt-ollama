# ================================================================================
# Hugging Face から GGUF をダウンロードしローカルパスを管理する
# ================================================================================
from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

from .model_setup import get_setting
from .models_catalog import default_model, require_model

SETTING_MODELS_DIR = "llm_prompt_ollama_models_dir"


# ================================================================================
# 既定の GGUF 保存ディレクトリを返す
# ================================================================================
def default_models_dir() -> Path:
    try:
        from modules.paths import models_path

        return Path(models_path) / "llm"
    except Exception:
        return Path("models") / "llm"


# ================================================================================
# Settings 上書きを含む実効の GGUF 保存ディレクトリを返す
# ================================================================================
def get_models_dir() -> Path:
    custom = get_setting(SETTING_MODELS_DIR, "")
    if custom:
        return Path(custom).expanduser().resolve()
    return default_models_dir().resolve()


# ================================================================================
# Hugging Face resolve URL を組み立てる
# ================================================================================
def _hf_resolve_url(hf_repo: str, hf_file: str) -> str:
    return f"https://huggingface.co/{hf_repo}/resolve/main/{hf_file}"


# ================================================================================
# カタログモデル ID に対応するローカル GGUF パスを返す
# ================================================================================
def gguf_path_for(model_id: str | None = None) -> Path:
    entry = require_model(model_id)
    return get_models_dir() / entry["hf_file"]


# ================================================================================
# カタログ既定モデルの GGUF パスを返す
# ================================================================================
def default_gguf_path() -> Path:
    return gguf_path_for(None)


# ================================================================================
# GGUF のダウンロード有無とパス情報を文言で返す
# ================================================================================
def describe_gguf_status(model_id: str | None = None) -> str:
    entry = require_model(model_id)
    path = gguf_path_for(entry["id"])
    if path.is_file():
        size_gb = path.stat().st_size / (1024 ** 3)
        return f"Found: {path} ({size_gb:.2f} GB)"
    return (
        f"Not downloaded yet. Will save to: {path}\n"
        f"Source: https://huggingface.co/{entry['hf_repo']}"
    )


# ================================================================================
# huggingface_hub 経由で GGUF をダウンロードする
# ================================================================================
def _download_via_hf_hub(
    dest_dir: Path,
    *,
    hf_repo: str,
    hf_file: str,
    force: bool = False,
) -> Path:
    from huggingface_hub import hf_hub_download

    # Avoid hf_transfer crashes on some hosts (Paperspace etc.)
    os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / hf_file
    if target.is_file() and not force and target.stat().st_size > 1_000_000:
        return target

    kwargs: dict = {
        "repo_id": hf_repo,
        "filename": hf_file,
        "local_dir": str(dest_dir),
    }
    # Newer huggingface_hub dropped some of these; probe once.
    try:
        path = hf_hub_download(
            **kwargs,
            local_dir_use_symlinks=False,
            resume_download=True,
            force_download=force,
        )
    except TypeError:
        path = hf_hub_download(**kwargs)
    return Path(path).resolve()


# ================================================================================
# urllib 経由で GGUF をダウンロードする（レジューム対応）
# ================================================================================
def _download_via_urllib(
    dest_dir: Path,
    *,
    hf_repo: str,
    hf_file: str,
    force: bool = False,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / hf_file
    if target.is_file() and not force and target.stat().st_size > 1_000_000:
        return target

    url = _hf_resolve_url(hf_repo, hf_file)
    partial = dest_dir / (hf_file + ".partial")
    headers = {"User-Agent": "sd-webui-llm-prompt-ollama"}
    start = 0
    if partial.is_file() and not force:
        start = partial.stat().st_size
        if start > 0:
            headers["Range"] = f"bytes={start}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            mode = "ab" if start and resp.status == 206 else "wb"
            if mode == "wb" and partial.is_file():
                partial.unlink()
            with open(partial, mode) as out:
                while True:
                    chunk = resp.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} downloading {url}: {e.reason}") from e

    if target.is_file():
        target.unlink()
    partial.replace(target)
    return target.resolve()


# ================================================================================
# カタログモデルの GGUF を models/llm 等へダウンロードする
# ================================================================================
def download_gguf(model_id: str | None = None, *, force: bool = False) -> tuple[Path, str]:
    entry = require_model(model_id)
    hf_repo = entry["hf_repo"]
    hf_file = entry["hf_file"]
    dest_dir = get_models_dir()
    existing = dest_dir / hf_file
    if existing.is_file() and not force and existing.stat().st_size > 1_000_000:
        size_gb = existing.stat().st_size / (1024 ** 3)
        return existing.resolve(), (
            f"Already present ({size_gb:.2f} GB):\n{existing.resolve()}\n"
            f"Source: https://huggingface.co/{hf_repo}"
        )

    errors: list[str] = []

    try:
        path = _download_via_hf_hub(
            dest_dir, hf_repo=hf_repo, hf_file=hf_file, force=force
        )
        size_gb = path.stat().st_size / (1024 ** 3)
        return path, (
            f"Downloaded via huggingface_hub ({size_gb:.2f} GB):\n{path}\n"
            f"Repo: {hf_repo}\nFile: {hf_file}"
        )
    except Exception as e:
        errors.append(f"huggingface_hub: {e}")

    try:
        path = _download_via_urllib(
            dest_dir, hf_repo=hf_repo, hf_file=hf_file, force=force
        )
        size_gb = path.stat().st_size / (1024 ** 3)
        return path, (
            f"Downloaded via HTTP ({size_gb:.2f} GB):\n{path}\n"
            f"URL: {_hf_resolve_url(hf_repo, hf_file)}"
        )
    except Exception as e:
        errors.append(f"urllib: {e}")

    raise RuntimeError(
        "GGUF download failed:\n- "
        + "\n- ".join(errors)
        + f"\nManual: https://huggingface.co/{hf_repo}"
    )


# ================================================================================
# カタログ既定モデルの GGUF をダウンロードする（互換エイリアス）
# ================================================================================
def download_default_gguf(*, force: bool = False) -> tuple[Path, str]:
    return download_gguf(None, force=force)


# ================================================================================
# カタログ既定の Hugging Face リポジトリ／ファイル名を返す
# ================================================================================
def _default_hf() -> tuple[str, str]:
    m = default_model()
    if not m:
        return "", ""
    return m["hf_repo"], m["hf_file"]


HF_REPO_ID = _default_hf()[0]
HF_GGUF_FILENAME = _default_hf()[1]
HF_RESOLVE_URL = _hf_resolve_url(HF_REPO_ID, HF_GGUF_FILENAME) if HF_REPO_ID and HF_GGUF_FILENAME else ""
