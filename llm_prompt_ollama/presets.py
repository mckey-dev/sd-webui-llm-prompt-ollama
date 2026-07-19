# ================================================================================
# 拡張ルートの presets.json からインストラクションプリセットを読み込む
# ================================================================================
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# Minimal safe fallback when presets.json is missing or invalid (no NSFW).
_FALLBACK_PRESET_ID = "Idea → SD prompt"
_FALLBACK_CUSTOM_ID = "Custom"
_FALLBACK_LANGS = ["English", "日本語"]
_FALLBACK_EN = (
    "You are writing ONE natural-language prompt for a Stable Diffusion / anime image model.\n"
    "The user provides an idea (possibly in Japanese or mixed language).\n"
    "Expand it into fluent English prose covering subject, appearance, clothing, pose, "
    "expression, setting, lighting, camera, and art style when relevant.\n"
    "Do not output comma-separated Danbooru tags unless the user explicitly asks for tags.\n"
    "Do not add explanations. Output only the prompt."
)
_FALLBACK_JA = (
    "あなたは Stable Diffusion／アニメ向け画像生成用の自然言語プロンプトを1つ作成します。\n"
    "ユーザーはアイデア（日本語や混在文でも可）を与えます。\n"
    "被写体・外見・服装・ポーズ・表情・背景・照明・カメラ・画風を、必要に応じて含む"
    "流暢な英語の文章に展開してください。\n"
    "ユーザーがタグを明示的に求めない限り、カンマ区切りの Danbooru タグ列にはしないでください。\n"
    "説明は不要です。プロンプト本文のみを出力してください（英語）。"
)


# ================================================================================
# 拡張機能のルートディレクトリを返す
# ================================================================================
def _extension_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ================================================================================
# presets.json のパスを返す
# ================================================================================
def presets_json_path() -> Path:
    return _extension_root() / "presets.json"


# ================================================================================
# presets.json 欠落・不正時の最小カタログを返す（NSFW なし）
# ================================================================================
def _fallback_catalog() -> dict[str, Any]:
    return {
        "default_lang": "English",
        "default_preset": _FALLBACK_PRESET_ID,
        "languages": list(_FALLBACK_LANGS),
        "presets": [
            {
                "id": _FALLBACK_PRESET_ID,
                "nsfw": False,
                "instructions": {
                    "English": _FALLBACK_EN,
                    "日本語": _FALLBACK_JA,
                },
            },
            {
                "id": _FALLBACK_CUSTOM_ID,
                "nsfw": False,
                "instructions": {"English": "", "日本語": ""},
            },
        ],
    }


# ================================================================================
# 生のプリセット辞書を正規化する（不正時は None）
# ================================================================================
def _normalize_preset(raw: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    pid = str(raw.get("id") or "").strip()
    if not pid:
        return None
    instructions_raw = raw.get("instructions")
    if not isinstance(instructions_raw, dict):
        return None
    instructions: dict[str, str] = {}
    for lang, text in instructions_raw.items():
        key = str(lang).strip()
        if not key:
            continue
        instructions[key] = "" if text is None else str(text)
    if not instructions:
        return None
    return {
        "id": pid,
        "nsfw": bool(raw.get("nsfw", False)),
        "instructions": instructions,
    }


# ================================================================================
# presets.json を読み込みキャッシュする
# ================================================================================
@lru_cache(maxsize=1)
def load_presets_catalog() -> dict[str, Any]:
    path = presets_json_path()
    if not path.is_file():
        return _fallback_catalog()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _fallback_catalog()

    if not isinstance(data, dict):
        return _fallback_catalog()

    presets_raw = data.get("presets")
    if not isinstance(presets_raw, list):
        return _fallback_catalog()

    presets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in presets_raw:
        entry = _normalize_preset(item)
        if not entry:
            continue
        if entry["id"] in seen:
            continue
        seen.add(entry["id"])
        presets.append(entry)

    if not presets:
        return _fallback_catalog()

    languages_raw = data.get("languages")
    if isinstance(languages_raw, list) and languages_raw:
        languages = [str(x).strip() for x in languages_raw if str(x).strip()]
    else:
        languages = list(_FALLBACK_LANGS)
    if not languages:
        languages = list(_FALLBACK_LANGS)

    default_preset = str(data.get("default_preset") or "").strip()
    if default_preset not in seen:
        default_preset = presets[0]["id"]

    default_lang = str(data.get("default_lang") or "").strip()
    if default_lang not in languages:
        default_lang = languages[0]

    return {
        "default_lang": default_lang,
        "default_preset": default_preset,
        "languages": languages,
        "presets": presets,
    }


# ================================================================================
# プリセットキャッシュを破棄して再読み込みする
# ================================================================================
def reload_presets_catalog() -> dict[str, Any]:
    load_presets_catalog.cache_clear()
    return load_presets_catalog()


# ================================================================================
# カタログ内の全プリセット一覧を返す
# ================================================================================
def list_presets() -> list[dict[str, Any]]:
    return list(load_presets_catalog()["presets"])


# ================================================================================
# ID でプリセットを取得する（見つからなければ None）
# ================================================================================
def get_preset(preset_id: str | None) -> dict[str, Any] | None:
    pid = (preset_id or "").strip()
    if not pid:
        return None
    for p in list_presets():
        if p["id"] == pid:
            return p
    return None


# ================================================================================
# 公開定数（起動時に presets.json から解決）
# ================================================================================
def _catalog() -> dict[str, Any]:
    return load_presets_catalog()


DEFAULT_LANG = str(_catalog()["default_lang"])
DEFAULT_PRESET = str(_catalog()["default_preset"])
LANG_CHOICES = list(_catalog()["languages"])
PRESET_CHOICES = [p["id"] for p in list_presets()]


# ================================================================================
# NSFW 向けプリセットかどうかを判定する
# ================================================================================
def is_nsfw_preset(preset: str) -> bool:
    entry = get_preset(preset)
    return bool(entry and entry.get("nsfw"))


# ================================================================================
# プリセット名と言語からインストラクション文を返す
# ================================================================================
def instruction_for_preset(preset: str, lang: str = DEFAULT_LANG) -> str:
    entry = get_preset(preset) or get_preset(DEFAULT_PRESET)
    if not entry:
        return ""
    instructions = entry.get("instructions") or {}
    if lang in instructions:
        return str(instructions[lang])
    if DEFAULT_LANG in instructions:
        return str(instructions[DEFAULT_LANG])
    if instructions:
        return str(next(iter(instructions.values())))
    return ""
