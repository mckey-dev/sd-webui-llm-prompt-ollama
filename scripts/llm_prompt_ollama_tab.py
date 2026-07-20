# ================================================================================
# LLM Prompt (Ollama) の Gradio UI タブを登録する
# ================================================================================
from __future__ import annotations

import base64
import io
import traceback

import gradio as gr

from modules import script_callbacks, shared
from modules import infotext_utils as parameters_copypaste

from llm_prompt_ollama.download import (
    SETTING_MODELS_DIR,
    default_models_dir,
    describe_gguf_status,
    download_gguf,
    get_models_dir,
    gguf_path_for,
    mmproj_path_for,
)
from llm_prompt_ollama.model_setup import (
    DEFAULT_GGUF_FILENAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_OLLAMA_URL,
    SETTING_API_URL,
    SETTING_DEFAULT_GGUF,
    SETTING_DEFAULT_MODEL,
    SETTING_OLLAMA_BIN,
    SETTING_SHOW_UNCENSORED_PRESETS,
    connection_help,
    create_model,
    get_setting,
    start_ollama_serve,
    restart_ollama_serve,
)
from llm_prompt_ollama.models_catalog import (
    choices_for_ui,
    default_model_id,
    get_model,
    has_catalog_models,
    reload_models_catalog,
    require_model,
    catalog_ollama_names,
)
from llm_prompt_ollama.ollama_client import (
    OllamaClient,
    OllamaError,
    format_full_connection_status,
    model_names_equivalent,
    pick_ollama_model_name,
)
from llm_prompt_ollama.presets import (
    DEFAULT_LANG,
    LANG_CHOICES,
    default_preset_for,
    get_custom_instruction,
    instruction_for_preset,
    preset_choices_for,
    reload_presets_catalog,
    save_custom_instruction,
)
from llm_prompt_ollama.wd_tagger import (
    DEFAULT_WD_REPO,
    tag_image,
    wd_model_choices,
)

_WD_CHOICES = wd_model_choices()


# ================================================================================
# Settings の Ollama API URL を返す
# ================================================================================
def _api_url() -> str:
    return get_setting(SETTING_API_URL, DEFAULT_OLLAMA_URL)


# ================================================================================
# Settings の既定 Ollama モデル名を返す
# ================================================================================
def _default_model() -> str:
    return get_setting(SETTING_DEFAULT_MODEL, DEFAULT_MODEL_NAME)


# ================================================================================
# Ollama モデル Dropdown の choices / value を更新内容に合わせる
# ================================================================================
def _model_dropdown_update(
    preferred: str,
    models: list[str],
    *,
    current: str | None = None,
) -> gr.update:
    pref = (preferred or "").strip() or _default_model()
    choices = models or ([pref] if pref else [])
    if current:
        picked = pick_ollama_model_name(current, choices)
        if picked:
            return gr.update(choices=choices, value=picked)
        if current in choices:
            return gr.update(choices=choices, value=current)
    picked = pick_ollama_model_name(pref, choices)
    if picked:
        return gr.update(choices=choices, value=picked)
    if pref in choices:
        return gr.update(choices=choices, value=pref)
    return gr.update(choices=choices, value=choices[0] if choices else pref)


# ================================================================================
# カタログ既定モデルの ID を返す
# ================================================================================
def _default_catalog_id() -> str:
    return default_model_id()


# ================================================================================
# カタログモデルに対応する GGUF パス（Settings 上書き含む）を返す
# ================================================================================
def _gguf_for_catalog(model_id: str | None) -> str:
    entry = get_model(model_id) if model_id else None
    if not entry:
        return ""
    custom = get_setting(SETTING_DEFAULT_GGUF, "")
    if custom and entry["id"] == default_model_id():
        return custom
    return str(gguf_path_for(entry["id"]))


# ================================================================================
# カタログモデルに対応する mmproj パス文字列を返す
# ================================================================================
def _mmproj_for_catalog(model_id: str | None) -> str:
    entry = get_model(model_id) if model_id else None
    if not entry:
        return ""
    path = mmproj_path_for(entry["id"])
    return str(path) if path else ""


# ================================================================================
# OllamaClient インスタンスを生成する
# ================================================================================
def _client(api_url: str | None = None) -> OllamaClient:
    return OllamaClient(api_url or _api_url())


# ================================================================================
# Download / Create ボタンの interactive 更新を返す
# ================================================================================
def _catalog_button_updates(enabled: bool):
    return gr.update(interactive=enabled), gr.update(interactive=enabled)


# ================================================================================
# PIL / numpy 画像を base64 文字列へ変換する
# ================================================================================
def _image_to_b64(image) -> str:
    from PIL import Image
    import numpy as np

    if image is None:
        raise ValueError("No image provided.")
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ================================================================================
# カタログ選択変更時に GGUF パス・状態・モデル名を更新する
# ================================================================================
def _on_catalog_change(catalog_id: str):
    entry = get_model(catalog_id)
    if not entry:
        return "", "", "", gr.update(), *_catalog_button_updates(False)
    return (
        _gguf_for_catalog(entry["id"]),
        _mmproj_for_catalog(entry["id"]),
        describe_gguf_status(entry["id"]),
        gr.update(value=entry["ollama_name"]),
        *_catalog_button_updates(True),
    )


# ================================================================================
# 選択中カタログモデルの GGUF をダウンロードする
# ================================================================================
def _download_gguf(catalog_id: str, force: bool):
    if not has_catalog_models():
        return "", "", "", "No models in models.json.", gr.update()
    try:
        path, msg = download_gguf(catalog_id, force=bool(force))
        entry = require_model(catalog_id)
        status = describe_gguf_status(entry["id"])
        return (
            str(path),
            _mmproj_for_catalog(entry["id"]),
            status,
            msg,
            gr.update(value=entry["ollama_name"]),
        )
    except Exception as e:
        traceback.print_exc()
        entry = get_model(catalog_id)
        if not entry:
            return "", "", "", f"Download failed: {type(e).__name__}: {e}", gr.update()
        return (
            _gguf_for_catalog(entry["id"]),
            _mmproj_for_catalog(entry["id"]),
            describe_gguf_status(entry["id"]),
            f"Download failed: {type(e).__name__}: {e}",
            gr.update(),
        )


# ================================================================================
# GGUF 状態と Local GGUF path を更新する（path は空のときだけ自動入力）
# ================================================================================
def _refresh_gguf_status(catalog_id: str, current_path: str):
    entry = get_model(catalog_id)
    if not entry:
        return "", str(get_models_dir()), "", ""
    status = describe_gguf_status(entry["id"])
    path = (current_path or "").strip() or _gguf_for_catalog(entry["id"])
    return status, str(get_models_dir()), path, _mmproj_for_catalog(entry["id"])


# ================================================================================
# models.json を再読み込みしカタログ Dropdown を更新する
# ================================================================================
def _refresh_catalog(current_id: str):
    reload_models_catalog()
    choices = choices_for_ui()
    if not choices:
        return (
            gr.update(choices=[], value=None),
            "",
            "",
            "",
            gr.update(),
            str(get_models_dir()),
            *_catalog_button_updates(False),
        )
    ids = [c[1] for c in choices]
    default_id = _default_catalog_id()
    if current_id in ids:
        value = current_id
    elif default_id in ids:
        value = default_id
    else:
        value = ids[0]
    entry = require_model(value)
    return (
        gr.update(choices=choices, value=value),
        _gguf_for_catalog(entry["id"]),
        _mmproj_for_catalog(entry["id"]),
        describe_gguf_status(entry["id"]),
        gr.update(value=entry["ollama_name"]),
        str(get_models_dir()),
        *_catalog_button_updates(True),
    )


# ================================================================================
# プリセット変更時にインストラクション文を差し替える
# ================================================================================
def _on_preset_change(preset: str, lang: str, _current: str, target: str = "idea"):
    if preset == "Custom":
        saved = get_custom_instruction(target, lang)
        if saved:
            return saved
        return gr.update()
    return instruction_for_preset(preset, lang)


# ================================================================================
# 言語変更時にインストラクション文を差し替える
# ================================================================================
def _on_lang_change(lang: str, preset: str, _current: str, target: str = "idea"):
    if preset == "Custom":
        saved = get_custom_instruction(target, lang)
        return saved if saved else ""
    return instruction_for_preset(preset, lang)


# ================================================================================
# presets.json を再読み込みし用途向け Dropdown を更新する
# ================================================================================
def _refresh_presets_for(target: str, current_preset: str, current_lang: str):
    reload_presets_catalog()
    choices = preset_choices_for(target)
    langs = list(LANG_CHOICES)
    try:
        from llm_prompt_ollama.presets import load_presets_catalog

        langs = list(load_presets_catalog()["languages"])
    except Exception:
        pass
    default_p = default_preset_for(target)
    preset = current_preset if current_preset in choices else default_p
    lang = current_lang if current_lang in langs else (langs[0] if langs else DEFAULT_LANG)
    if preset == "Custom":
        saved = get_custom_instruction(target, lang)
        instruction = saved if saved else gr.update()
    else:
        instruction = instruction_for_preset(preset, lang)
    return (
        gr.update(choices=langs, value=lang),
        gr.update(choices=choices, value=preset),
        instruction,
    )


# ================================================================================
# カタログ選択から優先 Ollama モデル名を返す
# ================================================================================
def _preferred_model_name(catalog_id: str | None) -> str:
    entry = get_model(catalog_id) if catalog_id else None
    if entry:
        return entry["ollama_name"]
    return _default_model()


# ================================================================================
# 接続 OK 時の status 文字列とモデル Dropdown 更新を返す
# ================================================================================
def _ollama_status_and_dropdown(
    api_url: str,
    catalog_id: str | None,
    current_model: str | None,
    *,
    prefix: str | None = None,
) -> tuple[str, gr.update]:
    client = _client(api_url)
    installed = client.list_models()
    running = client.list_running_models()
    status = format_full_connection_status(
        client.base_url,
        installed,
        running,
        catalog_ollama_names=catalog_ollama_names(),
        prefix=prefix,
    )
    preferred = _preferred_model_name(catalog_id)
    return status, _model_dropdown_update(preferred, installed, current=current_model or None)


# ================================================================================
# Ollama 接続確認とモデル一覧の更新を行う
# ================================================================================
def _check_connection(api_url: str, catalog_id: str, model_name: str):
    try:
        return _ollama_status_and_dropdown(api_url, catalog_id, model_name)
    except OllamaError as e:
        help_text = connection_help(api_url or _api_url())
        return f"{e}\n\n{help_text}", gr.update()
    except Exception as e:
        traceback.print_exc()
        return f"Connection failed: {e}", gr.update()


# ================================================================================
# Ollama サーバーを起動しモデル一覧を更新する
# ================================================================================
def _start_ollama(api_url: str, catalog_id: str, model_name: str):
    try:
        msg = start_ollama_serve(
            api_url=api_url or _api_url(),
            ollama_bin=get_setting(SETTING_OLLAMA_BIN, "") or None,
        )
        return msg, _ollama_status_and_dropdown(api_url, catalog_id, model_name)[1]
    except OllamaError as e:
        return str(e), gr.update()
    except Exception as e:
        traceback.print_exc()
        return f"Start failed: {e}", gr.update()


# ================================================================================
# Ollama サーバーを再起動しモデル一覧を更新する
# ================================================================================
def _restart_ollama(api_url: str, catalog_id: str, model_name: str):
    try:
        msg = restart_ollama_serve(
            api_url=api_url or _api_url(),
            ollama_bin=get_setting(SETTING_OLLAMA_BIN, "") or None,
        )
        return msg, _ollama_status_and_dropdown(api_url, catalog_id, model_name)[1]
    except OllamaError as e:
        return str(e), gr.update()
    except Exception as e:
        traceback.print_exc()
        return f"Restart failed: {e}", gr.update()


# ================================================================================
# メモリ上の全モデルをアンロードする
# ================================================================================
def _unload_all(api_url: str, catalog_id: str, model_name: str):
    try:
        client = _client(api_url)
        running = client.list_running_models()
        names: list[str] = []
        seen: set[str] = set()
        for item in running:
            n = str(item.get("name") or item.get("model") or "").strip()
            if not n or n in seen:
                continue
            seen.add(n)
            names.append(n)
        for n in names:
            client.unload_model(n)
        prefix = f"Unloaded {len(names)} model(s) from memory."
        return _ollama_status_and_dropdown(api_url, catalog_id, model_name, prefix=prefix)
    except OllamaError as e:
        help_text = connection_help(api_url or _api_url())
        return f"{e}\n\n{help_text}", gr.update()
    except Exception as e:
        traceback.print_exc()
        return f"Unload failed: {e}", gr.update()


# ================================================================================
# 選択中の1モデルをメモリからアンロードする
# ================================================================================
def _unload_selected(api_url: str, catalog_id: str, model_name: str):
    try:
        selected = (model_name or "").strip()
        if not selected:
            raise ValueError("Select an Ollama model name first.")
        client = _client(api_url)
        running = client.list_running_models()
        was_loaded = any(
            model_names_equivalent(selected, str(item.get("name") or item.get("model") or ""))
            for item in running
        )
        client.unload_model(selected)
        if was_loaded:
            prefix = f"Unloaded: {selected}"
        else:
            prefix = f"Unloaded: {selected} (was not loaded)"
        return _ollama_status_and_dropdown(api_url, catalog_id, model_name, prefix=prefix)
    except OllamaError as e:
        help_text = connection_help(api_url or _api_url())
        return f"{e}\n\n{help_text}", gr.update()
    except Exception as e:
        traceback.print_exc()
        return f"Unload failed: {e}", gr.update()


# ================================================================================
# ローカル GGUF から Ollama モデルを作成／更新する
# ================================================================================
def _create_model(api_url: str, model_name: str, gguf_path: str, mmproj_path: str):
    try:
        msg = create_model(
            model_name=model_name,
            gguf_path=gguf_path,
            mmproj_path=(mmproj_path or "").strip() or None,
            api_url=api_url or _api_url(),
            ollama_bin=get_setting(SETTING_OLLAMA_BIN, "") or None,
            prefer_api=True,
        )
        models = []
        try:
            models = _client(api_url).list_models()
        except Exception:
            pass
        return msg, _model_dropdown_update(model_name, models, current=model_name)
    except OllamaError as e:
        return str(e), gr.update()
    except Exception as e:
        traceback.print_exc()
        return f"Create failed: {type(e).__name__}: {e}", gr.update()


# ================================================================================
# アイデアから SD 用プロンプトを生成する（Idea タブ）
# ================================================================================
def _generate_idea(
    api_url: str,
    model_name: str,
    idea: str,
    preset: str,
    instruction: str,
    lang: str,
    temperature: float,
    top_p: float,
    num_predict: int,
):
    try:
        saved_custom = False
        if preset == "Custom":
            saved_custom = save_custom_instruction("idea", instruction, lang)

        if not (idea or "").strip():
            raise ValueError("Idea text is empty.")
        if not (model_name or "").strip():
            raise ValueError("Model name is empty. Check connection or create the model first.")

        system = instruction if preset == "Custom" else (instruction or instruction_for_preset(preset))
        client = _client(api_url)
        prompt = client.chat(
            model=model_name.strip(),
            user_content=(idea or "").strip(),
            system=system,
            temperature=float(temperature),
            top_p=float(top_p),
            num_predict=int(num_predict),
            think=False,
        )
        if not (prompt or "").strip():
            log = "Done — but model returned empty text."
            if saved_custom:
                log += " · Saved Custom instruction."
            return "", log
        log = f"Done ({len(prompt)} chars)"
        if saved_custom:
            log += " · Saved Custom instruction."
        return prompt, log
    except Exception as e:
        traceback.print_exc()
        return "", f"Generate failed: {type(e).__name__}: {e}"


# ================================================================================
# 画像を VLM で解析し SD 用プロンプトを生成する
# ================================================================================
def _generate_vlm(
    api_url: str,
    model_name: str,
    image,
    extra: str,
    preset: str,
    instruction: str,
    lang: str,
    temperature: float,
    top_p: float,
    num_predict: int,
):
    try:
        saved_custom = False
        if preset == "Custom":
            saved_custom = save_custom_instruction("vlm", instruction, lang)

        if image is None:
            raise ValueError("Image is required for VLM generate.")
        if not (model_name or "").strip():
            raise ValueError("Model name is empty. Create / select a vision model first.")

        system = instruction if preset == "Custom" else (instruction or instruction_for_preset(preset))
        user_content = (extra or "").strip() or (
            "Analyze this image and write one natural-language image-generation prompt."
        )
        b64 = _image_to_b64(image)
        client = _client(api_url)
        prompt = client.chat(
            model=model_name.strip(),
            user_content=user_content,
            system=system,
            images=[b64],
            temperature=float(temperature),
            top_p=float(top_p),
            num_predict=int(num_predict),
            think=False,
            timeout=600.0,
        )
        if not (prompt or "").strip():
            log = "VLM done — but model returned empty text."
            if saved_custom:
                log += " · Saved Custom instruction."
            return "", log
        log = f"VLM done ({len(prompt)} chars)"
        if saved_custom:
            log += " · Saved Custom instruction."
        return prompt, log
    except Exception as e:
        traceback.print_exc()
        return "", f"VLM generate failed: {type(e).__name__}: {e}"


# ================================================================================
# WD Tagger でタグ形式プロンプトを生成する
# ================================================================================
def _generate_wd(
    image,
    repo_id: str,
    general_thresh: float,
    character_thresh: float,
    underscore_to_space: bool,
    escape_parentheses: bool,
):
    try:
        if image is None:
            raise ValueError("Image is required for WD Tagger.")
        tags = tag_image(
            image,
            repo_id=repo_id or DEFAULT_WD_REPO,
            general_thresh=float(general_thresh),
            character_thresh=float(character_thresh),
            underscore_to_space=bool(underscore_to_space),
            escape_parentheses=bool(escape_parentheses),
        )
        if not tags.strip():
            return "", "WD Tagger done — no tags above threshold."
        return tags, f"WD Tagger done ({len(tags.split(','))} tags)"
    except Exception as e:
        traceback.print_exc()
        return "", f"WD Tagger failed: {type(e).__name__}: {e}"


# ================================================================================
# WebUI Settings に本拡張のオプションを登録する
# ================================================================================
def on_ui_settings():
    section = ("llm_prompt_ollama", "LLM Prompt (Ollama)")
    shared.opts.add_option(
        SETTING_API_URL,
        shared.OptionInfo(
            DEFAULT_OLLAMA_URL,
            "Ollama API URL",
            section=section,
        ).info("Default: http://127.0.0.1:11434"),
    )
    shared.opts.add_option(
        SETTING_DEFAULT_MODEL,
        shared.OptionInfo(
            DEFAULT_MODEL_NAME,
            "Default Ollama model name",
            section=section,
        ).info(
            f"Used after Create / Update. Default from models.json: {DEFAULT_MODEL_NAME}"
        ),
    )
    shared.opts.add_option(
        SETTING_MODELS_DIR,
        shared.OptionInfo(
            str(default_models_dir()),
            "Directory for GGUF downloads (models/llm)",
            section=section,
            component_args=shared.hide_dirs,
        ).info(
            f"GGUF files from models.json are saved here "
            f"(default file: {DEFAULT_GGUF_FILENAME}). Edit models.json to add models."
        ),
    )
    shared.opts.add_option(
        SETTING_DEFAULT_GGUF,
        shared.OptionInfo(
            "",
            "Default path to GGUF file (optional override)",
            section=section,
            component_args=shared.hide_dirs,
        ).info("Leave empty to use models/llm/<hf_file from models.json>."),
    )
    shared.opts.add_option(
        SETTING_OLLAMA_BIN,
        shared.OptionInfo(
            "",
            "Path to ollama binary (optional)",
            section=section,
            component_args=shared.hide_dirs,
        ).info("Used when /api/create fails and CLI fallback is needed. Leave empty to use PATH."),
    )
    shared.opts.add_option(
        SETTING_SHOW_UNCENSORED_PRESETS,
        shared.OptionInfo(
            False,
            "Show uncensored instruction presets",
            section=section,
        ).info(
            "When enabled, presets with uncensored: true in presets.json appear in Idea/VLM dropdowns."
        ),
    )


# ================================================================================
# LLM Prompt (Ollama) タブの UI を構築する
# ================================================================================
def on_ui_tabs():
    catalog_choices = choices_for_ui()
    has_catalog = bool(catalog_choices)
    if has_catalog:
        choice_values = [c[1] for c in catalog_choices]
        initial_id = _default_catalog_id()
        if initial_id not in choice_values:
            initial_id = choice_values[0]
        initial_entry = require_model(initial_id)
        initial_ollama = initial_entry["ollama_name"]
    else:
        initial_id = None
        initial_ollama = ""

    idea_preset_choices = preset_choices_for("idea")
    idea_default_preset = default_preset_for("idea")
    vlm_preset_choices = preset_choices_for("vlm")
    vlm_default_preset = default_preset_for("vlm")

    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown(
            "### LLM Prompt (Ollama)\n"
            "1. **モデルロード** — Ollama 接続・GGUF Download / Create（vision は mmproj 対応）  \n"
            "2. **Idea** — テキストから SD プロンプト  \n"
            "3. **VLM** — 画像解析から SD プロンプト（おまけ: WD Tagger）"
        )

        with gr.Tabs():
            # ----- モデルロード -----
            with gr.Tab("モデルロード"):
                with gr.Accordion("Ollama connection", open=True):
                    api_url = gr.Textbox(
                        label="Ollama API URL",
                        value=_api_url(),
                        lines=1,
                    )
                    with gr.Row():
                        check_btn = gr.Button("Check connection", variant="primary")
                        start_btn = gr.Button("Start Ollama")
                        restart_btn = gr.Button("Restart Ollama")
                        unload_all_btn = gr.Button("Unload all from memory")
                    conn_status = gr.Textbox(label="Connection status", lines=10, interactive=False)

                with gr.Accordion("Model from local GGUF", open=True):
                    with gr.Row():
                        catalog_dd = gr.Dropdown(
                            label="Catalog model (models.json)",
                            choices=catalog_choices,
                            value=initial_id,
                            allow_custom_value=False,
                            scale=4,
                        )
                        refresh_catalog_btn = gr.Button("更新", scale=1)
                    models_dir_box = gr.Textbox(
                        label="GGUF download directory",
                        value=str(get_models_dir()),
                        interactive=False,
                    )
                    models_dir_box.do_not_save_to_config = True
                    gguf_status = gr.Textbox(
                        label="GGUF status",
                        value="",
                        interactive=False,
                        lines=3,
                    )
                    gguf_status.do_not_save_to_config = True
                    force_redownload = gr.Checkbox(label="Force re-download", value=False)
                    with gr.Row():
                        download_btn = gr.Button(
                            "Download GGUF",
                            variant="primary",
                            interactive=has_catalog,
                        )
                        refresh_gguf_btn = gr.Button("↻ Status")
                    with gr.Row():
                        model_name = gr.Dropdown(
                            label="Ollama model name",
                            choices=[initial_ollama] if initial_ollama else [],
                            value=initial_ollama or None,
                            allow_custom_value=True,
                            scale=4,
                        )
                        unload_selected_btn = gr.Button("Unload selected model", scale=1)
                    gguf_path = gr.Textbox(
                        label="Local GGUF path",
                        value="",
                        lines=1,
                        placeholder="Catalog 選択・更新・Download で自動入力（手動編集可）",
                    )
                    gguf_path.do_not_save_to_config = True
                    mmproj_path = gr.Textbox(
                        label="Local mmproj path (auto)",
                        value="",
                        lines=1,
                        interactive=False,
                        placeholder="hf_mmproj があるカタログのみ表示",
                    )
                    mmproj_path.do_not_save_to_config = True
                    create_btn = gr.Button(
                        "Create / Update model",
                        variant="primary",
                        interactive=has_catalog,
                    )
                    create_log = gr.Textbox(label="Create / Download log", lines=8, interactive=False)

            # ----- Idea -----
            with gr.Tab("Idea"):
                with gr.Row():
                    with gr.Column(scale=1):
                        idea = gr.Textbox(
                            label="Idea / description (Japanese OK)",
                            lines=8,
                            placeholder="例: 雨の夜のネオン街を歩く銀髪の少女、サイバーパンク、映画的な照明",
                        )
                        idea_lang = gr.Radio(
                            choices=LANG_CHOICES,
                            value=DEFAULT_LANG,
                            label="Instruction language",
                        )
                        with gr.Row():
                            idea_preset = gr.Dropdown(
                                label="Instruction preset",
                                choices=idea_preset_choices,
                                value=idea_default_preset,
                                allow_custom_value=False,
                                scale=4,
                            )
                            refresh_idea_presets_btn = gr.Button("更新", scale=1)
                        idea_instruction = gr.Textbox(
                            label="Instruction (editable — use Custom to keep edits)",
                            lines=8,
                            value=instruction_for_preset(idea_default_preset, DEFAULT_LANG),
                        )
                        idea_generate_btn = gr.Button("Generate prompt", variant="primary")
                        with gr.Accordion("詳細設定", open=False):
                            idea_temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                            idea_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                            idea_num_predict = gr.Slider(
                                64, 2048, value=512, step=32, label="Max tokens (num_predict)"
                            )
                    with gr.Column(scale=1):
                        idea_prompt = gr.Textbox(
                            label="Generated prompt",
                            lines=18,
                            show_copy_button=True,
                            elem_id="llm_prompt_ollama_idea_generated_prompt",
                        )
                        with gr.Row(elem_id="llm_prompt_ollama_idea_prompt_actions"):
                            idea_send_txt = gr.Button("Send to txt2img")
                            idea_send_img = gr.Button("Send to img2img")
                        idea_log = gr.Textbox(label="Log", lines=6, interactive=False)

            # ----- VLM -----
            with gr.Tab("VLM"):
                with gr.Row():
                    with gr.Column(scale=1):
                        vlm_image = gr.Image(
                            label="Input image",
                            type="pil",
                            image_mode="RGBA",
                        )
                        vlm_extra = gr.Textbox(
                            label="Additional instruction (optional)",
                            lines=3,
                            placeholder="例: 構図と照明を強調して",
                        )
                        vlm_generate_btn = gr.Button("Generate with VLM", variant="primary")
                        with gr.Accordion("VLM オプション", open=False):
                            vlm_lang = gr.Radio(
                                choices=LANG_CHOICES,
                                value=DEFAULT_LANG,
                                label="Instruction language",
                            )
                            with gr.Row():
                                vlm_preset = gr.Dropdown(
                                    label="Instruction preset",
                                    choices=vlm_preset_choices,
                                    value=vlm_default_preset,
                                    allow_custom_value=False,
                                    scale=4,
                                )
                                refresh_vlm_presets_btn = gr.Button("更新", scale=1)
                            vlm_instruction = gr.Textbox(
                                label="Instruction (editable — use Custom to keep edits)",
                                lines=6,
                                value=instruction_for_preset(vlm_default_preset, DEFAULT_LANG),
                            )
                            vlm_temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                            vlm_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                            vlm_num_predict = gr.Slider(
                                64, 2048, value=512, step=32, label="Max tokens (num_predict)"
                            )
                        with gr.Accordion("WD Tagger", open=False):
                            wd_repo = gr.Dropdown(
                                label="WD Tagger model (V3 only)",
                                choices=_WD_CHOICES,
                                value=DEFAULT_WD_REPO,
                                allow_custom_value=False,
                            )
                            wd_general = gr.Slider(0.0, 1.0, value=0.35, step=0.05, label="General threshold")
                            wd_character = gr.Slider(0.0, 1.0, value=0.85, step=0.05, label="Character threshold")
                            wd_underscore_to_space = gr.Checkbox(
                                label="アンダースコアをスペースに置換する",
                                value=True,
                            )
                            wd_escape_parens = gr.Checkbox(
                                label="括弧にエスケープ処理をする",
                                value=True,
                            )
                            wd_btn = gr.Button("Tag with WD Tagger")
                    with gr.Column(scale=1):
                        vlm_prompt = gr.Textbox(
                            label="Generated prompt",
                            lines=18,
                            show_copy_button=True,
                            elem_id="llm_prompt_ollama_vlm_generated_prompt",
                        )
                        with gr.Row(elem_id="llm_prompt_ollama_vlm_prompt_actions"):
                            vlm_send_txt = gr.Button("Send to txt2img")
                            vlm_send_img = gr.Button("Send to img2img")
                        vlm_log = gr.Textbox(label="Log", lines=6, interactive=False)

        for tabname, button, source in (
            ("txt2img", idea_send_txt, idea_prompt),
            ("img2img", idea_send_img, idea_prompt),
            ("txt2img", vlm_send_txt, vlm_prompt),
            ("img2img", vlm_send_img, vlm_prompt),
        ):
            parameters_copypaste.register_paste_params_button(
                parameters_copypaste.ParamBinding(
                    paste_button=button,
                    tabname=tabname,
                    source_text_component=source,
                )
            )

        idea_preset.change(
            fn=lambda p, l, c: _on_preset_change(p, l, c, "idea"),
            inputs=[idea_preset, idea_lang, idea_instruction],
            outputs=[idea_instruction],
        )
        idea_lang.change(
            fn=lambda l, p, c: _on_lang_change(l, p, c, "idea"),
            inputs=[idea_lang, idea_preset, idea_instruction],
            outputs=[idea_instruction],
        )
        refresh_idea_presets_btn.click(
            fn=lambda p, l: _refresh_presets_for("idea", p, l),
            inputs=[idea_preset, idea_lang],
            outputs=[idea_lang, idea_preset, idea_instruction],
        )
        vlm_preset.change(
            fn=lambda p, l, c: _on_preset_change(p, l, c, "vlm"),
            inputs=[vlm_preset, vlm_lang, vlm_instruction],
            outputs=[vlm_instruction],
        )
        vlm_lang.change(
            fn=lambda l, p, c: _on_lang_change(l, p, c, "vlm"),
            inputs=[vlm_lang, vlm_preset, vlm_instruction],
            outputs=[vlm_instruction],
        )
        refresh_vlm_presets_btn.click(
            fn=lambda p, l: _refresh_presets_for("vlm", p, l),
            inputs=[vlm_preset, vlm_lang],
            outputs=[vlm_lang, vlm_preset, vlm_instruction],
        )

        catalog_dd.change(
            fn=_on_catalog_change,
            inputs=[catalog_dd],
            outputs=[gguf_path, mmproj_path, gguf_status, model_name, download_btn, create_btn],
        )
        refresh_catalog_btn.click(
            fn=_refresh_catalog,
            inputs=[catalog_dd],
            outputs=[
                catalog_dd,
                gguf_path,
                mmproj_path,
                gguf_status,
                model_name,
                models_dir_box,
                download_btn,
                create_btn,
            ],
        )
        check_btn.click(
            fn=_check_connection,
            inputs=[api_url, catalog_dd, model_name],
            outputs=[conn_status, model_name],
        )
        start_btn.click(
            fn=_start_ollama,
            inputs=[api_url, catalog_dd, model_name],
            outputs=[conn_status, model_name],
            show_progress=True,
        )
        restart_btn.click(
            fn=_restart_ollama,
            inputs=[api_url, catalog_dd, model_name],
            outputs=[conn_status, model_name],
            show_progress=True,
        )
        unload_all_btn.click(
            fn=_unload_all,
            inputs=[api_url, catalog_dd, model_name],
            outputs=[conn_status, model_name],
            show_progress=True,
        )
        unload_selected_btn.click(
            fn=_unload_selected,
            inputs=[api_url, catalog_dd, model_name],
            outputs=[conn_status, model_name],
            show_progress=True,
        )
        download_btn.click(
            fn=_download_gguf,
            inputs=[catalog_dd, force_redownload],
            outputs=[gguf_path, mmproj_path, gguf_status, create_log, model_name],
            show_progress=True,
        )
        refresh_gguf_btn.click(
            fn=_refresh_gguf_status,
            inputs=[catalog_dd, gguf_path],
            outputs=[gguf_status, models_dir_box, gguf_path, mmproj_path],
        )
        create_btn.click(
            fn=_create_model,
            inputs=[api_url, model_name, gguf_path, mmproj_path],
            outputs=[create_log, model_name],
            show_progress=True,
        )
        idea_generate_btn.click(
            fn=_generate_idea,
            inputs=[
                api_url,
                model_name,
                idea,
                idea_preset,
                idea_instruction,
                idea_lang,
                idea_temperature,
                idea_top_p,
                idea_num_predict,
            ],
            outputs=[idea_prompt, idea_log],
            show_progress=True,
        )
        vlm_generate_btn.click(
            fn=_generate_vlm,
            inputs=[
                api_url,
                model_name,
                vlm_image,
                vlm_extra,
                vlm_preset,
                vlm_instruction,
                vlm_lang,
                vlm_temperature,
                vlm_top_p,
                vlm_num_predict,
            ],
            outputs=[vlm_prompt, vlm_log],
            show_progress=True,
        )
        wd_btn.click(
            fn=_generate_wd,
            inputs=[
                vlm_image,
                wd_repo,
                wd_general,
                wd_character,
                wd_underscore_to_space,
                wd_escape_parens,
            ],
            outputs=[vlm_prompt, vlm_log],
            show_progress=True,
        )

    return [(demo, "LLM Prompt (Ollama)", "llm_prompt_ollama")]


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
