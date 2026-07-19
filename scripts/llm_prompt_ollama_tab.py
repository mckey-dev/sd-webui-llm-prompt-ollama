# ================================================================================
# LLM Prompt (Ollama) の Gradio UI タブを登録する
# ================================================================================
from __future__ import annotations

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
)
from llm_prompt_ollama.model_setup import (
    DEFAULT_GGUF_FILENAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_OLLAMA_URL,
    SETTING_API_URL,
    SETTING_DEFAULT_GGUF,
    SETTING_DEFAULT_MODEL,
    SETTING_OLLAMA_BIN,
    connection_help,
    create_model,
    get_setting,
    start_ollama_serve,
)
from llm_prompt_ollama.models_catalog import (
    choices_for_ui,
    default_model_id,
    get_model,
    has_catalog_models,
    reload_models_catalog,
    require_model,
)
from llm_prompt_ollama.ollama_client import OllamaClient, OllamaError
from llm_prompt_ollama.presets import (
    DEFAULT_LANG,
    DEFAULT_PRESET,
    LANG_CHOICES,
    PRESET_CHOICES,
    instruction_for_preset,
)


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
    # Settings override applies only when viewing the catalog default entry.
    if custom and entry["id"] == default_model_id():
        return custom
    return str(gguf_path_for(entry["id"]))


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
# カタログ選択変更時に GGUF パス・状態・モデル名を更新する
# ================================================================================
def _on_catalog_change(catalog_id: str):
    entry = get_model(catalog_id)
    if not entry:
        return "", "", gr.update(), *_catalog_button_updates(False)
    # Changing catalog always replaces path/status with the entry's expected values.
    return (
        _gguf_for_catalog(entry["id"]),
        describe_gguf_status(entry["id"]),
        gr.update(value=entry["ollama_name"]),
        *_catalog_button_updates(True),
    )


# ================================================================================
# 選択中カタログモデルの GGUF をダウンロードする
# ================================================================================
def _download_gguf(catalog_id: str, force: bool):
    if not has_catalog_models():
        return "", "", "No models in models.json.", gr.update()
    try:
        path, msg = download_gguf(catalog_id, force=bool(force))
        entry = require_model(catalog_id)
        status = describe_gguf_status(entry["id"])
        return str(path), status, msg, gr.update(value=entry["ollama_name"])
    except Exception as e:
        traceback.print_exc()
        entry = get_model(catalog_id)
        if not entry:
            return "", "", f"Download failed: {type(e).__name__}: {e}", gr.update()
        return (
            _gguf_for_catalog(entry["id"]),
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
        return "", str(get_models_dir()), ""
    status = describe_gguf_status(entry["id"])
    # Keep a user-edited path; fill only when still blank.
    path = (current_path or "").strip() or _gguf_for_catalog(entry["id"])
    return status, str(get_models_dir()), path


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
    # Catalog reload always syncs path/status to the selected entry.
    return (
        gr.update(choices=choices, value=value),
        _gguf_for_catalog(entry["id"]),
        describe_gguf_status(entry["id"]),
        gr.update(value=entry["ollama_name"]),
        str(get_models_dir()),
        *_catalog_button_updates(True),
    )


# ================================================================================
# プリセット変更時にインストラクション文を差し替える
# ================================================================================
def _on_preset_change(preset: str, lang: str, _current: str):
    if preset == "Custom":
        return gr.update()
    return instruction_for_preset(preset, lang)


# ================================================================================
# 言語変更時にインストラクション文を差し替える
# ================================================================================
def _on_lang_change(lang: str, preset: str, _current: str):
    if preset == "Custom":
        return gr.update()
    return instruction_for_preset(preset, lang)


# ================================================================================
# Ollama 接続確認とモデル一覧の更新を行う
# ================================================================================
def _check_connection(api_url: str):
    try:
        client = _client(api_url)
        status = client.health()
        models = client.list_models()
        choices = models or [_default_model()]
        value = _default_model() if _default_model() in choices else (choices[0] if choices else _default_model())
        return status, gr.update(choices=choices, value=value)
    except OllamaError as e:
        help_text = connection_help(api_url or _api_url())
        return f"{e}\n\n{help_text}", gr.update()
    except Exception as e:
        traceback.print_exc()
        return f"Connection failed: {e}", gr.update()


# ================================================================================
# Ollama サーバーを起動しモデル一覧を更新する
# ================================================================================
def _start_ollama(api_url: str):
    try:
        msg = start_ollama_serve(
            api_url=api_url or _api_url(),
            ollama_bin=get_setting(SETTING_OLLAMA_BIN, "") or None,
        )
        models = []
        try:
            models = _client(api_url).list_models()
        except Exception:
            pass
        choices = models or [_default_model()]
        value = _default_model() if _default_model() in choices else (choices[0] if choices else _default_model())
        return msg, gr.update(choices=choices, value=value)
    except OllamaError as e:
        return str(e), gr.update()
    except Exception as e:
        traceback.print_exc()
        return f"Start failed: {e}", gr.update()


# ================================================================================
# Ollama のモデル一覧を再取得する
# ================================================================================
def _refresh_models(api_url: str, current: str):
    try:
        models = _client(api_url).list_models()
        choices = models or [current or _default_model()]
        value = current if current in choices else (choices[0] if choices else _default_model())
        return gr.update(choices=choices, value=value), f"Models refreshed ({len(models)})."
    except OllamaError as e:
        return gr.update(), f"Refresh failed: {e}\n\n{connection_help(api_url or _api_url())}"
    except Exception as e:
        traceback.print_exc()
        return gr.update(), f"Refresh failed: {e}"


# ================================================================================
# ローカル GGUF から Ollama モデルを作成／更新する
# ================================================================================
def _create_model(api_url: str, model_name: str, gguf_path: str):
    try:
        msg = create_model(
            model_name=model_name,
            gguf_path=gguf_path,
            api_url=api_url or _api_url(),
            ollama_bin=get_setting(SETTING_OLLAMA_BIN, "") or None,
            prefer_api=True,
        )
        models = []
        try:
            models = _client(api_url).list_models()
        except Exception:
            pass
        choices = models or [model_name]
        value = model_name if model_name in choices else (choices[0] if choices else model_name)
        return msg, gr.update(choices=choices, value=value)
    except Exception as e:
        traceback.print_exc()
        return f"Create failed: {type(e).__name__}: {e}", gr.update()


# ================================================================================
# アイデアから SD 用プロンプトを生成する
# ================================================================================
def _generate(
    api_url: str,
    model_name: str,
    idea: str,
    preset: str,
    instruction: str,
    temperature: float,
    top_p: float,
    num_predict: int,
):
    try:
        if not (idea or "").strip():
            raise ValueError("Idea text is empty.")
        if not (model_name or "").strip():
            raise ValueError("Model name is empty. Check connection or create the model first.")

        system = instruction if preset == "Custom" else (instruction or instruction_for_preset(preset))
        user_content = (idea or "").strip()

        client = _client(api_url)
        prompt = client.chat(
            model=model_name.strip(),
            user_content=user_content,
            system=system,
            temperature=float(temperature),
            top_p=float(top_p),
            num_predict=int(num_predict),
            think=False,
        )
        if not (prompt or "").strip():
            return "", "Done — but model returned empty text. Try another idea or recreate the model."
        return prompt, f"Done ({len(prompt)} chars)"
    except Exception as e:
        traceback.print_exc()
        return "", f"Generate failed: {type(e).__name__}: {e}"


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

    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown(
            "### LLM Prompt (Ollama)\n"
            "テキストのアイデアから Stable Diffusion 用プロンプトを生成します。  \n"
            "利用可能な GGUF は拡張直下の **`models.json`** で管理します。  \n"
            "1. Ollama を起動（**Start Ollama**）→ **Check connection**  \n"
            "2. **Catalog model** を選び **Download GGUF**（初回）→ **Create / Update model**  \n"
            "3. アイデアを入力 → **Generate prompt** → Send to txt2img / img2img"
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("Ollama connection", open=True):
                    api_url = gr.Textbox(
                        label="Ollama API URL",
                        value=_api_url(),
                        lines=1,
                    )
                    with gr.Row():
                        check_btn = gr.Button("Check connection", variant="primary")
                        start_btn = gr.Button("Start Ollama")
                        refresh_btn = gr.Button("Refresh models")
                    conn_status = gr.Textbox(label="Connection status", lines=8, interactive=False)

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
                        lines=2,
                    )
                    # Must stay blank on startup; do not restore from ui-config.json.
                    gguf_status.do_not_save_to_config = True
                    force_redownload = gr.Checkbox(label="Force re-download", value=False)
                    with gr.Row():
                        download_btn = gr.Button(
                            "Download GGUF",
                            variant="primary",
                            interactive=has_catalog,
                        )
                        refresh_gguf_btn = gr.Button("↻ Status")
                    model_name = gr.Dropdown(
                        label="Ollama model name",
                        choices=[initial_ollama] if initial_ollama else [],
                        value=initial_ollama or None,
                        allow_custom_value=True,
                    )
                    gguf_path = gr.Textbox(
                        label="Local GGUF path",
                        value="",
                        lines=1,
                        placeholder="Catalog 選択・更新・Download で自動入力（手動編集可）",
                    )
                    # Must stay blank on startup; do not restore from ui-config.json.
                    gguf_path.do_not_save_to_config = True
                    create_btn = gr.Button(
                        "Create / Update model",
                        variant="primary",
                        interactive=has_catalog,
                    )
                    create_log = gr.Textbox(label="Create / Download log", lines=8, interactive=False)

                with gr.Accordion("Generation", open=True):
                    idea = gr.Textbox(
                        label="Idea / description (Japanese OK)",
                        lines=8,
                        placeholder="例: 雨の夜のネオン街を歩く銀髪の少女、サイバーパンク、映画的な照明",
                    )
                    lang = gr.Radio(
                        choices=LANG_CHOICES,
                        value=DEFAULT_LANG,
                        label="Instruction language",
                    )
                    preset = gr.Dropdown(
                        label="Instruction preset",
                        choices=PRESET_CHOICES,
                        value=DEFAULT_PRESET,
                        allow_custom_value=False,
                    )
                    instruction = gr.Textbox(
                        label="Instruction (editable — use Custom to keep edits)",
                        lines=8,
                        value=instruction_for_preset(DEFAULT_PRESET, DEFAULT_LANG),
                    )
                    temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
                    num_predict = gr.Slider(
                        64, 2048, value=512, step=32, label="Max tokens (num_predict)"
                    )
                    generate_btn = gr.Button("Generate prompt", variant="primary")

            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Generated prompt",
                    lines=18,
                    elem_id="llm_prompt_ollama_generated_prompt",
                )
                with gr.Row():
                    copy_btn = gr.Button(
                        "Copy prompt",
                        elem_id="llm_prompt_ollama_copy_prompt_btn",
                        scale=1,
                    )
                    buttons = parameters_copypaste.create_buttons(["txt2img", "img2img"])
                log = gr.Textbox(label="Log", lines=6, interactive=False)

        for tabname, button in buttons.items():
            parameters_copypaste.register_paste_params_button(
                parameters_copypaste.ParamBinding(
                    paste_button=button,
                    tabname=tabname,
                    source_text_component=prompt,
                )
            )

        # ================================================================================
        # クリップボードコピー結果のフィードバック文言を返す
        # ================================================================================
        def _copy_feedback(text: str):
            if (text or "").strip():
                return "Copied to clipboard."
            return "Nothing to copy."

        copy_btn.click(fn=_copy_feedback, inputs=[prompt], outputs=[log])

        preset.change(
            fn=_on_preset_change,
            inputs=[preset, lang, instruction],
            outputs=[instruction],
        )
        lang.change(
            fn=_on_lang_change,
            inputs=[lang, preset, instruction],
            outputs=[instruction],
        )
        catalog_dd.change(
            fn=_on_catalog_change,
            inputs=[catalog_dd],
            outputs=[gguf_path, gguf_status, model_name, download_btn, create_btn],
        )
        refresh_catalog_btn.click(
            fn=_refresh_catalog,
            inputs=[catalog_dd],
            outputs=[
                catalog_dd,
                gguf_path,
                gguf_status,
                model_name,
                models_dir_box,
                download_btn,
                create_btn,
            ],
        )
        check_btn.click(
            fn=_check_connection,
            inputs=[api_url],
            outputs=[conn_status, model_name],
        )
        start_btn.click(
            fn=_start_ollama,
            inputs=[api_url],
            outputs=[conn_status, model_name],
            show_progress=True,
        )
        refresh_btn.click(
            fn=_refresh_models,
            inputs=[api_url, model_name],
            outputs=[model_name, log],
        )
        download_btn.click(
            fn=_download_gguf,
            inputs=[catalog_dd, force_redownload],
            outputs=[gguf_path, gguf_status, create_log, model_name],
            show_progress=True,
        )
        refresh_gguf_btn.click(
            fn=_refresh_gguf_status,
            inputs=[catalog_dd, gguf_path],
            outputs=[gguf_status, models_dir_box, gguf_path],
        )
        create_btn.click(
            fn=_create_model,
            inputs=[api_url, model_name, gguf_path],
            outputs=[create_log, model_name],
            show_progress=True,
        )
        generate_btn.click(
            fn=_generate,
            inputs=[
                api_url,
                model_name,
                idea,
                preset,
                instruction,
                temperature,
                top_p,
                num_predict,
            ],
            outputs=[prompt, log],
            show_progress=True,
        )

    return [(demo, "LLM Prompt (Ollama)", "llm_prompt_ollama")]


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)
