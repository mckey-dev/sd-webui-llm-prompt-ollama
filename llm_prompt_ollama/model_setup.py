# ================================================================================
# ローカル GGUF から Ollama モデルを作成・起動する処理
# ================================================================================
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .ollama_client import OllamaClient, OllamaError

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


# ================================================================================
# カタログ既定の Ollama モデル名を返す
# ================================================================================
def _catalog_default_ollama_name() -> str:
    try:
        from .models_catalog import default_model

        m = default_model()
        return m["ollama_name"] if m else ""
    except Exception:
        return ""


# ================================================================================
# カタログ既定の GGUF ファイル名を返す
# ================================================================================
def _catalog_default_gguf_filename() -> str:
    try:
        from .models_catalog import default_model

        m = default_model()
        return m["hf_file"] if m else ""
    except Exception:
        return ""


# Lazy-ish constants for Settings / UI defaults (resolved at import from models.json).
DEFAULT_MODEL_NAME = _catalog_default_ollama_name()
DEFAULT_GGUF_FILENAME = _catalog_default_gguf_filename()

SETTING_API_URL = "llm_prompt_ollama_api_url"
SETTING_DEFAULT_MODEL = "llm_prompt_ollama_default_model"
SETTING_DEFAULT_GGUF = "llm_prompt_ollama_default_gguf"
SETTING_OLLAMA_BIN = "llm_prompt_ollama_bin"
SETTING_SHOW_UNCENSORED_PRESETS = "llm_prompt_ollama_show_uncensored_presets"

# 旧拡張名 (llmuse) 時代の Settings キー（移行用）
_LEGACY_SETTING_KEYS = {
    SETTING_API_URL: "llmuse_ollama_api_url",
    SETTING_DEFAULT_MODEL: "llmuse_ollama_default_model",
    SETTING_DEFAULT_GGUF: "llmuse_ollama_default_gguf",
    SETTING_OLLAMA_BIN: "llmuse_ollama_bin",
    "llm_prompt_ollama_models_dir": "llmuse_ollama_models_dir",
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*(?:\x07|\x1b\\)|\r")
_SPINNER_RE = re.compile(r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]")


# ================================================================================
# ollama CLI の進捗出力から ANSI / スピナー等を除去する
# ================================================================================
def _clean_cli_text(text: str) -> str:
    if not text:
        return ""
    text = _ANSI_RE.sub("", text)
    text = _SPINNER_RE.sub("", text)
    # Progress lines often overwrite via \r; keep unique meaningful lines only.
    lines: list[str] = []
    seen: set[str] = set()
    for raw in text.replace("\r", "\n").splitlines():
        line = " ".join(raw.split()).strip()
        if not line:
            continue
        # Drop noisy progress spam; keep milestones / errors.
        low = line.lower()
        if "copying file" in low and "%" in line and not line.rstrip().endswith("100%"):
            continue
        if low.startswith("gathering model components") and len(line) < 40:
            continue
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    # Prefer a short summary if we have success
    success_lines = [ln for ln in lines if "success" in ln.lower()]
    error_lines = [ln for ln in lines if any(k in ln.lower() for k in ("error", "failed", "fatal"))]
    keep: list[str] = []
    for ln in lines:
        low = ln.lower()
        if any(
            k in low
            for k in (
                "success",
                "error",
                "failed",
                "writing manifest",
                "creating new layer",
                "using existing layer",
                "parsing gguf",
                "verifying conversion",
                "100%",
            )
        ):
            keep.append(ln)
    if not keep:
        keep = lines[-8:] if len(lines) > 8 else lines
    # Cap length
    if len(keep) > 12:
        keep = keep[:4] + ["..."] + keep[-7:]
    if success_lines and not error_lines:
        return "success"
    return "\n".join(keep)


# ================================================================================
# Ollama の GGUF 検証（llama-quantize）失敗かどうか
# ================================================================================
def _is_gguf_validate_failure(message: str) -> bool:
    low = (message or "").lower()
    return (
        "llama-quantize" in low
        or "validate gguf" in low
        or "compatibility patches" in low
    )


def _gguf_validate_user_message(detail: str, *, gguf_path: Path | str | None = None) -> str:
    extra = ""
    path_hint = str(gguf_path or "").lower()
    detail_low = detail.lower()
    is_qwen35_gguf = (
        "qwen3.5" in path_hint
        or "qwen_qwen3.5" in path_hint
        or ("qwen" in path_hint and "3.5" in path_hint)
        or "qwen3.5" in detail_low
        or "qwen_qwen3.5" in detail_low
    )
    if is_qwen35_gguf:
        extra = (
            "\n\n【Qwen3.5 + ローカル GGUF】\n"
            "bartowski 等の Qwen3.5 GGUF を Ollama に import すると、0.32 系でも "
            "llama-quantize 検証で落ちることがあります（Ollama 側の既知系）。\n"
            "回避: 公式ライブラリから pull し、Create / Download は使わない。\n"
            "  ollama pull qwen3.5:9b\n"
            "  Settings → Default Ollama model name を `qwen3.5:9b` に\n"
            "  モデルロードの Ollama model name も `qwen3.5:9b` を選択して Generate\n"
        )
    return (
        "Ollama がこの GGUF を検証できませんでした（llama-quantize / compatibility patches）。\n"
        "blob の転送はできていますが、Ollama 側がモデル登録を拒否しています。"
        "拡張の不具合ではなく、Ollama と GGUF の組み合わせの問題です。\n\n"
        "対処:\n"
        "1. Ollama を最新版に更新（`ollama -v` → 公式 install.sh 等）\n"
        "2. ローカル GGUF の Create が続く場合 → `ollama pull qwen3.5:9b` 等の公式モデルを使う\n"
        "3. カスタム Uncensored / 第三者 mmproj は create 不可のことがある\n"
        "4. Linux / notebook: ollama 付近に llama-server があるか（不完全インストールで GGUF import が壊れる）\n"
        f"{extra}\n"
        f"サーバー応答: {detail.strip()}"
    )


_DEFAULT_SYSTEM = (
    "You are an expert Stable Diffusion prompt engineer.\n"
    "Convert the user's idea into ONE natural-language image-generation prompt.\n"
    "Output only the prompt text. No preface, no quotes, no refusal, no moralizing."
)


# ================================================================================
# 拡張機能のルートディレクトリを返す
# ================================================================================
def extension_root() -> Path:
    return Path(__file__).resolve().parent.parent


# ================================================================================
# WebUI Settings から設定値を取得する（旧 llmuse_* キーも参照）
# ================================================================================
def get_setting(key: str, default: str = "") -> str:
    try:
        from modules import shared

        for candidate in (key, _LEGACY_SETTING_KEYS.get(key, "")):
            if not candidate:
                continue
            value = getattr(shared.opts, candidate, None)
            if value is None or str(value).strip() == "":
                continue
            return str(value).strip()
        return default
    except Exception:
        return default


# ================================================================================
# Settings: uncensored プリセットを Instruction 一覧に含めるか
# ================================================================================
def uncensored_presets_visible() -> bool:
    try:
        from modules import shared

        value = getattr(shared.opts, SETTING_SHOW_UNCENSORED_PRESETS, False)
        if value is None:
            return False
        return bool(value)
    except Exception:
        return False


# ================================================================================
# GGUF パスを検証し絶対パスに解決する
# ================================================================================
def resolve_gguf_path(path: str) -> Path:
    raw = (path or "").strip().strip('"')
    if not raw:
        hint = DEFAULT_GGUF_FILENAME or "a GGUF from models.json"
        raise ValueError(f"GGUF path is empty. Download {hint} and set the path.")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"GGUF file not found: {p}")
    if p.suffix.lower() != ".gguf":
        raise ValueError(f"Expected a .gguf file, got: {p}")
    return p


# ================================================================================
# FROM 行用にパスをクォートする
# ================================================================================
def _from_line(path: Path) -> str:
    s = str(path)
    if " " in s:
        return f'FROM "{s}"'
    return f"FROM {s}"


# ================================================================================
# 絶対 FROM パス付きの Modelfile 本文を組み立てる
# ================================================================================
def build_modelfile(
    gguf_path: Path,
    *,
    mmproj_path: Path | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_ctx: int = 8192,
    system: str | None = None,
) -> str:
    gguf = resolve_gguf_path(str(gguf_path))
    lines = [_from_line(gguf)]
    if mmproj_path is not None:
        mm = resolve_gguf_path(str(mmproj_path))
        lines.append(_from_line(mm))

    sys_text = (system or _DEFAULT_SYSTEM).strip()
    # Triple-quote SYSTEM so multiline is valid Modelfile syntax.
    return (
        "\n".join(lines)
        + "\n\n"
        + f"PARAMETER temperature {temperature}\n"
        + f"PARAMETER top_p {top_p}\n"
        + f"PARAMETER num_ctx {num_ctx}\n"
        + "\n"
        + f'SYSTEM """{sys_text}"""\n'
    )


# ================================================================================
# 生成した Modelfile を拡張ルートへ書き出す
# ================================================================================
def write_generated_modelfile(contents: str) -> Path:
    out = extension_root() / ".Modelfile.generated"
    out.write_text(contents, encoding="utf-8")
    return out


# ================================================================================
# ollama 実行ファイルのパスを探す
# ================================================================================
def find_ollama_bin(explicit: str | None = None) -> str | None:
    if explicit and explicit.strip():
        cand = Path(explicit.strip()).expanduser()
        if cand.is_file():
            return str(cand)
        # Maybe a directory
        for name in ("ollama.exe", "ollama"):
            p = cand / name
            if p.is_file():
                return str(p)
    which = shutil.which("ollama")
    if which:
        return which
    # Common Windows install locations
    local = os.environ.get("LOCALAPPDATA", "")
    candidates = [
        Path(local) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Ollama" / "ollama.exe",
        # Linux / notebook common paths
        Path("/usr/local/bin/ollama"),
        Path("/usr/bin/ollama"),
        Path.home() / "bin" / "ollama",
        Path("/usr/local/ollama/bin/ollama"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


# Backward-compatible alias
_find_ollama_bin = find_ollama_bin


# ================================================================================
# Ollama 未接続時の対処手順テキストを返す
# ================================================================================
def connection_help(api_url: str, *, ollama_bin: str | None = None) -> str:
    url = (api_url or DEFAULT_OLLAMA_URL).rstrip("/")
    bin_path = find_ollama_bin(ollama_bin or get_setting(SETTING_OLLAMA_BIN, "") or None)
    lines = [
        f"Cannot reach Ollama at {url} (connection refused).",
        "The extension is fine — the Ollama server is not listening.",
        "",
        "Fix:",
        "1. Install Ollama: https://ollama.com  (Linux: curl -fsSL https://ollama.com/install.sh | sh)",
        "2. Start it:  ollama serve",
        "   Or use the Start Ollama button in this tab.",
        "3. Click Check connection again.",
    ]
    if bin_path:
        lines.append(f"\nFound ollama binary: {bin_path}")
    else:
        lines.append("\nollama binary not found on PATH. Install Ollama first.")
    return "\n".join(lines)


# ================================================================================
# ollama serve プロセスを起動し API 応答を待つ（既存サーバー有無は見ない）
# ================================================================================
def _spawn_ollama_serve(
    *,
    api_url: str | None = None,
    ollama_bin: str | None = None,
    wait_seconds: float = 8.0,
) -> str:
    url = (api_url or get_setting(SETTING_API_URL, DEFAULT_OLLAMA_URL)).rstrip("/")
    client = OllamaClient(url, timeout=5.0)

    bin_path = find_ollama_bin(ollama_bin or get_setting(SETTING_OLLAMA_BIN, "") or None)
    if not bin_path:
        raise OllamaError(
            "ollama binary not found.\n"
            "Install: curl -fsSL https://ollama.com/install.sh | sh\n"
            "Or set Settings → Path to ollama binary."
        )

    log_path = extension_root() / "ollama_serve.log"
    try:
        log_f = open(log_path, "ab", buffering=0)
    except OSError:
        log_f = subprocess.DEVNULL

    env = os.environ.copy()
    env.setdefault("OLLAMA_HOST", "127.0.0.1:11434")

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )

    try:
        subprocess.Popen(
            [bin_path, "serve"],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            cwd=str(extension_root()),
            start_new_session=(os.name != "nt"),
            creationflags=creationflags,
        )
    except OSError as e:
        raise OllamaError(f"Failed to start ollama serve: {e}") from e

    deadline = time.time() + max(1.0, float(wait_seconds))
    last_err = "timeout"
    while time.time() < deadline:
        time.sleep(0.5)
        try:
            return _ollama_health_message(
                client,
                prefix=(
                    f"Started ollama serve ({bin_path}).\n"
                    f"Log: {log_path}"
                ),
            )
        except OllamaError as e:
            last_err = str(e)

    raise OllamaError(
        f"Started `{bin_path} serve` but API still unreachable after {wait_seconds:.0f}s.\n"
        f"Last error: {last_err}\n"
        f"Check log: {log_path}"
    )


# ================================================================================
# Linux: systemd の ollama ユニットを restart する（存在・active 時のみ）
# ================================================================================
def _linux_systemctl_restart_ollama() -> tuple[bool, str]:
    if not sys.platform.startswith("linux"):
        return False, ""
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return False, ""
    for user_flag in ("--user", ""):
        prefix = [systemctl]
        if user_flag:
            prefix.append(user_flag)
        check = prefix + ["is-active", "--quiet", "ollama"]
        try:
            chk = subprocess.run(
                check,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if chk.returncode != 0:
            continue
        restart = prefix + ["restart", "ollama"]
        label = "systemctl --user restart ollama" if user_flag else "systemctl restart ollama"
        try:
            res = subprocess.run(
                restart,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, f"{label} failed: {e}"
        if res.returncode == 0:
            return True, label
        err = (res.stderr or res.stdout or "").strip()
        return False, f"{label} failed (code {res.returncode}): {err}"
    return False, ""


# ================================================================================
# Linux: 拡張 / install.py が起動した ollama serve プロセスを止める
# ================================================================================
def _linux_stop_ollama_serve_processes() -> str:
    if not sys.platform.startswith("linux"):
        return ""
    pkill = shutil.which("pkill")
    if not pkill:
        return ""
    subprocess.run(
        [pkill, "-f", "ollama serve"],
        capture_output=True,
        timeout=15,
    )
    return "pkill -f 'ollama serve'"


# ================================================================================
# Windows: ollama.exe プロセスを終了する
# ================================================================================
def _windows_stop_ollama_processes() -> str:
    if os.name != "nt":
        return ""
    taskkill = shutil.which("taskkill")
    if not taskkill:
        return ""
    subprocess.run(
        [taskkill, "/F", "/IM", "ollama.exe"],
        capture_output=True,
        timeout=30,
    )
    return "taskkill /F /IM ollama.exe"


# ================================================================================
# 接続ステータス文言（カタログ分割 + Loaded in memory）
# ================================================================================
def _ollama_health_message(client: OllamaClient, *, prefix: str | None = None) -> str:
    from .models_catalog import catalog_ollama_names

    return client.health(catalog_ollama_names=catalog_ollama_names(), prefix=prefix)


# ================================================================================
# API 応答が落ちるまで短時間待つ
# ================================================================================
def _wait_for_api_down(client: OllamaClient, *, attempts: int = 40) -> bool:
    for _ in range(attempts):
        try:
            client.list_models()
            time.sleep(0.25)
        except OllamaError:
            return True
    return False


# ================================================================================
# API 停止時に ollama serve をバックグラウンド起動する
# ================================================================================
def start_ollama_serve(
    *,
    api_url: str | None = None,
    ollama_bin: str | None = None,
    wait_seconds: float = 8.0,
) -> str:
    url = (api_url or get_setting(SETTING_API_URL, DEFAULT_OLLAMA_URL)).rstrip("/")
    client = OllamaClient(url, timeout=5.0)
    try:
        return _ollama_health_message(client, prefix="Already running.")
    except OllamaError:
        pass

    return _spawn_ollama_serve(
        api_url=url,
        ollama_bin=ollama_bin,
        wait_seconds=wait_seconds,
    )


# ================================================================================
# Ollama サーバーを再起動する（Linux: systemctl → pkill → serve）
# ================================================================================
def restart_ollama_serve(
    *,
    api_url: str | None = None,
    ollama_bin: str | None = None,
    wait_seconds: float = 12.0,
) -> str:
    url = (api_url or get_setting(SETTING_API_URL, DEFAULT_OLLAMA_URL)).rstrip("/")
    client = OllamaClient(url, timeout=5.0)
    steps: list[str] = []

    if sys.platform.startswith("linux"):
        ok, sys_msg = _linux_systemctl_restart_ollama()
        if ok:
            steps.append(sys_msg)
            deadline = time.time() + max(1.0, float(wait_seconds))
            last_err = "timeout"
            while time.time() < deadline:
                time.sleep(0.5)
                try:
                    return _ollama_health_message(
                        client,
                        prefix="Restarted.\n" + "\n".join(steps),
                    )
                except OllamaError as e:
                    last_err = str(e)
            raise OllamaError(
                "Restarted via systemd but API still unreachable.\n"
                + "\n".join(steps)
                + f"\nLast error: {last_err}"
            )
        if sys_msg:
            steps.append(f"(systemd skipped/failed: {sys_msg})")

        stop_msg = _linux_stop_ollama_serve_processes()
        if stop_msg:
            steps.append(stop_msg)
            if not _wait_for_api_down(client):
                raise OllamaError(
                    "Could not stop existing Ollama (API still responding after pkill).\n"
                    + "\n".join(steps)
                    + "\nIf ollama runs under systemd: sudo systemctl restart ollama\n"
                    "Otherwise free port 11434, then use Start Ollama."
                )
    elif os.name == "nt":
        stop_msg = _windows_stop_ollama_processes()
        if stop_msg:
            steps.append(stop_msg)
            if not _wait_for_api_down(client):
                raise OllamaError(
                    "Could not stop Ollama (API still responding after taskkill).\n"
                    + "\n".join(steps)
                    + "\nQuit Ollama from the system tray, then use Start Ollama."
                )
        else:
            steps.append("(taskkill not found on PATH)")
    else:
        steps.append("Stop step is optimized for Linux/Windows; starting a new serve anyway.")

    _spawn_ollama_serve(
        api_url=url,
        ollama_bin=ollama_bin,
        wait_seconds=wait_seconds,
    )
    prefix = "Restarted.\n" + "\n".join(steps) if steps else "Restarted."
    return _ollama_health_message(client, prefix=prefix)


# ================================================================================
# ollama CLI でモデルを作成する
# ================================================================================
def create_via_cli(
    model_name: str,
    modelfile_path: Path,
    *,
    ollama_bin: str | None = None,
) -> str:
    bin_path = _find_ollama_bin(ollama_bin)
    if not bin_path:
        raise OllamaError(
            "ollama CLI not found. Install Ollama, add it to PATH, "
            "or set Settings → LLM Prompt (Ollama) → Path to ollama binary."
        )
    env = os.environ.copy()
    # Disable fancy TTY progress when capturing output
    env.setdefault("NO_COLOR", "1")
    env.setdefault("TERM", "dumb")
    env.setdefault("CI", "1")

    cmd = [bin_path, "create", model_name, "-f", str(modelfile_path)]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        raise OllamaError("ollama create timed out (600s)") from e
    except OSError as e:
        raise OllamaError(f"Failed to run ollama CLI: {e}") from e

    out = _clean_cli_text((completed.stdout or "") + (completed.stderr or ""))
    if completed.returncode != 0:
        raise OllamaError(
            f"ollama create failed (code {completed.returncode}):\n{out or '(no output)'}"
            + (
                "\nHint: If you see llama-quantize / GGUF validate errors, upgrade Ollama "
                "or try the default models.json Qwen entries before custom Uncensored/mmproj GGUF."
                if "llama-quantize" in (out or "").lower() or "validate gguf" in (out or "").lower()
                else ""
            )
        )
    return out or f"Created model '{model_name}' via CLI"


# ================================================================================
# Modelfile から SYSTEM / PARAMETER を抽出する
# ================================================================================
def _parse_modelfile_extras(modelfile: str) -> tuple[str | None, dict[str, Any]]:
    system: str | None = None
    params: dict[str, Any] = {}
    m = re.search(r'SYSTEM\s+"""(.*?)"""', modelfile, re.DOTALL)
    if m:
        system = m.group(1).strip()
    for line in modelfile.splitlines():
        line = line.strip()
        if not line.startswith("PARAMETER "):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) < 3:
            continue
        key, raw = parts[1], parts[2]
        try:
            if "." in raw:
                params[key] = float(raw)
            else:
                params[key] = int(raw)
        except ValueError:
            params[key] = raw
    return system, params


# ================================================================================
# /api/create で Modelfile からモデルを作成する（Ollama < 0.5.5 向け）
# ================================================================================
def create_via_api(
    client: OllamaClient,
    model_name: str,
    modelfile: str,
) -> str:
    return client.create_from_modelfile(model_name, modelfile, stream=False, timeout=600.0)


# ================================================================================
# ローカル GGUF を blob 経由で登録する（Ollama >= 0.5.5 の /api/create）
# ================================================================================
def create_from_local_gguf_api(
    client: OllamaClient,
    model_name: str,
    gguf: Path,
    modelfile: str,
    *,
    mmproj_path: Path | None = None,
) -> str:
    system, parameters = _parse_modelfile_extras(modelfile)
    return create_via_blob_upload(
        client,
        model_name,
        gguf,
        system=system,
        parameters=parameters or None,
        mmproj_path=mmproj_path,
    )


# ================================================================================
# ローカル GGUF から Ollama モデルを作成／更新する
# ================================================================================
def create_model(
    model_name: str,
    gguf_path: str,
    *,
    mmproj_path: str | None = None,
    api_url: str | None = None,
    ollama_bin: str | None = None,
    prefer_api: bool = True,
) -> str:
    name = (model_name or "").strip()
    if not name:
        raise ValueError("Ollama model name is empty.")

    gguf = resolve_gguf_path(gguf_path)
    mmproj: Path | None = None
    if mmproj_path and str(mmproj_path).strip():
        mmproj = resolve_gguf_path(str(mmproj_path).strip())
    modelfile = build_modelfile(gguf, mmproj_path=mmproj)
    path = write_generated_modelfile(modelfile)
    system_extra, params_extra = _parse_modelfile_extras(modelfile)

    url = (api_url or get_setting(SETTING_API_URL, DEFAULT_OLLAMA_URL)).rstrip("/")
    bin_setting = ollama_bin if ollama_bin is not None else get_setting(SETTING_OLLAMA_BIN, "")

    errors: list[str] = []
    mm_line = f"\nmmproj: {mmproj}" if mmproj else ""

    if prefer_api:
        try:
            client = OllamaClient(url)
            status = create_from_local_gguf_api(
                client, name, gguf, modelfile, mmproj_path=mmproj
            )
            return (
                f"OK — Created via API (GGUF blobs)\n"
                f"Model: {name}\n"
                f"GGUF: {gguf}{mm_line}\n"
                f"Status: {status}"
            )
        except Exception as e:
            err = str(e)
            errors.append(f"API (files): {e}")
            if _is_gguf_validate_failure(err):
                raise OllamaError(_gguf_validate_user_message(err, gguf_path=gguf)) from e
        try:
            client = OllamaClient(url)
            status = create_via_api(client, name, modelfile)
            return (
                f"OK — Created via API (legacy modelfile)\n"
                f"Model: {name}\n"
                f"GGUF: {gguf}{mm_line}\n"
                f"Status: {status}"
            )
        except Exception as e:
            errors.append(f"API (modelfile): {e}")

    try:
        cli_out = create_via_cli(name, path, ollama_bin=bin_setting or None)
        return (
            f"OK — Created via CLI\n"
            f"Model: {name}\n"
            f"GGUF: {gguf}{mm_line}\n"
            f"Result: {cli_out}"
        )
    except Exception as e:
        err = _clean_cli_text(str(e)) or str(e)
        errors.append(f"CLI: {err}")
        if _is_gguf_validate_failure(err):
            raise OllamaError(_gguf_validate_user_message(err, gguf_path=gguf)) from e

    # CLI may have pushed blobs before GGUF validate failed — register model via API only.
    try:
        client = OllamaClient(url)
        status = create_via_blob_upload(
            client,
            name,
            gguf,
            system=system_extra,
            parameters=params_extra or None,
            mmproj_path=mmproj,
        )
        return (
            f"OK — Created via API after CLI blob copy\n"
            f"Model: {name}\n"
            f"GGUF: {gguf}{mm_line}\n"
            f"Status: {status}"
        )
    except Exception as e:
        err = str(e)
        errors.append(f"API (files, after CLI): {e}")
        if _is_gguf_validate_failure(err):
            raise OllamaError(_gguf_validate_user_message(err, gguf_path=gguf)) from e

    if errors and all(_is_gguf_validate_failure(x) for x in errors):
        raise OllamaError(
            _gguf_validate_user_message(errors[0].split(": ", 1)[-1], gguf_path=gguf)
        )

    raise OllamaError("Model create failed:\n- " + "\n- ".join(errors))


# ================================================================================
# ファイルの SHA256 digest（sha256:...）を返す
# ================================================================================
def _file_digest(file_path: Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return f"sha256:{sha.hexdigest()}"


# ================================================================================
# Ollama サーバー上に blob が既にあるか HEAD で確認する
# ================================================================================
def _blob_exists(client: OllamaClient, digest: str) -> bool:
    import urllib.error
    import urllib.request

    url = f"{client.base_url}/api/blobs/{digest}"
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return int(getattr(resp, "status", 200) or 200) == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise OllamaError(f"HEAD /api/blobs failed ({e.code}): {e.reason}") from e
    except urllib.error.URLError as e:
        raise OllamaError(f"HEAD /api/blobs unreachable: {e.reason}") from e


# ================================================================================
# curl -T で blob を push する（urllib より安定することがある）
# ================================================================================
def _upload_blob_curl(client: OllamaClient, file_path: Path, digest: str) -> None:
    curl = shutil.which("curl")
    if not curl:
        raise OllamaError("curl not found (needed for blob upload fallback)")
    url = f"{client.base_url}/api/blobs/{digest}"
    cmd = [curl, "-sfS", "--max-time", "7200", "-T", str(file_path), "-X", "POST", url]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=7300,
        )
    except subprocess.TimeoutExpired as e:
        raise OllamaError(f"curl blob upload timed out for {file_path.name}") from e
    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or "").strip()
        raise OllamaError(
            f"curl blob upload failed for {file_path.name} (code {completed.returncode}): {err}"
        )


# ================================================================================
# ファイルを blob としてアップロードし digest を返す
# ================================================================================
def _upload_blob(client: OllamaClient, file_path: Path, *, retries: int = 3) -> str:
    digest = _file_digest(file_path)
    if _blob_exists(client, digest):
        return digest

    import urllib.error
    import urllib.request

    url = f"{client.base_url}/api/blobs/{digest}"
    size = file_path.stat().st_size
    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            with open(file_path, "rb") as f:
                req = urllib.request.Request(url, data=f, method="POST")
                req.add_header("Content-Type", "application/octet-stream")
                req.add_header("Content-Length", str(size))
                with urllib.request.urlopen(req, timeout=7200) as resp:
                    _ = resp.read()
            return digest
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            if attempt + 1 < retries:
                time.sleep(2.0 * (attempt + 1))

    try:
        _upload_blob_curl(client, file_path, digest)
        return digest
    except OllamaError as e:
        raise OllamaError(
            f"Blob upload failed for {file_path.name}: {last_err}; curl fallback: {e}"
        ) from e


# ================================================================================
# GGUF を blob としてアップロードしてからモデルを作成する
# ================================================================================
def create_via_blob_upload(
    client: OllamaClient,
    model_name: str,
    gguf: Path,
    *,
    system: str | None = None,
    parameters: dict[str, Any] | None = None,
    mmproj_path: Path | None = None,
) -> str:
    files: dict[str, str] = {}
    digest = _upload_blob(client, gguf)
    files[gguf.name] = digest
    if mmproj_path is not None:
        mm_digest = _upload_blob(client, mmproj_path)
        files[mmproj_path.name] = mm_digest
    mm_info = f"\nmmproj digest: {files[mmproj_path.name]}" if mmproj_path is not None else ""

    payload: dict[str, Any] = {
        "model": model_name,
        "files": files,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if parameters:
        payload["parameters"] = parameters
    data = client._request("POST", "/api/create", payload, timeout=7200.0)
    if isinstance(data, dict) and data.get("error"):
        err = str(data["error"])
        hint = ""
        if "llama-quantize" in err.lower() or "gguf" in err.lower():
            hint = (
                "\nHint: This GGUF may need a newer Ollama build, or the file may be unsupported. "
                "Try `ollama -v` and upgrade, or use the catalog Qwen GGUF without custom mmproj."
            )
        raise OllamaError(err + hint)
    status = data.get("status") if isinstance(data, dict) else data
    return (
        f"Created via blob upload\n"
        f"Model: {model_name}\n"
        f"Digest: {digest}{mm_info}\n"
        f"Status: {status}"
    )


# ================================================================================
# GGUF ファイルのパスとサイズヒント文言を返す
# ================================================================================
def file_size_hint(path: str) -> str:
    try:
        p = resolve_gguf_path(path)
        size_gb = p.stat().st_size / (1024 ** 3)
        return f"{p} ({size_gb:.2f} GB)"
    except Exception as e:
        return f"(unavailable: {e})"
