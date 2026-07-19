# ================================================================================
# ローカル GGUF から Ollama モデルを作成・起動する処理
# ================================================================================
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

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

_DEFAULT_SYSTEM = (
    "You are an expert Stable Diffusion prompt engineer.\n"
    "Convert the user's idea into ONE fluent English image-generation prompt.\n"
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
# 絶対 FROM パス付きの Modelfile 本文を組み立てる
# ================================================================================
def build_modelfile(
    gguf_path: Path,
    *,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_ctx: int = 8192,
    system: str | None = None,
) -> str:
    gguf = resolve_gguf_path(str(gguf_path))
    # Ollama accepts absolute paths; quote if spaces.
    from_path = str(gguf)
    if " " in from_path:
        from_line = f'FROM "{from_path}"'
    else:
        from_line = f"FROM {from_path}"

    sys_text = (system or _DEFAULT_SYSTEM).strip()
    # Triple-quote SYSTEM so multiline is valid Modelfile syntax.
    return (
        f"{from_line}\n"
        f"\n"
        f"PARAMETER temperature {temperature}\n"
        f"PARAMETER top_p {top_p}\n"
        f"PARAMETER num_ctx {num_ctx}\n"
        f"\n"
        f'SYSTEM """{sys_text}"""\n'
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
# API 停止時に ollama serve をバックグラウンド起動する
# ================================================================================
def start_ollama_serve(
    *,
    api_url: str | None = None,
    ollama_bin: str | None = None,
    wait_seconds: float = 8.0,
) -> str:
    import time

    url = (api_url or get_setting(SETTING_API_URL, DEFAULT_OLLAMA_URL)).rstrip("/")
    client = OllamaClient(url, timeout=5.0)
    try:
        status = client.health()
        return f"Already running.\n{status}"
    except OllamaError:
        pass

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
    # Ensure local API binds as expected in notebooks / headless hosts
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
            status = client.health()
            return (
                f"Started ollama serve ({bin_path}).\n"
                f"Log: {log_path}\n"
                f"{status}"
            )
        except OllamaError as e:
            last_err = str(e)

    raise OllamaError(
        f"Started `{bin_path} serve` but API still unreachable after {wait_seconds:.0f}s.\n"
        f"Last error: {last_err}\n"
        f"Check log: {log_path}"
    )


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
        )
    return out or f"Created model '{model_name}' via CLI"


# ================================================================================
# /api/create で Modelfile からモデルを作成する
# ================================================================================
def create_via_api(
    client: OllamaClient,
    model_name: str,
    modelfile: str,
) -> str:
    return client.create_from_modelfile(model_name, modelfile, stream=False, timeout=600.0)


# ================================================================================
# ローカル GGUF から Ollama モデルを作成／更新する
# ================================================================================
def create_model(
    model_name: str,
    gguf_path: str,
    *,
    api_url: str | None = None,
    ollama_bin: str | None = None,
    prefer_api: bool = True,
) -> str:
    name = (model_name or "").strip()
    if not name:
        raise ValueError("Ollama model name is empty.")

    gguf = resolve_gguf_path(gguf_path)
    modelfile = build_modelfile(gguf)
    path = write_generated_modelfile(modelfile)

    url = (api_url or get_setting(SETTING_API_URL, DEFAULT_OLLAMA_URL)).rstrip("/")
    bin_setting = ollama_bin if ollama_bin is not None else get_setting(SETTING_OLLAMA_BIN, "")

    errors: list[str] = []

    if prefer_api:
        try:
            client = OllamaClient(url)
            status = create_via_api(client, name, modelfile)
            return (
                f"OK — Created via API\n"
                f"Model: {name}\n"
                f"GGUF: {gguf}\n"
                f"Status: {status}"
            )
        except Exception as e:
            errors.append(f"API: {e}")

    try:
        cli_out = create_via_cli(name, path, ollama_bin=bin_setting or None)
        return (
            f"OK — Created via CLI\n"
            f"Model: {name}\n"
            f"GGUF: {gguf}\n"
            f"Result: {cli_out}"
        )
    except Exception as e:
        errors.append(f"CLI: {_clean_cli_text(str(e)) or e}")

    # Last resort: upload blob + /api/create files (slow for large GGUF)
    try:
        digest_msg = create_via_blob_upload(OllamaClient(url), name, gguf, modelfile)
        return digest_msg
    except Exception as e:
        errors.append(f"Blob upload: {e}")

    raise OllamaError("Model create failed:\n- " + "\n- ".join(errors))


# ================================================================================
# GGUF を blob としてアップロードしてからモデルを作成する
# ================================================================================
def create_via_blob_upload(
    client: OllamaClient,
    model_name: str,
    gguf: Path,
    modelfile: str,
) -> str:
    sha = hashlib.sha256()
    with open(gguf, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    digest = f"sha256:{sha.hexdigest()}"

    # Upload blob (curl -T uses PUT)
    import urllib.request

    url = f"{client.base_url}/api/blobs/{digest}"
    size = gguf.stat().st_size
    with open(gguf, "rb") as f:
        req = urllib.request.Request(url, data=f, method="PUT")
        req.add_header("Content-Type", "application/octet-stream")
        req.add_header("Content-Length", str(size))
        with urllib.request.urlopen(req, timeout=3600) as resp:
            _ = resp.read()

    payload = {
        "model": model_name,
        "name": model_name,
        "files": {gguf.name: digest},
        "stream": False,
    }
    _ = modelfile  # reserved for future SYSTEM/PARAMETER injection
    data = client._request("POST", "/api/create", payload, timeout=600.0)
    status = data.get("status") if isinstance(data, dict) else data
    return (
        f"Created via blob upload\n"
        f"Model: {model_name}\n"
        f"Digest: {digest}\n"
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
