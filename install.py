# ================================================================================
# Ollama の導入・起動を保証するインストールスクリプト
# Linux では未導入時に公式 install.sh を実行し、全 OS で API 停止時に serve を起動する
# Windows / macOS では自動インストールせず手動手順を表示する
# pip パッケージは入れず、例外でも WebUI 起動を止めない
# ================================================================================
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import launch
except ImportError:
    launch = None  # type: ignore

PREFIX = "[sd-webui-llm-prompt-ollama]"
DEFAULT_API = "http://127.0.0.1:11434"
INSTALL_SH_URL = "https://ollama.com/install.sh"


# ================================================================================
# 拡張機能のルートディレクトリを返す
# ================================================================================
def _extension_root() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "llm_prompt_ollama").is_dir():
        return here
    if here.name == "scripts" and (here.parent / "llm_prompt_ollama").is_dir():
        print(
            f"{PREFIX} WARNING: install.py is under scripts/; "
            f"expected at extension root. Using {here.parent}",
            flush=True,
        )
        return here.parent
    return here


EXT_ROOT = _extension_root()


# ================================================================================
# WebUI の skip_install 指定があるか判定する
# ================================================================================
def _skip_install() -> bool:
    if launch is None:
        return False
    try:
        return bool(getattr(launch.args, "skip_install", False))
    except Exception:
        return False


# ================================================================================
# プレフィックス付きでログを出力する
# ================================================================================
def _log(msg: str) -> None:
    print(f"{PREFIX} {msg}", flush=True)


# ================================================================================
# ollama 実行ファイルのパスを探す
# ================================================================================
def _find_ollama_bin() -> str | None:
    which = shutil.which("ollama")
    if which:
        return which
    candidates = [
        Path("/usr/local/bin/ollama"),
        Path("/usr/bin/ollama"),
        Path.home() / "bin" / "ollama",
        Path("/usr/local/ollama/bin/ollama"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Ollama" / "ollama.exe",
        Path("/Applications/Ollama.app/Contents/Resources/ollama"),
    ]
    for c in candidates:
        try:
            if c.is_file():
                return str(c)
        except OSError:
            continue
    return None


# ================================================================================
# Ollama API が応答しているか確認する
# ================================================================================
def _api_up(base_url: str = DEFAULT_API, timeout: float = 3.0) -> bool:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _ = resp.read(64)
        return True
    except Exception:
        return False


# ================================================================================
# シェルコマンドを実行する（launch.run があれば利用）
# ================================================================================
def _run_cmd(cmd: str, *, desc: str, errdesc: str, env: dict | None = None) -> None:
    if launch is not None and hasattr(launch, "run"):
        launch.run(cmd, desc=desc, errdesc=errdesc, live=True, custom_env=env)
        return
    completed = subprocess.run(cmd, shell=True, check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"{errdesc} (exit {completed.returncode})")


# ================================================================================
# Ollama の install.sh に必要な zstd を確保する
# ================================================================================
def _ensure_zstd() -> bool:
    if shutil.which("zstd"):
        _log("zstd found.")
        return True

    _log("zstd not found — required by Ollama install.sh. Trying to install...")
    attempts = [
        ("sudo -n apt-get update -y && sudo -n apt-get install -y zstd", "apt-get (sudo)"),
        ("apt-get update -y && apt-get install -y zstd", "apt-get"),
        ("sudo -n dnf install -y zstd", "dnf (sudo)"),
        ("dnf install -y zstd", "dnf"),
        ("sudo -n yum install -y zstd", "yum (sudo)"),
        ("sudo -n pacman -Sy --noconfirm zstd", "pacman (sudo)"),
    ]
    for cmd, label in attempts:
        # Skip package managers that aren't present
        pm = cmd.split()[0] if not cmd.startswith("sudo") else cmd.split()[2]
        if pm in ("apt-get", "dnf", "yum", "pacman") and not shutil.which(pm):
            continue
        if cmd.startswith("sudo") and not shutil.which("sudo"):
            continue
        try:
            _log(f"Installing zstd via {label}...")
            _run_cmd(cmd, desc=f"Install zstd ({label})", errdesc=f"zstd install failed ({label})")
        except Exception as e:
            _log(f"{label} failed: {e}")
            continue
        if shutil.which("zstd"):
            _log("zstd installed successfully.")
            return True

    _log(
        "Could not install zstd automatically. Install manually then restart WebUI:\n"
        "  Debian/Ubuntu: sudo apt-get install -y zstd\n"
        "  RHEL/Fedora:   sudo dnf install -y zstd\n"
        "  Arch:          sudo pacman -S zstd"
    )
    return False


# ================================================================================
# 公式 install.sh を取得して Ollama をインストールする（Linux）
# ================================================================================
def _run_install_sh() -> bool:
    if not _ensure_zstd():
        return False

    _log(f"Installing Ollama via {INSTALL_SH_URL} ...")
    script_path = EXT_ROOT / ".ollama_install.sh"
    try:
        with urllib.request.urlopen(INSTALL_SH_URL, timeout=120) as resp:
            script_path.write_bytes(resp.read())
    except Exception as e:
        _log(f"Failed to download install.sh: {e}")
        # Fallback: curl | sh
        curl = shutil.which("curl")
        if not curl:
            return False
        try:
            cmd = f'"{curl}" -fsSL {INSTALL_SH_URL} | sh'
            _run_cmd(cmd, desc="Install Ollama", errdesc="Ollama install failed")
            return _find_ollama_bin() is not None
        except Exception as e2:
            _log(f"curl|sh install failed: {e2}")
            return False

    try:
        script_path.chmod(script_path.stat().st_mode | 0o755)
    except OSError:
        pass

    env = os.environ.copy()
    # Allow non-interactive installs where supported
    env.setdefault("OLLAMA_INSTALL", "1")

    try:
        _run_cmd(
            f'sh "{script_path}"',
            desc="Install Ollama",
            errdesc="Ollama install failed",
            env=env,
        )
    except Exception as e:
        _log(f"install.sh failed: {e}")
        return False
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except TypeError:
            try:
                if script_path.is_file():
                    script_path.unlink()
            except OSError:
                pass
        except OSError:
            pass

    bin_path = _find_ollama_bin()
    if bin_path:
        _log(f"Ollama installed: {bin_path}")
        return True
    _log(
        "install.sh finished but `ollama` was not found on PATH. "
        "You may need root/sudo, or install manually: https://ollama.com"
    )
    return False


# ================================================================================
# ollama serve をバックグラウンド起動し API 応答を待つ
# ================================================================================
def _start_serve(bin_path: str, wait_seconds: float = 10.0) -> bool:
    if _api_up():
        return True

    _log(f"Starting `ollama serve` ({bin_path}) ...")
    log_path = EXT_ROOT / "ollama_serve.log"
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
            cwd=str(EXT_ROOT),
            start_new_session=(os.name != "nt"),
            creationflags=creationflags,
        )
    except OSError as e:
        _log(f"Failed to start ollama serve: {e}")
        return False

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        time.sleep(0.5)
        if _api_up():
            _log(f"Ollama API is up at {DEFAULT_API} (log: {log_path})")
            return True

    _log(
        f"Started ollama serve but API still unreachable after {wait_seconds:.0f}s. "
        f"Check {log_path}"
    )
    return False


# ================================================================================
# インストール／起動のメイン処理
# ================================================================================
def main() -> None:
    if _skip_install():
        _log("skip_install: leaving Ollama alone.")
        return

    # Optional WD Tagger dependency (does not block WebUI if install fails).
    try:
        if launch is not None and hasattr(launch, "is_installed") and hasattr(launch, "run_pip"):
            if not launch.is_installed("onnxruntime"):
                _log("Installing optional dependency: onnxruntime (WD Tagger)")
                launch.run_pip("install onnxruntime", "onnxruntime for WD Tagger")
    except Exception as e:
        _log(f"onnxruntime install skipped: {e}")

    bin_path = _find_ollama_bin()
    platform = sys.platform

    if not bin_path:
        if platform.startswith("linux"):
            ok = _run_install_sh()
            bin_path = _find_ollama_bin()
            if not ok or not bin_path:
                _log(
                    "Ollama auto-install did not succeed. "
                    "Try: sudo apt-get install -y zstd && curl -fsSL https://ollama.com/install.sh | sh"
                )
                return
        elif platform == "darwin":
            _log(
                "Ollama not found. On macOS install from https://ollama.com "
                "(auto-install via install.py is Linux-only)."
            )
            return
        else:
            # Windows and others
            _log(
                "Ollama not found. On Windows install from https://ollama.com "
                "(auto-install via install.py is Linux-only)."
            )
            return
    else:
        _log(f"Found ollama: {bin_path}")

    if _api_up():
        _log(f"Ollama API OK: {DEFAULT_API}")
        return

    _start_serve(bin_path)


try:
    main()
except Exception as e:
    # Never break WebUI launch
    print(f"{PREFIX} Unexpected error (ignored): {e}", flush=True)
