# ================================================================================
# Ollama HTTP API クライアント
# ================================================================================
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
import urllib.error
import urllib.request
from typing import Any


# ================================================================================
# Ollama HTTP 呼び出し失敗時の例外
# ================================================================================
class OllamaError(RuntimeError):
    pass


_THINK_BLOCK_RE = re.compile(
    r"<think>.*?</think>|<thinking>.*?</thinking>",
    re.IGNORECASE | re.DOTALL,
)


# ================================================================================
# chat / generate レスポンスからアシスタント文を取り出す
# ================================================================================
def _extract_assistant_text(data: dict[str, Any]) -> str:
    if data.get("error"):
        raise OllamaError(str(data["error"]))

    # /api/chat
    msg = data.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        thinking = msg.get("thinking") or msg.get("reasoning")
        parts: list[str] = []
        if content is not None and str(content).strip():
            parts.append(str(content))
        if thinking is not None and str(thinking).strip():
            # Prefer final content; keep thinking only if content empty
            if not parts:
                parts.append(str(thinking))
        text = "\n".join(parts).strip()
        if text:
            text = _THINK_BLOCK_RE.sub("", text).strip()
            return text

    # /api/generate
    for key in ("response", "output", "text"):
        val = data.get(key)
        if val is not None and str(val).strip():
            return _THINK_BLOCK_RE.sub("", str(val)).strip()

    return ""


# ================================================================================
# Ollama の HTTP ボディを JSON / NDJSON として解釈する
# ================================================================================
def _parse_response_body(raw: str) -> Any:
    raw = (raw or "").strip()
    if not raw:
        return None

    # Prefer a single JSON document (pretty-printed OK).
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # NDJSON stream: merge chat/generate chunks
    merged: dict[str, Any] | None = None
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    response_parts: list[str] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        merged = obj if merged is None else {**merged, **obj}
        msg = obj.get("message")
        if isinstance(msg, dict):
            c = msg.get("content")
            if c:
                content_parts.append(str(c))
            t = msg.get("thinking") or msg.get("reasoning")
            if t:
                thinking_parts.append(str(t))
        r = obj.get("response")
        if r:
            response_parts.append(str(r))

    if merged is None:
        return raw

    if content_parts or thinking_parts:
        msg = dict(merged.get("message") or {})
        msg["content"] = "".join(content_parts)
        if thinking_parts and not msg.get("content"):
            msg["thinking"] = "".join(thinking_parts)
        merged["message"] = msg
    if response_parts:
        merged["response"] = "".join(response_parts)
    return merged


# ================================================================================
# Ollama モデル名から :latest 等のタグを除いたベース名を返す
# ================================================================================
def normalize_ollama_model_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return ""
    if ":" in n:
        base, tag = n.rsplit(":", 1)
        if tag == "latest":
            return base
    return n


# ================================================================================
# 2 つの Ollama モデル名が同一モデルを指すか（タグ無視）を返す
# ================================================================================
def model_names_equivalent(a: str, b: str) -> bool:
    na = normalize_ollama_model_name(a)
    nb = normalize_ollama_model_name(b)
    return bool(na) and na == nb


# ================================================================================
# 希望名に一致するインストール済みモデル名を返す（見つからなければ None）
# ================================================================================
def pick_ollama_model_name(preferred: str, installed: list[str]) -> str | None:
    pref = (preferred or "").strip()
    if not pref:
        return None
    if pref in installed:
        return pref
    for m in installed:
        if model_names_equivalent(m, pref):
            return m
    return None


# ================================================================================
# バイト数を GiB 表示用の短い文字列にする
# ================================================================================
def _format_vram_gib(size_bytes: Any) -> str | None:
    try:
        n = int(size_bytes)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return f"{n / (1024 ** 3):.1f} GiB"


# ================================================================================
# expires_at を短い相対表記にする（失敗時は None）
# ================================================================================
def _format_expires_hint(expires_at: Any) -> str | None:
    raw = (expires_at or "").strip() if isinstance(expires_at, str) else ""
    if not raw:
        return None
    try:
        normalized = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = (dt - now).total_seconds()
        if delta <= 0:
            return "expires soon"
        if delta < 120:
            return f"expires ~{int(delta)}s"
        if delta < 3600:
            return f"expires ~{int(delta / 60)}m"
        return f"expires ~{int(delta / 3600)}h"
    except (ValueError, TypeError):
        return None


# ================================================================================
# メモリ上にロードされているモデルの表示ブロック
# ================================================================================
def format_loaded_in_memory(running: list[dict[str, Any]] | None) -> str:
    items = running or []
    if not items:
        return "Loaded in memory: (none)"
    lines = ["Loaded in memory:"]
    for item in items:
        name = str(item.get("name") or item.get("model") or "?").strip()
        parts: list[str] = []
        vram = _format_vram_gib(item.get("size_vram"))
        if vram:
            parts.append(f"VRAM {vram}")
        exp = _format_expires_hint(item.get("expires_at"))
        if exp:
            parts.append(exp)
        if parts:
            lines.append(f"{name} ({', '.join(parts)})")
        else:
            lines.append(f"{name} (in memory)")
    return "\n".join(lines)


# ================================================================================
# インストール済みモデル（/api/tags）の表示ブロック
# ================================================================================
def format_installed_models_block(
    models: list[str],
    *,
    catalog_ollama_names: list[str] | None = None,
) -> str:
    catalog = [c.strip() for c in (catalog_ollama_names or []) if (c or "").strip()]
    if not models:
        if catalog:
            return "Catalog models (models.json): (none installed — Download / Create)"
        return "Models: (none)"

    catalog_hits: list[str] = []
    other: list[str] = []
    for m in models:
        if catalog and any(model_names_equivalent(m, c) for c in catalog):
            catalog_hits.append(m)
        elif catalog:
            other.append(m)

    lines: list[str] = []
    if catalog:
        if catalog_hits:
            lines.append("Catalog models (models.json):")
            lines.extend(catalog_hits)
        else:
            lines.append("Catalog models (models.json): (none installed — Download / Create)")
        if other:
            lines.append("Other Ollama models:")
            lines.extend(other)
    else:
        lines.append("Models:")
        lines.extend(models)
    return "\n".join(lines)


# ================================================================================
# Connection status 全文（OK + Loaded + Installed）
# ================================================================================
def format_full_connection_status(
    base_url: str,
    installed_models: list[str],
    running_models: list[dict[str, Any]] | None,
    *,
    catalog_ollama_names: list[str] | None = None,
    prefix: str | None = None,
) -> str:
    url = (base_url or "").rstrip("/")
    chunks: list[str] = []
    if prefix and prefix.strip():
        chunks.append(prefix.strip())
    chunks.append(f"OK — {url}")
    chunks.append(format_loaded_in_memory(running_models))
    chunks.append(
        format_installed_models_block(
            installed_models,
            catalog_ollama_names=catalog_ollama_names,
        )
    )
    return "\n\n".join(chunks)


# ================================================================================
# 接続 OK 時のステータス文言（後方互換；full status と同内容）
# ================================================================================
def format_connection_status(
    base_url: str,
    models: list[str],
    *,
    catalog_ollama_names: list[str] | None = None,
    running_models: list[dict[str, Any]] | None = None,
) -> str:
    return format_full_connection_status(
        base_url,
        models,
        running_models,
        catalog_ollama_names=catalog_ollama_names,
    )


# ================================================================================
# Ollama サーバーとの通信を行うクライアント
# ================================================================================
class OllamaClient:
    # ================================================================================
    # API ベース URL とタイムアウトを初期化する
    # ================================================================================
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 120.0):
        self.base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        self.timeout = float(timeout)

    # ================================================================================
    # Ollama API へ HTTP リクエストを送る
    # ================================================================================
    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise OllamaError(f"HTTP {e.code} {path}: {body or e.reason}") from e
        except urllib.error.URLError as e:
            reason = e.reason
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}: {reason}. "
                "Is Ollama installed and running? Try: ollama serve"
            ) from None
        except TimeoutError:
            raise OllamaError(f"Timeout talking to Ollama at {self.base_url}") from None
        except ConnectionError as e:
            raise OllamaError(
                f"Cannot reach Ollama at {self.base_url}: {e}. "
                "Is Ollama installed and running? Try: ollama serve"
            ) from None

        return _parse_response_body(raw)

    # ================================================================================
    # 接続確認とモデル一覧の短いステータス文言を返す
    # ================================================================================
    def health(self, *, catalog_ollama_names: list[str] | None = None, prefix: str | None = None) -> str:
        models = self.list_models()
        running = self.list_running_models()
        return format_full_connection_status(
            self.base_url,
            models,
            running,
            catalog_ollama_names=catalog_ollama_names,
            prefix=prefix,
        )

    # ================================================================================
    # メモリ上で実行中のモデル一覧（/api/ps）
    # ================================================================================
    def list_running_models(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/api/ps", timeout=15.0)
        if not isinstance(data, dict):
            return []
        out: list[dict[str, Any]] = []
        for item in data.get("models") or []:
            if isinstance(item, dict):
                out.append(item)
        return out

    # ================================================================================
    # 指定モデルをメモリからアンロードする（keep_alive: 0）
    # ================================================================================
    def unload_model(self, model: str) -> None:
        name = (model or "").strip()
        if not name:
            raise OllamaError("Model name is empty.")
        chat_payload: dict[str, Any] = {
            "model": name,
            "messages": [],
            "keep_alive": 0,
            "stream": False,
        }
        try:
            self._request("POST", "/api/chat", chat_payload, timeout=120.0)
            return
        except OllamaError:
            pass
        gen_payload: dict[str, Any] = {
            "model": name,
            "keep_alive": 0,
            "stream": False,
        }
        self._request("POST", "/api/generate", gen_payload, timeout=120.0)

    # ================================================================================
    # 登録済みモデル名の一覧を返す
    # ================================================================================
    def list_models(self) -> list[str]:
        data = self._request("GET", "/api/tags", timeout=15.0)
        if not isinstance(data, dict):
            return []
        out: list[str] = []
        for item in data.get("models") or []:
            name = item.get("name") or item.get("model")
            if name:
                out.append(str(name))
        return sorted(out)

    # ================================================================================
    # Modelfile 本文でモデルを作成／更新する
    # ================================================================================
    def create_from_modelfile(
        self,
        model_name: str,
        modelfile: str,
        *,
        stream: bool = False,
        timeout: float = 600.0,
    ) -> str:
        payload = {
            "model": model_name,
            "name": model_name,
            "modelfile": modelfile,
            "stream": stream,
        }
        data = self._request("POST", "/api/create", payload, timeout=timeout)
        if isinstance(data, dict):
            status = data.get("status") or data.get("error")
            if data.get("error"):
                raise OllamaError(str(data["error"]))
            return str(status or "success")
        return str(data or "success")

    # ================================================================================
    # チャット API（失敗時は generate）でプロンプトを生成する
    # ================================================================================
    def chat(
        self,
        model: str,
        user_content: str,
        *,
        system: str | None = None,
        images: list[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        num_predict: int | None = None,
        think: bool = False,
        timeout: float = 300.0,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system and system.strip():
            messages.append({"role": "system", "content": system.strip()})
        user_msg: dict[str, Any] = {"role": "user", "content": user_content}
        if images:
            user_msg["images"] = list(images)
        messages.append(user_msg)

        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = float(temperature)
        if top_p is not None:
            options["top_p"] = float(top_p)
        if num_predict is not None:
            options["num_predict"] = int(num_predict)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        # Qwen3.5 thinking models often leave content empty unless think is off.
        payload_with_think = {**payload, "think": bool(think)}

        try:
            data = self._request("POST", "/api/chat", payload_with_think, timeout=timeout)
        except OllamaError as e:
            # Older Ollama may reject unknown "think" field
            if "think" in str(e).lower() or "400" in str(e):
                data = self._request("POST", "/api/chat", payload, timeout=timeout)
            else:
                raise

        if not isinstance(data, dict):
            raise OllamaError(f"Unexpected chat response: {data!r}")

        text = _extract_assistant_text(data)
        if text:
            return text

        # Fallback: /api/generate (some GGUF builds chat-template oddly)
        gen_prompt = user_content
        if system and system.strip():
            gen_prompt = f"{system.strip()}\n\nUser:\n{user_content}\n\nAssistant:"
        gen_payload: dict[str, Any] = {
            "model": model,
            "prompt": gen_prompt,
            "stream": False,
        }
        if images:
            gen_payload["images"] = list(images)
        if options:
            gen_payload["options"] = options
        gen_with_think = {**gen_payload, "think": bool(think)}
        try:
            gen = self._request("POST", "/api/generate", gen_with_think, timeout=timeout)
        except OllamaError:
            gen = self._request("POST", "/api/generate", gen_payload, timeout=timeout)

        if isinstance(gen, dict):
            text = _extract_assistant_text(gen)
            if text:
                return text
            raise OllamaError(
                "Model returned empty text (chat + generate).\n"
                f"chat keys={list(data.keys())} message={data.get('message')!r}\n"
                f"generate keys={list(gen.keys())}"
            )
        raise OllamaError(f"Model returned empty text. Raw chat response: {data!r}")
