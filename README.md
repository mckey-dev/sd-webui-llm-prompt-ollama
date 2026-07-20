# sd-webui-llm-prompt-ollama

**更新日:** 2026-07-20

[Stable Diffusion WebUI Forge Neo](https://github.com/lllyasviel/stable-diffusion-webui-forge) 向け拡張機能です。ローカル（または LAN 上）の [Ollama](https://ollama.com) で LLM / VLM を実行し、テキストや画像から Stable Diffusion 用プロンプトを生成して txt2img / img2img に送ります。Forge 本体のコードは変更しません。

---

## 概要

| タブ | 役割 |
|------|------|
| **モデルロード** | Ollama 接続、Hugging Face から GGUF 取得、Ollama モデルの Create / Update、VRAM 解放 |
| **Idea** | 日本語・英語などのアイデア文 → SD プロンプト（自然文または Danbooru タグ系プリセット） |
| **VLM** | 画像 → プロンプト（vision 対応 Ollama モデル） |
| **WD Tagger**（VLM タブ内） | SmilingWolf V3 ONNX でタグ抽出（Ollama 不要） |

カタログと指示文は拡張直下の `models.json` / `presets.json` で編集できます。WebUI の **Settings → LLM Prompt (Ollama)** で API URL や保存先などを変更できます。

---

## 要件

- **Forge Neo**（本拡張を `extensions/` に配置した WebUI）
- **Ollama**（HTTP API。既定 `http://127.0.0.1:11434`）
- **GGUF**（リポジトリには含めません。`models.json` で Hugging Face からダウンロード）
- **VLM** 利用時: mmproj 付き vision モデル、または Ollama 公式の multimodal モデル
- **WD Tagger**（任意）: Python パッケージ `onnxruntime`, `huggingface_hub`（WebUI の venv 内）

GGUF の既定保存先は **`models/llm/`**（Settings で変更可能）。

---

## インストール

1. 本リポジトリを `extensions/sd-webui-llm-prompt-ollama/` に clone またはコピーする。
2. [Ollama](https://ollama.com) をインストールし、API が応答する状態にする。
3. WebUI を起動し、タブ **LLM Prompt (Ollama)** を開く。
4. **モデルロード** → **Check connection** で接続を確認する。

### Linux での自動セットアップ

WebUI 起動時、`install.py` が次を試みます（Forge Neo の `--skip_install` 指定時はスキップ）。

- `ollama` コマンドが無い → 公式 `install.sh` の実行
- API が応答しない → `ollama serve` のバックグラウンド起動

タブから **Start Ollama** / **Restart Ollama** でも同様の操作ができます。

---

## クイックスタート

1. **Catalog model** で `[text]`（Idea）または `[vision]`（VLM）を選ぶ。
2. **Download GGUF** → **Create / Update model**（初回のみ。2 回目以降は GGUF 更新時）。
3. **Ollama model name** が Generate で使う名前（Create 後に一覧へ反映）。
4. **Idea** または **VLM** でプリセットを選び **Generate** → **Generated prompt** 右上のコピーアイコン、または **Send to txt2img / img2img**。

Ollama 公式モデルだけ使う場合は Download / Create を省略し、Settings と **Ollama model name** を `qwen3.5:9b` などに合わせて **Check connection** 後に Generate できます。

---

## モデルロード

### Connection status の見方

接続成功時、**Installed**（ディスク上）と **Loaded in memory**（VRAM/RAM 上）を分けて表示します。

```
OK — http://127.0.0.1:11434

Loaded in memory:
Qwen3.5-9B-Q4_K_M:latest (VRAM 5.0 GiB, expires ~5m)

Catalog models (models.json):
Qwen3.5-9B-Q4_K_M:latest
Qwen3.5-9B-Q4_K_M-vision:latest
Other Ollama models:
llama3.2:latest
```

- **Loaded in memory: (none)** — 現在メモリにモデルは載っていない（SD 生成向けに VRAM を空けた状態）。
- **Catalog models** — `models.json` に定義された名前と一致するインストール済みモデル（1 行 1 件）。
- **Other Ollama models** — 別途 `ollama pull` したモデルなど。

`:latest` 付き表示と無しは同一モデルとして扱います。

### Ollama 接続ボタン

| ボタン | 動作 |
|--------|------|
| **Check connection** | API 確認、`Connection status` と **Ollama model name** ドロップダウンを更新 |
| **Start Ollama** | API 未応答時のみ `ollama serve` を起動（**ローカル** Ollama 向け） |
| **Restart Ollama** | Linux: `systemctl` → `pkill` → `serve` / Windows: `taskkill` → `serve` |
| **Unload all from memory** | ロード中の全モデルを API でアンロード（`keep_alive: 0`） |
| **Unload selected model** | ドロップダウンで選んだ 1 モデルのみアンロード |

メモリだけ空けたい場合は **Restart** より **Unload** を優先してください。リモートの Ollama API URL をタブに指定している場合、**Start / Restart** はローカルプロセス用のため使わず、**Unload** は API 経由で利用できます。

Generate 後、モデルは既定でしばらくメモリに残ります（Ollama の `keep_alive` 既定約 5 分）。SD 生成前に **Unload all** すると VRAM 競合を避けやすくなります。

### GGUF と Create

| UI | 説明 |
|----|------|
| **Catalog model** | `models.json` の一覧（横の **更新** で再読み込み） |
| **Download GGUF** | HF から GGUF（vision は mmproj も）を `models/llm/` へ |
| **Create / Update model** | ローカル GGUF から Ollama モデルを作成・更新 |
| **↻ Status** | 選択カタログの GGUF 有無を再表示 |

---

## Idea / VLM

- **Instruction preset** / **Instruction language** でシステム指示を選ぶ（**Custom** で自由記述）。
- **Generate** 後、**Generated prompt** テキストボックス右上のコピーアイコン（Gradio 標準）でクリップボードへコピーできます。
- 詳細パラメータ（temperature 等）は各タブのアコーディオン内。
- `presets.json` を編集したら **Instruction preset** 横の **更新** を押す。
- `uncensored: true` のプリセットは **Settings → Show uncensored instruction presets** を ON にしないと一覧に出ません。

---

## WD Tagger

VLM タブ内の **Tag with WD Tagger** で、SmilingWolf V3 ONNX モデルを実行します（Ollama 不要）。

- General / Character 閾値、`_` → スペース、括弧エスケープは UI のチェックボックスで調整。
- モデルキャッシュ: `models/llm/wd-tagger/`

---

## models.json

| フィールド | 意味 |
|------------|------|
| `default` | 初回選択するカタログの `id` |
| `id` | 内部 ID |
| `label` | ドロップダウン表示 |
| `ollama_name` | Ollama 上のモデル名（Create / Generate） |
| `hf_repo` / `hf_file` | Hugging Face の GGUF |
| `hf_mmproj` | （任意）vision 用 mmproj |
| `modality` | （任意）`text` / `vision` |

同梱例:

- `Qwen3.5-9B-Q4_K_M` — Idea（text）
- `Qwen3.5-9B-Q4_K_M-vision` — VLM（同一 GGUF + mmproj）

---

## presets.json

| フィールド | 意味 |
|------------|------|
| `languages` | 言語ラジオの選択肢 |
| `presets[].id` | プリセット名 |
| `presets[].for` | `idea` / `vlm`（未指定は idea のみ） |
| `presets[].uncensored` | `true` のとき Settings で表示 ON が必要 |
| `presets[].instructions` | 言語コード → システム指示文 |

JSON の構文エラーがあると、ファイル全体の読み込みに失敗することがあります。

---

## Settings（LLM Prompt (Ollama)）

| 項目 | 説明 |
|------|------|
| Ollama API URL | 既定 `http://127.0.0.1:11434` |
| Default Ollama model name | Create 後などの既定名（通常は `models.json` の default に追随） |
| Directory for GGUF downloads | 既定 `models/llm` |
| Path to ollama binary | `/api/create` 失敗時の CLI フォールバック（空なら PATH 探索） |
| Show uncensored instruction presets | 既定 OFF |

旧設定キー（`llmuse_ollama_*` 等）は、新キーが未設定のときのみ参照されます。

---

## Create / Update の内部処理

1. 拡張が `Modelfile` テンプレートから `.Modelfile.generated` を生成する。
2. GGUF を blob アップロードし、`/api/create`（`files` / `system` / `parameters`）を呼ぶ。
3. 失敗時は `ollama create -f`（CLI）→ 必要に応じて従来 API にフォールバック。

### Qwen3.5 GGUF で validate / `llama-quantize` が失敗する場合

Ollama のバージョンによっては、Hugging Face の Qwen3.5 GGUF import が拒否されます。テキスト用途のみなら次で回避できます。

```bash
ollama pull qwen3.5:9b
```

Settings と **Ollama model name** を `qwen3.5:9b` にし、Download / Create を使わず **Idea** から Generate します。Vision は公式 multimodal タグを `pull` するか、GGUF + mmproj が通る Ollama 版を待つ必要があります。

---

## リポジトリ構成

```text
extensions/sd-webui-llm-prompt-ollama/
├── install.py              … Linux 向け Ollama 導入・serve 起動
├── models.json             … GGUF カタログ
├── presets.json            … Idea / VLM 指示プリセット
├── Modelfile               … Create 用テンプレート
├── scripts/
│   └── llm_prompt_ollama_tab.py
├── style.css
└── llm_prompt_ollama/      … Ollama クライアント、DL、Create、プリセット、WD Tagger
```

---

## トラブルシュート

| 症状 | 対処 |
|------|------|
| Connection refused | **Start Ollama** または手動で `ollama serve` |
| **Loaded in memory** が空かない | **Unload all** / **Unload selected**、または時間経過で自動アンロード |
| Create / Download が無効 | `models.json` の JSON・必須フィールド・**更新** を確認 |
| Create API 400 / files 未指定 | Ollama を最新に更新し、拡張を最新化 |
| validate GGUF / llama-quantize | 上記 `ollama pull` 回避、または Ollama 更新 |
| プリセットが増えない | `presets.json` の JSON、**更新**、Uncensored は Settings ON |
| **Other Ollama models** に不要な名前 | `ollama rm <name>` で削除 |
| WD Tagger エラー | venv 内で `pip install onnxruntime huggingface_hub` |
| Restart 後も API が古い | Windows はトレイの Ollama を終了してから **Start** |

---

## 注意事項

- `*.gguf` など大容量ファイルは git 管理対象外です。
- Linux の `install.py` は外部の公式 Ollama インストーラを実行することがあります。
- 本拡張は Ollama の利用を支援するものであり、Ollama 本体のライセンス・各 GGUF のライセンスは配布元に従ってください。
