# sd-webui-llm-prompt-ollama

Stable Diffusion WebUI **Forge Neo** 向け拡張です。  
ローカルの [Ollama](https://ollama.com) 上の LLM を使い、テキストのアイデアから画像生成用プロンプトを作り、txt2img / img2img へ送れます。

Forge Neo 本体のコードは変更しません。このフォルダだけで完結します。

---

## できること

- 専用タブ **LLM Prompt (Ollama)**
- 日本語などのアイデア → 英語の SD プロンプト（または Danbooru タグ）
- `models.json` で GGUF カタログを管理し、タブから選択・ダウンロード・Ollama モデル作成
- `presets.json` で Instruction プリセットを編集可能
- **Send to txt2img / img2img**
- Settings で API URL・保存先・バイナリパスなどを設定

---

## 必要環境

- Stable Diffusion WebUI Forge Neo
- [Ollama](https://ollama.com)
- GGUF ファイル（リポジトリには含めません）

既定カタログ（`models.json`）:

| 項目 | 値 |
|---|---|
| モデル | Qwen3.5-9B Q4_K_M（約 6.2GB） |
| Hugging Face | [bartowski/Qwen_Qwen3.5-9B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF) |
| ファイル | `Qwen_Qwen3.5-9B-Q4_K_M.gguf` |
| 保存先（既定） | `models/llm/` |

追加の GGUF は `models.json` に追記できます。

拡張自体の必須 pip 依存はありません（標準ライブラリ中心）。  
**Download GGUF** は環境に `huggingface_hub` があればそれを使い、無ければ HTTP ダウンロードにフォールバックします。

---

## インストール

1. このフォルダを次に置く（または git clone）

```text
extensions/sd-webui-llm-prompt-ollama/
```

```bash
cd extensions
git clone https://github.com/<your-user>/sd-webui-llm-prompt-ollama.git
```

2. Ollama を用意する
   - **Windows / macOS:** [ollama.com](https://ollama.com) からインストール（トレイ常駐で API `http://127.0.0.1:11434`）
   - **Linux:** WebUI 起動時の `install.py` が未導入なら公式 `install.sh` を実行し、必要なら `ollama serve` を起動します（`--skip_install` 時は何もしません）
3. WebUI を起動し、タブ **LLM Prompt (Ollama)** で **Check connection**

### install.py について（Linux）

| 項目 | 内容 |
|---|---|
| 自動インストール | `ollama` 未検出時のみ公式インストーラを実行 |
| 自動起動 | API 未応答なら `ollama serve` をバックグラウンド起動 |
| 権限 | 環境によっては root/sudo が必要 |
| 依存 | 公式インストーラは **zstd** 必須（不足時は apt/dnf 等で入れようとします） |

手動例:

```bash
sudo apt-get install -y zstd
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

---

## 使い方

1. タブ **LLM Prompt (Ollama)** を開く
2. **Start Ollama**（未起動の場合）→ **Check connection**
3. **Catalog model** で使う GGUF を選ぶ（一覧は `models.json`）
   - `models.json` を編集したら **更新** で再読み込み
4. 初回のみ
   - **Download GGUF**（既定の保存先は `models/llm/`）
   - **Create / Update model**（`ollama_name` が Ollama 上のモデル名になります）
5. Idea を入力し、プリセットと言語を選ぶ
6. **Generate prompt**
7. **Send to txt2img / img2img**（または Copy）

手編集した Instruction を残したいときはプリセット **Custom** を選んでください。

### Local GGUF path / GGUF status

| タイミング | 動作 |
|---|---|
| 起動直後 | 空欄 |
| Catalog 選択変更 / **更新** | 想定パスと状態を自動入力 |
| **Download GGUF** 成功後 | 保存先パスを自動入力 |
| **↻ Status** | 状態を更新。パスは空のときだけ自動入力（手編集は維持） |

`models.json` が無い・壊れている・有効エントリが 0 件のときは、**Download GGUF** と **Create / Update model** は無効になります。

---

## models.json（GGUF カタログ）

拡張直下の `models.json` で、ダウンロード可能な GGUF と Ollama モデル名を定義します。

| フィールド | 意味 |
|---|---|
| `default` | 起動時に選ばれるエントリの `id` |
| `models[].id` | 内部キー（UI の選択値） |
| `models[].label` | ドロップダウン表示名 |
| `models[].ollama_name` | Create / Generate で使う Ollama モデル名 |
| `models[].hf_repo` | Hugging Face リポジトリ ID |
| `models[].hf_file` | リポジトリ内の GGUF ファイル名 |

例:

```json
{
  "default": "Qwen_Qwen3.5-9B-GGUF",
  "models": [
    {
      "id": "Qwen_Qwen3.5-9B-GGUF",
      "label": "Qwen3.5-9B-Q4_K_M.gguf (~6.2GB)",
      "ollama_name": "Qwen3.5-9B-Q4_K_M",
      "hf_repo": "bartowski/Qwen_Qwen3.5-9B-GGUF",
      "hf_file": "Qwen_Qwen3.5-9B-Q4_K_M.gguf"
    }
  ]
}
```

---

## presets.json（Instruction プリセット）

拡張直下の `presets.json` で、生成時のシステム指示を定義します。  
編集後は WebUI の再起動で反映されます。

| フィールド | 意味 |
|---|---|
| `default_lang` | 既定の Instruction language |
| `default_preset` | 既定のプリセット `id` |
| `languages` | 言語タブの一覧 |
| `presets[].id` | ドロップダウン表示名 |
| `presets[].nsfw` | NSFW 向けフラグ（将来用・UI フィルタ等） |
| `presets[].instructions` | 言語キー → 指示文 |

同梱プリセット例: Idea → SD prompt / Danbooru tags / Short punchy / Detailed cinematic / Custom

---

## Settings

**Settings → LLM Prompt (Ollama)**

| 設定 | 内容 |
|---|---|
| Ollama API URL | 既定 `http://127.0.0.1:11434` |
| Default Ollama model name | 既定は `models.json` の default の `ollama_name` |
| Directory for GGUF downloads | 既定 `models/llm` |
| Default path to GGUF file | 空なら `models/llm/<hf_file>`（カタログ default） |
| Path to ollama binary | `/api/create` 失敗時の CLI 用（空なら PATH） |

---

## モデル作成の仕組み

1. 拡張が `.Modelfile.generated` を書き出す（`FROM` に GGUF の絶対パス）
2. まず Ollama **`POST /api/create`** を試す
3. 失敗したら **`ollama create <name> -f .Modelfile.generated`**
4. それでも失敗したら GGUF の blob アップロード経由を試す（大きいファイルでは時間がかかります）

テンプレートは拡張直下の `Modelfile` です。

**Connection status** に出る `Models: ...` は、`models.json` ではなく **Ollama に既に登録されているモデル一覧** です。不要なモデルは `ollama rm <name>` で削除できます。

---

## ディレクトリ構成

```text
extensions/sd-webui-llm-prompt-ollama/
├── README.md
├── install.py
├── models.json
├── presets.json
├── Modelfile
├── .gitignore
├── style.css
├── javascript/
│   └── llm_prompt_ollama_copy.js
├── scripts/
│   └── llm_prompt_ollama_tab.py
└── llm_prompt_ollama/
    ├── __init__.py
    ├── ollama_client.py
    ├── model_setup.py
    ├── models_catalog.py
    ├── download.py
    └── presets.py
```

---

## トラブルシュート

| 症状 | 対処 |
|---|---|
| Connection refused | Ollama 未起動。**Start Ollama** または `ollama serve`。Windows はトレイ常駐を確認 |
| Linux で install.py が失敗 | `zstd` 不足が多い。入れたあと WebUI 再起動、または手動 install.sh |
| ollama CLI not found | Ollama を入れ直すか Settings でバイナリパスを指定 |
| Download / Create が押せない | `models.json` が無い・不正・有効モデル 0 件。ファイルを直して **更新** |
| Create failed / architecture error | 当該 GGUF がお使いの Ollama 版で未対応の可能性。Ollama を最新化 |
| Generate が遅い・VRAM 不足 | SD と同時利用だと重い。SD モデルの Unload や Ollama 側 GPU 設定を調整 |

---

## 注意

- Forge 本体・`requirements.txt`・venv のピンは変更しません
- GGUF は git 管理対象外です（`*.gguf`）
- `install.py` は Linux で外部の公式インストーラを実行することがあります
- 追加の Instruction や NSFW 向け文言が必要な場合は、`presets.json` に自分でエントリを追加してください
