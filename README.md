# sd-webui-llm-prompt-ollama

更新日: 2026-07-20

Stable Diffusion WebUI **Forge Neo** 向け拡張です。  
ローカルの [Ollama](https://ollama.com) 上の LLM を使い、テキストのアイデアから画像生成用プロンプトを作り、txt2img / img2img へ送れます。

Forge Neo 本体のコードは変更しません。このフォルダだけで完結します。

---

## できること

- 専用タブ **LLM Prompt (Ollama)**（サブタブ: **モデルロード** / **Idea** / **VLM**）
- **Idea** — 日本語などのアイデア → 英語の SD プロンプト（または Danbooru タグ）
- **VLM** — 入力画像を解析して画像生成用プロンプトを生成（ローカル Ollama vision）
- おまけ **WD Tagger (V3)** — VLM なしで Danbooru 風タグ列を生成（ONNX・ローカル）
- `models.json` で GGUF カタログを管理（任意 `hf_mmproj` で vision）
- `presets.json` で Instruction プリセットを編集（`for`: `idea` / `vlm`、`uncensored` フラグ）
- **Send to txt2img / img2img**（Idea / VLM それぞれ独立）
- **Settings** で API URL・GGUF 保存先・Ollama バイナリパス・**Uncensored プリセットの表示**などを設定

---

## 必要環境

- Stable Diffusion WebUI Forge Neo
- [Ollama](https://ollama.com)
- GGUF ファイル（リポジトリには含めません）
- **VLM タブ**を使う場合は、Ollama 上の **Vision モデル**が必要です（`models.json` で `hf_mmproj` 付きエントリを Download / Create するか、vision 対応モデルを用意）

### 同梱カタログ（`models.json`）

| id | 用途 | Ollama 名 | GGUF / mmproj |
|---|---|---|---|
| `Qwen_Qwen3.5-9B-GGUF` | テキスト（Idea） | `Qwen3.5-9B-Q4_K_M` | [bartowski/Qwen_Qwen3.5-9B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF) の `Qwen_Qwen3.5-9B-Q4_K_M.gguf`（約 6.2GB） |
| `Qwen_Qwen3.5-9B-GGUF-vision` | Vision（VLM） | `Qwen3.5-9B-Q4_K_M-vision` | 同上 GGUF + `mmproj-Qwen_Qwen3.5-9B-bf16.gguf`（約 +0.9GB） |

GGUF の保存先（既定）は **`models/llm/`**（Settings で変更可）。

追加の GGUF は `models.json` に追記できます。vision モデルは同じ repo の `hf_mmproj` を指定すると Download / Create で二重 `FROM` の Modelfile を生成します。

拡張自体の必須 pip 依存はありません（標準ライブラリ中心）。  
**Download GGUF** は環境に `huggingface_hub` があればそれを使い、無ければ HTTP ダウンロードにフォールバックします。  
**WD Tagger** は任意で `onnxruntime`（`install.py` が未導入時に試行）と `huggingface_hub` を使います。

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

### モデルロード

1. タブ **LLM Prompt (Ollama)** → **モデルロード**
2. **Start Ollama**（未起動の場合）→ **Check connection**
3. **Catalog model** で使う GGUF を選ぶ（Idea 用は `[text]`、VLM 用は `[vision]` 付きエントリ）
4. 初回のみ **Download GGUF** → **Create / Update model**  
   - Idea と VLM で別カタログを使う場合は、それぞれ Download / Create が必要です

タブの **Ollama API URL** は起動時に Settings の値で初期化されます。恒久変更は **Settings → LLM Prompt (Ollama)** で行い、WebUI 再起動またはタブ再読み込み後に反映されます。

### Idea（テキスト）

1. **Idea** タブでアイデアを入力
2. プリセット／言語を選び **Generate prompt**
3. 右カラムから **Send to txt2img / img2img**（または Copy）
4. Temperature 等は **詳細設定** アコーディオン内

### VLM（画像解析）

1. **VLM** タブで画像をアップロード
2. （任意）追加指示を入力 → **Generate with VLM**
3. プリセット等は **VLM オプション** アコーディオン内
4. 画像からタグだけ欲しい場合は **WD Tagger**（Ollama 不要）— 詳細は [WD Tagger (V3)](#wd-tagger-v3)

### WD Tagger (V3)

VLM タブの **WD Tagger** アコーディオンから、入力画像を [SmilingWolf](https://huggingface.co/SmilingWolf) の **V3 ONNX** モデルで解析し、Danbooru 風のカンマ区切りタグ列を **Generated prompt** に出力します（**Tag with WD Tagger**）。

| UI | 既定 | 説明 |
|---|---|---|
| WD Tagger model | `wd-swinv2-tagger-v3` | V3 のみ選択可（ConvNeXt / ViT / ViT-large / EVA02-large 等） |
| General threshold | 0.35 | **general** タグ（CSV category `0`）の採用下限 |
| Character threshold | 0.85 | **キャラクタ**タグ（category `4`）の採用下限 |
| アンダースコアをスペースに置換する | ON | 出力タグの `_` → スペース |
| 括弧にエスケープ処理をする | ON | `(` / `)` の直前に `\`（例: `hatsune_miku` → `hatsune miku`、`foo(bar)` → `foo\(bar\)`） |

**タグの並び:** 閾値を超えた **キャラクタ名タグを先頭** に、続けて general タグ（いずれも信頼度の高い順）。**rating**（category `9`）は出力しません。

キャラ名が出ない場合は **Character threshold** を下げてください（例: 0.75）。逆に上げると誤検出は減りますが、載りにくくなります。

初回実行時、選んだモデルは Hugging Face から `models/llm/wd-tagger/<repo>/` に ONNX と `selected_tags.csv` をダウンロードします（`huggingface_hub` と `onnxruntime` が必要）。

### Instruction プリセット

| 操作 | 反映方法 |
|---|---|
| `presets.json` を編集した | Idea / VLM の **Instruction preset** 横 **更新**（または WebUI 再起動） |
| Settings で Uncensored 表示を ON/OFF | **Apply settings** のあと、各タブ **更新**（または再起動） |

- 手編集した Instruction を残したいときはプリセット **Custom** を選んでください。
- 名称が `(Uncensored)` のプリセットは **`uncensored: true`** です。既定では Settings で非表示です（下記 Settings）。

### Local GGUF path / GGUF status

| タイミング | 動作 |
|---|---|
| 起動直後 | 空欄 |
| Catalog 選択変更 / **更新** | 想定パスと状態を自動入力（mmproj 含む） |
| **Download GGUF** 成功後 | 保存先パスを自動入力 |
| **↻ Status** | 状態を更新。パスは空のときだけ自動入力（手編集は維持） |

`models.json` が無い・壊れている・有効エントリが 0 件のときは、**Download GGUF** と **Create / Update model** は無効になります。

---

## models.json（GGUF カタログ）

拡張直下の `models.json` で、ダウンロード可能な GGUF と Ollama モデル名を定義します。  
カタログ変更後は **モデルロード** タブの **更新** で Dropdown を再読み込みします。

| フィールド | 意味 |
|---|---|
| `default` | 起動時に選ばれるエントリの `id` |
| `models[].id` | 内部キー（UI の選択値） |
| `models[].label` | ドロップダウン表示名 |
| `models[].ollama_name` | Create / Generate で使う Ollama モデル名 |
| `models[].hf_repo` | Hugging Face リポジトリ ID |
| `models[].hf_file` | リポジトリ内の GGUF ファイル名 |
| `models[].hf_mmproj` | （任意）同一 repo の mmproj GGUF。Create 時に二重 `FROM` |
| `models[].modality` | （任意）`text` / `vision`。未指定時は mmproj 有無で判定 |

例（同梱に近い形）:

```json
{
  "default": "Qwen_Qwen3.5-9B-GGUF",
  "models": [
    {
      "id": "Qwen_Qwen3.5-9B-GGUF",
      "label": "Qwen3.5-9B-Q4_K_M.gguf (~6.2GB) [text]",
      "ollama_name": "Qwen3.5-9B-Q4_K_M",
      "hf_repo": "bartowski/Qwen_Qwen3.5-9B-GGUF",
      "hf_file": "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
      "modality": "text"
    },
    {
      "id": "Qwen_Qwen3.5-9B-GGUF-vision",
      "label": "Qwen3.5-9B-Q4_K_M + mmproj-bf16 (~6.2GB + ~0.9GB) [vision]",
      "ollama_name": "Qwen3.5-9B-Q4_K_M-vision",
      "hf_repo": "bartowski/Qwen_Qwen3.5-9B-GGUF",
      "hf_file": "Qwen_Qwen3.5-9B-Q4_K_M.gguf",
      "hf_mmproj": "mmproj-Qwen_Qwen3.5-9B-bf16.gguf",
      "modality": "vision"
    }
  ]
}
```

---

## presets.json（Instruction プリセット）

拡張直下の `presets.json` で、生成時の **システム指示**（Instruction）を定義します。  
JSON の構文エラーがあると読み込みに失敗し、最小限の組み込みプリセットだけが使われます（追加した項目は出ません）。

| フィールド | 意味 |
|---|---|
| `default_lang` | 既定の Instruction language |
| `default_preset` | 既定のプリセット `id` |
| `languages` | 言語タブの選択肢 |
| `presets[].id` | ドロップダウン表示名（一意） |
| `presets[].for` | （任意）`["idea"]` / `["vlm"]` / 両方。未指定は `idea` のみ |
| `presets[].uncensored` | Uncensored 向けプリセットか。**Settings の表示 ON 時のみ** Idea/VLM 一覧に出る。旧キー `nsfw` も読み込み可 |
| `presets[].instructions` | 言語キー → 指示文 |

### 同梱プリセット一覧

| id | for | uncensored | 概要 |
|---|---|:---:|---|
| Idea → SD prompt | idea | | 自然言語プロンプト（英語） |
| Idea → SD prompt (Uncensored) | idea | ✓ | 上記 + 検閲・拒否なし指示 |
| Idea → Danbooru tags | idea | | Danbooru 風タグ列 |
| Idea → Danbooru tags (Uncensored) | idea | ✓ | 上記 Uncensored 版 |
| Short punchy prompt | idea | | 短い自然言語 |
| Detailed cinematic prompt | idea | | 詳細・シネマティック |
| Image → SD prompt | vlm | | 画像から自然言語プロンプト |
| Image → SD prompt (Uncensored) | vlm | ✓ | 上記 Uncensored 版 |
| Image → Detailed cinematic | vlm | | 画像からシネマティック |
| Image → Detailed cinematic (Uncensored) | vlm | ✓ | 上記 Uncensored 版 |
| Custom | idea, vlm | | Instruction を手編集して保持 |

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
| Show uncensored instruction presets | **既定 OFF**。ON で `uncensored: true` のプリセットを Idea/VLM の一覧に表示 |

旧拡張名 **llmuse** 時代の Settings キー（`llmuse_ollama_*` 等）も、未設定時のフォールバックとして参照されます。

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
    ├── presets.py
    └── wd_tagger.py
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
| 追加したプリセットが一覧に出ない | `presets.json` の **JSON 構文**を確認（1 文字の `"` 余りでも全体が読めません）。修正後 **更新** |
| `(Uncensored)` プリセットが出ない | Settings で **Show uncensored instruction presets** を ON → Apply → Idea/VLM **更新** |
| VLM 用プリセットが Idea に無い（逆も） | `presets[].for` が `vlm` / `idea` のみのため。両方で使う場合は `"for": ["idea", "vlm"]` |
| WD Tagger でキャラ名が出ない | **Character threshold** を下げる。元画像・モデルによっては general のみになることもある |
| WD Tagger 初回が失敗 | `pip install onnxruntime huggingface_hub`（または WebUI venv 内）。GPU があれば CUDA 版 onnxruntime も可 |

---

## 注意

- Forge 本体・`requirements.txt`・venv のピンは変更しません
- GGUF は git 管理対象外です（`*.gguf`）
- `install.py` は Linux で外部の公式インストーラを実行することがあります
- 追加の Instruction や Uncensored 向け文言が必要な場合は、`presets.json` にエントリを追加し、`uncensored: true` と Settings の表示 ON を組み合わせてください
