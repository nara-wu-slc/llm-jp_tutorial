# Lab4SLC llm-jp チュートリアル

## 1. リポジトリの取得と仮想環境の作成

```bash
# 作業ディレクトリに移動 (/path/to/dir は好きなところで良い、相対パスでもOK)
cd /path/to/dir

# 本チュートリアルのリポジトリをクローンして移動
git clone https://github.com/nara-wu-slc/llm-jp_tutorial.git
cd llm-jp_tutorial

# 本チュートリアル用のPython仮想環境を /path/to/dir/llm-jp_tutorial/.venv に作成
python3 -m venv .venv

# 仮想環境を有効化
source .venv/bin/activate
```

## 2. ライブラリのインストール

### (A) Transformers 版（通常モデル）を使う場合

```bash
# Pytorch をインストール (研究室の環境に合わせて CUDA-12.4対応版にする）
pip3 install torch --index-url https://download.pytorch.org/whl/cu124

# Huggingface Transformers をインストール
pip3 install transformers

# その他必要なものもインストール
pip3 install wheel
pip3 install accelerate flash-attn
```

### (B) GGUF 量子化モデル版（省メモリ・推奨）を使う場合

GPUメモリ（VRAM）の使用量を抑えて動かしたい場合は、`llama-cpp-python` を使って公式の GGUF 量子化モデル（`Q4_K_M`）を実行できます。

```bash
# 必要な補助パッケージをインストール
pip3 install huggingface-hub jinja2

# llama-cpp-python (CUDA 12.4対応ビルド済みwheel) をインストール
pip3 install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

※ Python のバージョン等の都合で上記 wheel が見つからない場合や、最新の `llama.cpp` 機能が必要な場合は、以下のようにソースから CUDA 有効でビルドしてインストールしてください（Ubuntu 22.04 LTS）：

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip3 install llama-cpp-python --no-cache-dir
```

## 3. キャッシュディレクトリの設定

```bash
# モデル保存用のキャッシュディレクトリの設定を読み込む
# これをしないと自分のホームディレクトリ配下にバカデカいファイルを読み込んでしまって無駄が多い
source /slc/share/dot.zshrc.slc
```

## 4. デモプログラムの実行

プログラムの立ち上げが終わると `input>` というプロンプトが出てくるので文字列を入力します。空行のまま Enter すると入力終了となり応答が生成されます（何も入力せずに空行のまま Enter するとプログラムを閉じます）。

### (1) GGUF 量子化モデル版（`sample_interactive_gguf.py`）

公式の GGUF 量子化モデル（デフォルトで 4bit 量子化の `Q4_K_M` を使用）を読み込んで対話を行います。

```bash
# CUDA_VISIBLE_DEVICES=0 は1枚目のGPUだけを使う、というおまじない
# -m の後はモデルの名前:
#   llm-jp-4.1-8b-thinking-gguf      (約5.5GB, 1枚のGPUでも余裕をもって動作)
#   llm-jp-4.1-32b-a3b-thinking-gguf (MoEモデル, 推論が高速)
#   llm-jp-4.1-33b-thinking-gguf     (Dense 33Bモデル)
env CUDA_VISIBLE_DEVICES=0 python3 ./sample_interactive_gguf.py -m llm-jp-4.1-8b-thinking-gguf
```

**GPUメモリが足りない場合のオプション：**
- `-ngl` (`--n-gpu-layers`)：GPU に載せるレイヤー数を指定します（デフォルトは `-1` で全レイヤーを GPU に配置）。32B / 33B モデルなどで GPU メモリが不足する場合は `-ngl 20` のように指定すると、残りのレイヤーを CPU メモリ側で処理できます。
- `-c` (`--ctx-size`)：コンテキスト長を指定します（デフォルトは `4096`）。
- `--reasoning-effort`：思考の深さを `low` / `medium` / `high` から指定できます（デフォルトは `medium`）。

```bash
# 例: 32b-a3b モデルで一部レイヤーのみ GPU にオフロードして動かす場合
env CUDA_VISIBLE_DEVICES=0 python3 ./sample_interactive_gguf.py -m llm-jp-4.1-32b-a3b-thinking-gguf -ngl 20
```

### (2) Transformers 版（`sample_interactive.py`）

非量子化モデルを Hugging Face Transformers で動かす場合のコマンドです。

```bash
# -m の後はモデルの名前 (https://huggingface.co/llm-jp 参照、但し 172b はメモリに載らないと思うので落としてない)
# 対話用途では "-instruct" または "-thinking" がついたモデルを使う想定:
#   llm-jp-3-1.8b-instruct / llm-jp-3-3.7b-instruct / llm-jp-3-13b-instruct
#   llm-jp-4.1-8b-thinking / llm-jp-4.1-32b-a3b-thinking / llm-jp-4.1-33b-thinking
env CUDA_VISIBLE_DEVICES=0 python3 ./sample_interactive.py -m llm-jp-3-1.8b-instruct
```

実行例
```text
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████| 6/6 [00:05<00:00,  1.01it/s]
input> ドラえもんの兄について教えてください
input> 
ドラえもんには二人の兄弟がいます。彼らは「野比のび助」と「セワシ」という名前で、それぞれ異なる性格や特徴を持っています。

まず、「野比のび助」はドラえもんの父親であり、愛称は「パパ」です。彼はのび太よりも少し年上で、非常におおらかで優しい性格をしています。仕事は未来デパートのセールスマンとして働いており、時にはタイムマシンを使って過去ののび太たちのもとへ訪れます。

次に、「セワシ」はドラえもんの甥にあたり、のび太の孫息子にあたります。セワシものび太と同じようにおっちょこちょいな一面がありますが、未来から来ただけあって知識が豊富で、特にお小遣い稼ぎに関するアイデアをたくさん持っています。

この二人の兄弟の関係は、ドラえもんとその仲間たちが困っているときに助けになることがよくあります。例えば、のび太が宿題に追われているとき、セワシが最新の学習道具を紹介して手助けしたり、のび助が特別な料理を作って家族団らんの場を提供したりします。このようにして、三世代にわたって支え合いながら生活しているのが、ドラえもんの世界の魅力でもあります。</s>
input> 
```
