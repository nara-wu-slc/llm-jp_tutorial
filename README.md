# Lab4SLC llm-jp チュートリアル

```
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

```
# Pytorch をインストール (研究室の環境に合わせて CUDA-12.4対応版にする）
pip3 install torch --index-url https://download.pytorch.org/whl/cu124

# Huggingface Transformers をインストール
pip3 install transformers

# その他必要なものもインストール
pip3 install wheel accelerate flash-attn
```

```
# モデル保存用のキャッシュディレクトリの設定を読み込む
source /slc/share/dot.zshrc.slc
```

```
# ChatGPT的な対話を行うデモプログラムを回す
# CUDA_VISIBLE_DEVICES=0 は1枚目のGPUだけを使う、というおまじない
# -m の後はモデルの名前 (https://huggingface.co/llm-jp/llm-jp-3-13b 参照、但し 172b はメモリに載らないと思うので落としてない)
# "-instruct" がついたモデルを使う想定
# プログラムの立ち上げが終わると
# input>
# というプロンプトが出てくるので文字列を入力。空行のままEnterすると入力終了（入力が空であればプログラムを閉じる）
env CUDA_VISIBLE_DEVICES=0 ./sample_interactive.py -m llm-jp-3-1.8b-instruct
```

実行例
```
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████| 6/6 [00:05<00:00,  1.01it/s]
input> ドラえもんの兄について教えてください
input> 
ドラえもんには二人の兄弟がいます。彼らは「野比のび助」と「セワシ」という名前で、それぞれ異なる性格や特徴を持っています。

まず、「野比のび助」はドラえもんの父親であり、愛称は「パパ」です。彼はのび太よりも少し年上で、非常におおらかで優しい性格をしています。仕事は未来デパートのセールスマンとして働いており、時にはタイムマシンを使って過去ののび太たちのもとへ訪れます。

次に、「セワシ」はドラえもんの甥にあたり、のび太の孫息子にあたります。セワシものび太と同じようにおっちょこちょいな一面がありますが、未来から来ただけあって知識が豊富で、特にお小遣い稼ぎに関するアイデアをたくさん持っています。

この二人の兄弟の関係は、ドラえもんとその仲間たちが困っているときに助けになることがよくあります。例えば、のび太が宿題に追われているとき、セワシが最新の学習道具を紹介して手助けしたり、のび助が特別な料理を作って家族団らんの場を提供したりします。このようにして、三世代にわたって支え合いながら生活しているのが、ドラえもんの世界の魅力でもあります。</s>
input> 
```
