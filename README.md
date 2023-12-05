# Nejumi-leaderboard Neo


## 目次

- [Install](#Install)
- [Data Preparation](#Data Preparation)


## Install
[] 複数のrepositoryがある場合のinstal方法を考える必要がある
[] サブモジュールがあるrepositoryのinstall instructionを書く

## Data Prepartion
### llm-jp-evalのデータセット準備
wandbのArtifactsを使う場合は、この手順は不要！　
下記に、参考までにwandbのArtifactsにデータを登録したプロセスを記す。
現状version1.0.0のデータセットがArtifactsに登録をされている。
1. jasterのdatasetを作成
```bash
  poetry run python scripts/preprocess_dataset.py  \
  --dataset-name all  \
  --output-dir dataset \
```
2. 作成されたdatasetを元に、upload_jaster.pyを用いてArtifactsに登録
```bash
python3 scripts/upload_jaster.py -e wandb-japan -p llm-leaderboard -d llm-jp-eval/dataset -v 1.0.0
```
### mt-benchのデータセット準備
wandbのArtifactsを使う場合は、この手順は不要！　
下記に、参考までにwandbのArtifactsにデータを登録したプロセスを記す。
現状2023年11月30日時点のデータセットがArtifactsに登録をされている。
1. 
```bash
  python3 scripts/upload_mtbench_question.py -e wandb-japan -p llm-leaderboard -v 20231130
```
