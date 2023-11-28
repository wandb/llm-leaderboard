# Nejumi-leaderboard Neo


## 目次

- [Install](#Install)
- [Data Preparation](#Data Preparation)


## Install
[] 複数のrepositoryがある場合のinstal方法を考える必要がある

## Data Prepartion
### llm-jp-evalのデータセット準備
wandbのArtifactsを使う場合は、この手順は不要。
下記に、wandbのArtifactsにデータを登録したプロセスを記す。
現状version1.0.0のデータセットがArtifactsに登録をされている。
1. まず、datasetを作成
```bash
  poetry run python scripts/preprocess_dataset.py  \
  --dataset-name all  \
  --output-dir dataset \
```
2. 作成されたdatasetを元に、upload_jaster.pyを用いてArtifactsに登録




