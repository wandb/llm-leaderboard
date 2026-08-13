#!/usr/bin/env python3
"""
W&B リーダーボードテーブル集計スクリプト

指定した W&B run の summary / artifact に含まれる leaderboard_table 系テーブルを
ダウンロードし、モデル×ベンチマークの wide / long 形式スコア表に統合します。
--write-back で統合結果を新しい run として W&B に書き戻すこともできます。

使い方の例:

    python scripts/analysis/wandb_merge_tables_from_run.py --run <entity>/<project>/<run_id>
    python scripts/analysis/wandb_merge_tables_from_run.py --run <entity>/<project>/<run_id> --write-back --project <target_project>

出力:
    <out>/merged_wide_scores.csv  モデル×ベンチマークのスコア表（wide 形式）
    <out>/scores_long.csv         Report 用の long 形式スコア表
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import wandb

SCORE_PRIORITY = [
    "AVG",
    "AVG_mtbench",
    "hallucination_resistance",
    "overall_score",
    "m_ifeval_score",
    "robust_score",
    "resolution_rate",
    "Overall Acc",
    "acc",
    "score",
    "Accuracy",
    "overall",
]


def is_table_meta(x) -> bool:
    """W&B summary のテーブルメタデータかどうかを判定"""
    return isinstance(x, dict) and x.get("_type") in {"table-file", "table"}


def load_table_json(path: Path) -> pd.DataFrame:
    """W&B table json を DataFrame として読み込む"""
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or "columns" not in obj or "data" not in obj:
        raise ValueError(f"Not a wandb table json: {path}")
    return pd.DataFrame(obj["data"], columns=obj["columns"])


def safe_name(s: str) -> str:
    """パスに使えるように / と : を置換"""
    return s.replace("/", "_").replace(":", "_")


def parse_benchmark_from_key(table_key: str) -> str:
    """テーブルキーからベンチマーク名を抽出"""
    if table_key == "leaderboard_table":
        return "aggregate"
    if table_key.endswith("_leaderboard_table"):
        return table_key[: -len("_leaderboard_table")]
    m = re.match(r"(.+?)_leaderboard_table_", table_key)
    if m:
        return m.group(1)
    m = re.match(r"(.+?)_table", table_key)
    return m.group(1) if m else table_key


def canonical_model_name(name: str):
    """nvidia__xxx 形式を nvidia/xxx に正規化"""
    if isinstance(name, str) and ("__" in name) and ("/" not in name):
        return name.replace("__", "/")
    return name


def choose_score_col(cols) -> Optional[str]:
    """スコア列を優先度順に選ぶ"""
    cols = list(cols)
    for c in SCORE_PRIORITY:
        if c in cols:
            return c
    for c in cols:
        if isinstance(c, str) and any(k in c.lower() for k in ["score", "acc", "rate"]):
            return c
    return None


def download_one_table_via_artifact(api: wandb.Api, meta: dict, out_dir: Path) -> Optional[Path]:
    """summary のテーブルメタデータから artifact 経由でテーブルをダウンロード"""
    art_path = meta.get("_latest_artifact_path") or meta.get("artifact_path")
    rel_path = meta.get("path")  # often like "media/table/xxx.table.json"
    if not art_path or not rel_path:
        return None

    artifact = api.artifact(art_path)
    root = out_dir / safe_name(art_path)
    root.mkdir(parents=True, exist_ok=True)

    # path_prefix で部分ダウンロード
    artifact.download(root=str(root), path_prefix=rel_path)

    local = root / rel_path
    if local.exists():
        return local

    # fallback: 配下の *.table.json を探す
    candidates = list(root.rglob("*.table.json"))
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        # 最も大きいものを採用
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
        return candidates[0]
    return None


def download_tables_from_run_files(run, out_dir: Path) -> List[Path]:
    """run.files() を走査して .table.json をダウンロード"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in run.files():
        if f.name.endswith(".table.json"):
            p = Path(f.download(root=str(out_dir), replace=True).name)
            paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="entity/project/run_id  e.g. <entity>/<project>/<run_id>")
    ap.add_argument("--out", default="artifacts/wandb_tables_dump", help="local output dir (default is already git-ignored)")
    ap.add_argument("--filter", default="leaderboard_table", help="substring filter, e.g. output_table or leaderboard_table or empty for all")
    ap.add_argument("--write-back", action="store_true", help="log merged table back to W&B as a new run")
    ap.add_argument("--project", default=None, help="override project for write-back")
    ap.add_argument("--entity", default=None, help="override entity for write-back")
    args = ap.parse_args()

    api = wandb.Api()
    run = api.run(args.run)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) summary のテーブルメタデータを優先
    table_files: List[Tuple[str, Path]] = []
    for k, v in dict(run.summary).items():
        if is_table_meta(v):
            if args.filter and args.filter not in k:
                continue
            p = download_one_table_via_artifact(api, v, out_root / "by_artifact")
            if p:
                table_files.append((k, p))

    # 2) run files から不足分を補完（summary に無いテーブルも拾う）
    seen_keys = {k for k, _ in table_files}
    files = download_tables_from_run_files(run, out_root / "by_run_files")
    for p in files:
        key = p.stem  # filename minus .json
        if key in seen_keys:
            continue
        if args.filter and args.filter not in key:
            continue
        table_files.append((key, p))

    if not table_files:
        raise SystemExit("No table json found from summary artifacts nor run files.")

    # wide スコア表を構築（1行=モデル、1列=ベンチマークスコア）
    wide_parts = []
    skipped = []

    for key, path in table_files:
        df = load_table_json(path)

        if "model_name" not in df.columns:
            skipped.append((key, "no model_name"))
            continue

        bench = parse_benchmark_from_key(key)
        score_col = choose_score_col(df.columns)
        if not score_col:
            skipped.append((key, "no score col"))
            continue

        tmp = df[["model_name", score_col]].copy()
        tmp["model_name"] = tmp["model_name"].map(canonical_model_name)

        metric_name = f"{bench}.{score_col}"
        tmp = tmp.rename(columns={score_col: metric_name}).set_index("model_name")
        wide_parts.append(tmp)

    if not wide_parts:
        raise SystemExit(f"No usable leaderboard tables. skipped={skipped[:20]}")

    merged_wide = pd.concat(wide_parts, axis=1)
    merged_wide = merged_wide.groupby(level=0).first().reset_index()

    # Report のグループ化棒グラフ用の long 形式
    score_cols = [c for c in merged_wide.columns if c != "model_name"]
    scores_long = (
        merged_wide.melt(
            id_vars=["model_name"],
            value_vars=score_cols,
            var_name="metric",
            value_name="score",
        )
        .dropna(subset=["score"])
    )
    scores_long["benchmark"] = scores_long["metric"].str.split(".", n=1).str[0]

    out_csv_wide = out_root / "merged_wide_scores.csv"
    out_csv_long = out_root / "scores_long.csv"
    merged_wide.to_csv(out_csv_wide, index=False, encoding="utf-8-sig")
    scores_long.to_csv(out_csv_long, index=False, encoding="utf-8-sig")

    print(f"Wide rows={len(merged_wide):,} cols={len(merged_wide.columns)} -> {out_csv_wide}")
    print(f"Long rows={len(scores_long):,} cols={len(scores_long.columns)} -> {out_csv_long}")

    if skipped:
        print("Skipped tables (first 20):")
        for x in skipped[:20]:
            print("  ", x)

    if args.write_back:
        entity = args.entity or run.entity
        project = args.project or run.project

        with wandb.init(entity=entity, project=project, job_type="postprocess", name=f"merge_scores_from_{run.id}") as wr:
            wr.config.update(
                {
                    "source_run": args.run,
                    "filter": args.filter,
                    "n_tables": len(table_files),
                    "n_models": int(len(merged_wide)),
                    "n_metrics": int(len(score_cols)),
                    "skipped_tables": int(len(skipped)),
                },
                allow_val_change=True,
            )

            wr.log(
                {
                    "merged_wide_scores": wandb.Table(dataframe=merged_wide),
                    "scores_long": wandb.Table(dataframe=scores_long),
                }
            )

            art = wandb.Artifact(name=f"merged-scores-{safe_name(run.id)}", type="dataset")
            art.add_file(str(out_csv_wide), name="merged_wide_scores.csv")
            art.add_file(str(out_csv_long), name="scores_long.csv")
            wr.log_artifact(art)

            print("Logged merged_wide_scores + scores_long + artifact back to W&B")


if __name__ == "__main__":
    main()
