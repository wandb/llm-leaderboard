#!/usr/bin/env python3
"""
Build a provisional Taiwan leaderboard from completed W&B full-eval runs.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "base_config_taiwan.yaml"
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "taiwan_full_eval_models.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "taiwan_full_eval" / "provisional_leaderboard"
DEFAULT_JAPANESE_EXPORT = REPO_ROOT / "wandb_export_2026-06-25T13_17_55.153+09_00.csv"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True)


def expected_runs(manifest_path: Path) -> pd.DataFrame:
    manifest = _plain(OmegaConf.load(manifest_path))
    return pd.DataFrame(manifest.get("models", []))


def wandb_path(base_config_path: Path, entity: str | None, project: str | None) -> tuple[str, str]:
    cfg = OmegaConf.load(base_config_path)
    resolved_entity = entity or cfg.wandb.entity
    resolved_project = project or cfg.wandb.project
    return str(resolved_entity), str(resolved_project)


def find_latest_matching_runs(
    api: wandb.Api,
    path: str,
    run_names: set[str],
    limit: int,
) -> dict[str, Any]:
    matches: dict[str, Any] = {}
    for run in api.runs(
        path,
        order="-created_at",
        per_page=min(limit, 1000),
        include_sweeps=False,
    )[:limit]:
        if run.name not in run_names or run.name in matches:
            continue
        matches[run.name] = run
        if len(matches) == len(run_names):
            break
    return matches


def read_wandb_table(api: wandb.Api, entity: str, project: str, run_id: str, table_name: str) -> pd.DataFrame:
    artifact_path = f"{entity}/{project}/run-{run_id}-{table_name}:latest"
    artifact = api.artifact(artifact_path)
    with tempfile.TemporaryDirectory(prefix=f"{table_name}_") as tmpdir:
        local_json = artifact.get_entry(f"{table_name}.table.json").download(root=tmpdir)
        with open(local_json, encoding="utf-8") as f:
            table_json = json.load(f)
    return pd.DataFrame(data=table_json.get("data", []), columns=table_json.get("columns", []))


def read_run_tables(api: wandb.Api, entity: str, project: str, run: Any) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    if run.state != "finished":
        return None, None, f"run_state_{run.state}"
    try:
        leaderboard = read_wandb_table(api, entity, project, run.id, "taiwan_leaderboard_table")
        units = read_wandb_table(api, entity, project, run.id, "taiwan_unit_scores_table")
    except Exception as exc:
        return None, None, f"missing_table:{exc}"
    leaderboard["source_run_id"] = run.id
    leaderboard["source_run_name"] = run.name
    leaderboard["source_run_url"] = run.url
    units["source_run_id"] = run.id
    units["source_run_name"] = run.name
    return leaderboard, units, "ok"


def add_japanese_reference(leaderboard: pd.DataFrame, japanese_export: Path) -> pd.DataFrame:
    if leaderboard.empty or not japanese_export.exists():
        return leaderboard
    jp = pd.read_csv(japanese_export)
    if "runname" not in jp.columns:
        return leaderboard

    jp = jp.rename(
        columns={
            "runname": "japanese_runname",
            "TOTAL_SCORE": "japanese_total_score",
            "汎用的言語性能(GLP)_AVG": "japanese_glp",
            "アラインメント(ALT)_AVG": "japanese_alt",
        }
    )
    keep = [
        "japanese_runname",
        "japanese_total_score",
        "japanese_glp",
        "japanese_alt",
        "runid",
    ]
    jp = jp[[column for column in keep if column in jp.columns]].copy()

    def normalized_name(value: Any) -> str:
        text = str(value)
        for prefix in ("taiwan/full/",):
            if text.startswith(prefix):
                text = text[len(prefix) :]
        return text

    leaderboard = leaderboard.copy()
    if "japanese_runname_override" in leaderboard.columns:
        leaderboard["_jp_key"] = leaderboard["japanese_runname_override"].fillna(
            leaderboard["source_run_name"]
        ).map(normalized_name)
    else:
        leaderboard["_jp_key"] = leaderboard["source_run_name"].map(normalized_name)
    jp["_jp_key"] = jp["japanese_runname"].map(normalized_name)
    merged = leaderboard.merge(jp, on="_jp_key", how="left").drop(columns=["_jp_key"])
    if "japanese_total_score" in merged.columns and "Overall" in merged.columns:
        merged["taiwan_minus_japanese_total_points"] = (
            merged["Overall"] / 100.0 - merged["japanese_total_score"]
        )
    return merged


def write_markdown(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    if df.empty:
        path.write_text("_No completed Taiwan full-eval runs yet._\n", encoding="utf-8")
        return
    available = [column for column in columns if column in df.columns]
    out = df[available].copy()
    for column in out.select_dtypes(include=["float"]).columns:
        out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    path.write_text(out.to_markdown(index=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--japanese-export", type=Path, default=DEFAULT_JAPANESE_EXPORT)
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--entity")
    parser.add_argument("--project")
    parser.add_argument("--recent-run-limit", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    entity, project = wandb_path(args.base_config, args.entity, args.project)
    path = f"{entity}/{project}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    expected = expected_runs(args.manifest)
    run_names = set(expected["run_name"].astype(str))
    api = wandb.Api(timeout=60)
    matched_runs = find_latest_matching_runs(api, path, run_names, args.recent_run_limit)

    leaderboard_rows = []
    unit_rows = []
    status_rows = []
    for model in expected.to_dict(orient="records"):
        run_name = str(model["run_name"])
        run = matched_runs.get(run_name)
        if run is None:
            status = "missing_run"
        else:
            leaderboard, units, status = read_run_tables(api, entity, project, run)
            if leaderboard is not None:
                leaderboard["slug"] = model["slug"]
                if model.get("japanese_runname"):
                    leaderboard["japanese_runname_override"] = model["japanese_runname"]
                leaderboard_rows.append(leaderboard)
            if units is not None:
                units["slug"] = model["slug"]
                unit_rows.append(units)
        status_rows.append(
            {
                "slug": model["slug"],
                "run_name": run_name,
                "run_id": getattr(run, "id", None) if run else None,
                "state": getattr(run, "state", None) if run else None,
                "status": status,
                "url": getattr(run, "url", None) if run else None,
            }
        )

    leaderboard_df = pd.concat(leaderboard_rows, ignore_index=True) if leaderboard_rows else pd.DataFrame()
    unit_df = pd.concat(unit_rows, ignore_index=True) if unit_rows else pd.DataFrame()
    status_df = pd.DataFrame(status_rows)
    if not leaderboard_df.empty and "Overall" in leaderboard_df.columns:
        leaderboard_df = leaderboard_df.sort_values("Overall", ascending=False)
    leaderboard_df = add_japanese_reference(leaderboard_df, args.japanese_export)

    leaderboard_df.to_csv(args.output_dir / "leaderboard.csv", index=False)
    unit_df.to_csv(args.output_dir / "unit_scores.csv", index=False)
    status_df.to_csv(args.output_dir / "run_status.csv", index=False)
    write_markdown(
        leaderboard_df,
        args.output_dir / "leaderboard.md",
        [
            "slug",
            "model_name",
            "Overall",
            "GLP",
            "ALT",
            "missing_required_count",
            "pending_count",
            "japanese_total_score",
            "taiwan_minus_japanese_total_points",
            "source_run_id",
        ],
    )
    write_markdown(
        status_df,
        args.output_dir / "run_status.md",
        ["slug", "state", "status", "run_id", "url"],
    )
    print(f"Wrote {args.output_dir}")
    print(status_df.to_string(index=False))


if __name__ == "__main__":
    main()
