from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
from omegaconf import OmegaConf

from config_singleton import WandbConfigSingleton

def _format_table_name(template: str, cfg: Any) -> str:
    return template.format(num_few_shots=cfg.get("num_few_shots", 2))


def _coerce_score_to_native_scale(value: Any, scale: str) -> float:
    if value is None:
        return float("nan")
    is_percent_literal = False
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return float("nan")
        if value.endswith("%"):
            is_percent_literal = True
            value = value[:-1].strip()
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(score):
        return float("nan")
    if not is_percent_literal:
        return score
    if scale == "fraction":
        return score / 100.0
    if scale == "judge_0_10":
        return score / 10.0
    if scale == "percent":
        return score
    raise ValueError(f"Unsupported score scale: {scale}")


def _normalize_score(value: Any, scale: str) -> float:
    score = _coerce_score_to_native_scale(value, scale)
    if np.isnan(score):
        return float("nan")
    if scale == "fraction":
        return score * 100.0
    if scale == "judge_0_10":
        return score * 10.0
    if scale == "percent":
        return score
    raise ValueError(f"Unsupported score scale: {scale}")


def _read_wandb_table_once(table_name: str, run) -> pd.DataFrame:
    import json
    import tempfile

    artifact_path = f"{run.entity}/{run.project}/run-{run.id}-{table_name}:latest"
    artifact = run.use_artifact(artifact_path)
    with tempfile.TemporaryDirectory(prefix=f"{table_name}_") as tmpdir:
        local_json = artifact.get_entry(f"{table_name}.table.json").download(root=tmpdir)
        with open(local_json, encoding="utf-8") as f:
            table_json = json.load(f)
    return pd.DataFrame(data=table_json.get("data", []), columns=table_json.get("columns", []))


def _read_score_source(run, cfg, source: dict[str, Any]) -> tuple[Any, str]:
    table_name = _format_table_name(source["table_name"], cfg)
    column = source["column"]
    table = _read_wandb_table_once(table_name=table_name, run=run)
    if column not in table.columns:
        raise KeyError(f"{column} not found in {table_name}")
    return table[column].iloc[0], table_name


def _unit_raw_score(run, cfg, unit: dict[str, Any]) -> tuple[float, str]:
    if "sources" in unit:
        values = []
        table_names = []
        for source in unit["sources"]:
            value, table_name = _read_score_source(run, cfg, source)
            values.append(_coerce_score_to_native_scale(value, unit["scale"]))
            table_names.append(table_name)
        if unit.get("aggregation", "mean") != "mean":
            raise ValueError(f"Unsupported aggregation: {unit.get('aggregation')}")
        return float(np.mean(values)), ",".join(table_names)

    source = {"table_name": unit["table_name"], "column": unit["column"]}
    value, table_name = _read_score_source(run, cfg, source)
    return _coerce_score_to_native_scale(value, unit["scale"]), table_name


def _mean_if_complete(values: list[float]) -> float:
    if not values or any(np.isnan(v) for v in values):
        return float("nan")
    return float(np.mean(values))


def _mean_for_included_units(unit_df: pd.DataFrame, category: str | None = None) -> float:
    included = unit_df[unit_df["score_included"]]
    if category is not None:
        included = included[included["category"] == category]
    return _mean_if_complete(included["score_0_to_100"].tolist())


def _validate_taxonomy(taxonomy: dict[str, Any]) -> None:
    categories = taxonomy.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("taxonomy.categories must be a non-empty mapping")

    units = taxonomy.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("taxonomy.units must be a non-empty list")

    valid_scales = {"fraction", "judge_0_10", "percent"}
    errors = []
    for category, category_cfg in categories.items():
        weight = (category_cfg or {}).get("weight")
        try:
            if float(weight) <= 0:
                errors.append(f"categories.{category}.weight must be positive")
        except (TypeError, ValueError):
            errors.append(f"categories.{category}.weight must be numeric")

    for unit in units:
        unit_id = unit.get("id", "<missing>")
        if unit.get("category") not in categories:
            errors.append(f"{unit_id}: category {unit.get('category')!r} is not declared")
        scale = unit.get("scale")
        if scale not in valid_scales:
            errors.append(f"{unit_id}: scale must be one of {sorted(valid_scales)}, got {scale!r}")
        if bool(unit.get("required", True)) and bool(unit.get("pending", False)):
            errors.append(f"{unit_id}: required=true and pending=true are mutually exclusive")

    if errors:
        raise ValueError("Invalid Taiwan taxonomy: " + "; ".join(errors))


def _category_weights(taxonomy: dict[str, Any]) -> dict[str, float]:
    return {
        str(category): float(category_cfg["weight"])
        for category, category_cfg in taxonomy["categories"].items()
    }


def _weighted_overall(category_scores: dict[str, float], weights: dict[str, float]) -> float:
    if not category_scores or any(np.isnan(value) for value in category_scores.values()):
        return float("nan")
    total_weight = sum(weights.get(category, 1.0) for category in category_scores)
    if total_weight <= 0:
        return float("nan")
    return float(
        sum(category_scores[category] * weights.get(category, 1.0) for category in category_scores)
        / total_weight
    )


def _required_missing_rows(unit_df: pd.DataFrame) -> pd.DataFrame:
    return unit_df[
        unit_df["required"]
        & unit_df["score_included"]
        & unit_df["score_0_to_100"].apply(lambda value: bool(np.isnan(value)))
    ]


def _write_aggregate_summary(
    run: Any,
    *,
    category_scores: dict[str, float],
    overall: float,
    missing_required_count: int,
    pending_count: int,
) -> None:
    summary_scores = {
        "GLP": category_scores.get("GLP", float("nan")),
        "ALT": category_scores.get("ALT", float("nan")),
        "Overall": overall,
    }
    for name, value in summary_scores.items():
        serialized = float(value) if np.isfinite(value) else None
        run.summary[name] = serialized
        run.summary[f"taiwan_{name.casefold()}_score"] = serialized
    run.summary["taiwan_missing_required_count"] = int(missing_required_count)
    run.summary["taiwan_pending_count"] = int(pending_count)


def evaluate():
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config

    taxonomy_path = Path(
        cfg.get("taiwan_aggregate", {}).get(
            "taxonomy_path", "taxonomies/nejumi45_taiwan.yaml"
        )
    )
    taxonomy = OmegaConf.to_container(OmegaConf.load(taxonomy_path), resolve=True)
    _validate_taxonomy(taxonomy)
    units = taxonomy["units"]
    category_weights = _category_weights(taxonomy)

    unit_rows = []
    leaderboard_dict = {
        "model_name": cfg.model.pretrained_model_name_or_path,
        "model_size_category": cfg.model.get("size_category", np.nan),
        "base_model": cfg.model.get("base_model", np.nan),
        "taxonomy_version": taxonomy["version"],
    }

    for unit in units:
        status = "ok"
        raw_score = float("nan")
        normalized_score = float("nan")
        source_tables = ""
        error = ""
        try:
            raw_score, source_tables = _unit_raw_score(run, cfg, unit)
            normalized_score = _normalize_score(raw_score, unit["scale"])
        except Exception as exc:
            status = "missing_required" if unit.get("required", True) else "missing"
            if unit.get("pending", False):
                status = "pending"
            error = str(exc)

        leaderboard_dict[unit["display_name"]] = normalized_score
        unit_rows.append(
            {
                "unit_id": unit["id"],
                "category": unit["category"],
                "display_name": unit["display_name"],
                "score_0_to_100": normalized_score,
                "raw_score": raw_score,
                "scale": unit["scale"],
                "required": bool(unit.get("required", True)),
                "pending": bool(unit.get("pending", False)),
                "score_included": not bool(unit.get("pending", False)),
                "status": status,
                "source_tables": source_tables,
                "error": error,
            }
        )

    unit_df = pd.DataFrame(unit_rows)

    category_scores = {
        category: _mean_for_included_units(unit_df, category)
        for category in category_weights
    }
    leaderboard_dict.update(category_scores)
    leaderboard_dict["Overall"] = _weighted_overall(category_scores, category_weights)
    leaderboard_dict["overall_weighting"] = "taxonomy_category_weighted_mean"
    missing_required = _required_missing_rows(unit_df)
    leaderboard_dict["missing_required_count"] = int(len(missing_required))
    leaderboard_dict["pending_count"] = int((unit_df["status"] == "pending").sum())

    leaderboard_table = pd.DataFrame([leaderboard_dict])
    glp_radar = unit_df.query("category == 'GLP'")[
        ["display_name", "score_0_to_100", "status"]
    ].rename(columns={"display_name": "category", "score_0_to_100": "score"})
    alt_radar = unit_df.query("category == 'ALT'")[
        ["display_name", "score_0_to_100", "status"]
    ].rename(columns={"display_name": "category", "score_0_to_100": "score"})

    run.log(
        {
            "taiwan_leaderboard_table": wandb.Table(dataframe=leaderboard_table),
            "taiwan_unit_scores_table": wandb.Table(dataframe=unit_df),
            "taiwan_glp_radar_table": wandb.Table(dataframe=glp_radar),
            "taiwan_alt_radar_table": wandb.Table(dataframe=alt_radar),
        }
    )
    _write_aggregate_summary(
        run,
        category_scores=category_scores,
        overall=leaderboard_dict["Overall"],
        missing_required_count=len(missing_required),
        pending_count=leaderboard_dict["pending_count"],
    )
    if not missing_required.empty:
        details = "; ".join(
            f"{row.unit_id}: {row.error}"
            for row in missing_required.itertuples(index=False)
        )
        raise RuntimeError(
            "Taiwan aggregate is incomplete; required benchmark scores are "
            f"missing ({len(missing_required)}): {details}"
        )
