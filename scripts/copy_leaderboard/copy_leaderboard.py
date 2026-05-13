#!/usr/bin/env python3
"""
Copy Leaderboard Tool

A tool to copy W&B Nejumi Leaderboard runs, artifacts, and reports
to your own W&B environment.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple
from unittest.mock import patch

import click
import requests
import yaml

import wandb
from wandb.apis.public import Run
from wandb.util import coalesce, get_module, remove_keys_with_none_values

# Reports API
# Try wandb_workspaces first (newer API), then fall back to wandb.apis.reports
try:
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces.reports.v2 import Report
    HAS_REPORTS = True
    REPORTS_API_VERSION = "v2"
except ImportError:
    try:
        with patch("click.echo"):
            import wandb.apis.reports as wr
            from wandb.apis.reports import Report
        HAS_REPORTS = True
        REPORTS_API_VERSION = "legacy"
    except ImportError:
        HAS_REPORTS = False
        REPORTS_API_VERSION = None
        wr = None
        Report = None

pl = get_module(
    "polars",
    required="Please install polars: `pip install polars`",
)

_tqdm = get_module(
    "tqdm",
    required="Please install tqdm: `pip install tqdm`",
)
tqdm = _tqdm.tqdm


class Config:
    """Configuration loader for copy_leaderboard."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    @property
    def source(self) -> Dict[str, Any]:
        return self._config.get("source", {})

    @property
    def destination(self) -> Dict[str, Any]:
        return self._config.get("destination", {})

    @property
    def artifacts(self) -> List[str]:
        return self._config.get("artifacts", [])

    @property
    def options(self) -> Dict[str, Any]:
        return self._config.get("options", {})


def convert_to_serializable(obj: Any) -> Any:
    """Convert wandb objects to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    if hasattr(obj, 'keys'):  # dict-like (including SummarySubDict)
        try:
            return {k: convert_to_serializable(v) for k, v in dict(obj).items()}
        except:
            return str(obj)
    # Try to convert to basic types
    try:
        return float(obj)
    except (TypeError, ValueError):
        pass
    try:
        return str(obj)
    except:
        return None


class WandbParquetRun:
    """Wrapper for W&B Run with parquet metrics support."""

    def __init__(self, run: Run, src_api: wandb.Api):
        self.run = run
        self.api = src_api

    def run_id(self) -> str:
        return self.run.id

    def entity(self) -> str:
        return self.run.entity

    def project(self) -> str:
        return self.run.project

    def config(self) -> Dict[str, Any]:
        return dict(self.run.config)

    def summary(self) -> Dict[str, Any]:
        """Get summary as serializable dict."""
        return convert_to_serializable(self.run.summary)

    def metrics(self) -> Iterable[Dict[str, float]]:
        """Get metrics from parquet files if available, otherwise scan_history."""
        history_paths = []
        try:
            for art in self.run.logged_artifacts():
                if art.type != "wandb-history":
                    continue
                path = art.download()
                history_paths.append(path)
        except Exception:
            pass

        if not history_paths:
            wandb.termwarn("No parquet files detected -- using scan_history")
            yield from self.run.scan_history()
            return

        for path in history_paths:
            for p in Path(path).glob("*.parquet"):
                df = pl.read_parquet(p)
                for row in df.iter_rows(named=True):
                    row = remove_keys_with_none_values(row)
                    yield row

    def run_group(self) -> Optional[str]:
        return self.run.group

    def job_type(self) -> Optional[str]:
        return self.run.job_type

    def display_name(self) -> str:
        return self.run.display_name

    def notes(self) -> Optional[str]:
        return self.run.notes

    def tags(self) -> List[str]:
        return list(self.run.tags) if self.run.tags else []

    def runtime(self) -> Optional[int]:
        wandb_runtime = self.run.summary.get("_wandb", {}).get("runtime")
        base_runtime = self.run.summary.get("_runtime")
        t = coalesce(wandb_runtime, base_runtime)
        if t is None:
            return t
        return int(t)

    def start_time(self) -> Optional[int]:
        t = dt.fromisoformat(self.run.created_at).timestamp()
        return int(t)

    def logged_artifacts(self) -> Iterable[Tuple[wandb.Artifact, str, str]]:
        """Get logged artifacts for this run. Returns (artifact, original_name, download_path) tuples."""
        try:
            arts = self.run.logged_artifacts()
            for art in arts:
                try:
                    name, _ = art.name.split(":v")
                    with patch("click.echo"):
                        download_path = art.download()

                    # Create new artifact
                    new_art = wandb.Artifact(name, "temp")
                    new_art._type = art.type
                    with patch("click.echo"):
                        new_art.add_dir(download_path)

                    yield new_art, art.name, download_path
                except Exception as e:
                    wandb.termwarn(f"Skipping logged artifact {art.name}: {e}")
                    continue
        except Exception as e:
            wandb.termwarn(f"Could not get logged artifacts: {e}")

    def used_artifacts(self) -> Iterable[Tuple[wandb.Artifact, str, str]]:
        """Get used artifacts for this run. Returns (artifact, original_name, download_path) tuples."""
        try:
            arts = self.run.used_artifacts()
            for art in arts:
                try:
                    name, _ = art.name.split(":v")
                    with patch("click.echo"):
                        download_path = art.download()

                    # Create new artifact
                    new_art = wandb.Artifact(name, "temp")
                    new_art._type = art.type
                    with patch("click.echo"):
                        new_art.add_dir(download_path)

                    yield new_art, art.name, download_path
                except Exception as e:
                    wandb.termwarn(f"Skipping used artifact {art.name}: {e}")
                    continue
        except Exception as e:
            wandb.termwarn(f"Could not get used artifacts: {e}")


class LeaderboardMigrator:
    """Main class for migrating W&B leaderboard data."""

    def __init__(
        self,
        src_base_url: str,
        src_api_key: str,
        dst_base_url: str,
        dst_api_key: str,
    ):
        self.src_base_url = src_base_url
        self.src_api_key = src_api_key
        self.dst_base_url = dst_base_url
        self.dst_api_key = dst_api_key

        self.src_api = wandb.Api(
            api_key=src_api_key,
            overrides={"base_url": src_base_url},
        )
        # Lazy init dst_api since it may fail authentication
        self._dst_api = None

    @property
    def dst_api(self) -> Optional[wandb.Api]:
        """Lazily initialize destination API."""
        if self._dst_api is None:
            try:
                self._dst_api = wandb.Api(
                    api_key=self.dst_api_key,
                    overrides={"base_url": self.dst_base_url},
                )
            except Exception as e:
                wandb.termwarn(f"Could not initialize destination API: {e}")
        return self._dst_api

    def collect_runs(
        self,
        entity: str,
        project: str,
        tag: Optional[str] = None,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        skip_ids: Optional[List[str]] = None,
    ) -> Generator[WandbParquetRun, None, None]:
        """Collect runs from source project."""
        filters: Dict[str, Any] = {}

        if tag:
            filters["tags"] = {"$in": [tag]}
        if skip_ids:
            filters["name"] = {"$nin": skip_ids}
        if start_date:
            filters["createdAt"] = {"$gte": start_date}

        runs = self.src_api.runs(f"{entity}/{project}", filters=filters)

        for i, run in enumerate(runs):
            if limit and i >= limit:
                break
            yield WandbParquetRun(run, self.src_api)

    def migrate_run(
        self,
        run: WandbParquetRun,
        dst_entity: str,
        dst_project: str,
    ) -> bool:
        """Migrate a single run to destination with its artifacts."""
        try:
            # First, collect artifacts while logged into source
            wandb.login(key=self.src_api_key, host=self.src_base_url, relogin=True)
            
            # Collect used and logged artifacts before switching credentials
            used_arts = list(run.used_artifacts())
            logged_arts = list(run.logged_artifacts())
            
            # Now login to destination and set environment
            os.environ["WANDB_API_KEY"] = self.dst_api_key
            os.environ["WANDB_BASE_URL"] = self.dst_base_url
            wandb.login(key=self.dst_api_key, host=self.dst_base_url, relogin=True)

            # Initialize a new run in destination
            new_run = wandb.init(
                entity=dst_entity,
                project=dst_project,
                name=run.display_name(),
                config=run.config(),
                tags=run.tags(),
                notes=run.notes(),
                group=run.run_group(),
                job_type=run.job_type(),
                reinit=True,
            )

            try:
                # Log metrics
                for metric in run.metrics():
                    step = metric.pop("_step", None)
                    # Convert step to int if it's a float
                    if step is not None:
                        step = int(step)
                    # Convert metric values to serializable types
                    clean_metric = convert_to_serializable(metric)
                    if clean_metric:
                        new_run.log(clean_metric, step=step)

                # Log summary
                summary = run.summary()
                if summary:
                    for key, value in summary.items():
                        if not key.startswith("_"):
                            try:
                                new_run.summary[key] = convert_to_serializable(value)
                            except Exception:
                                pass

                # Log table artifacts as wandb.Table objects
                # This works around SDK limitations with log_artifact
                tables_logged = 0
                for art, original_name, download_path in logged_arts:
                    try:
                        # Check if this is a run_table type artifact
                        if art.type == "run_table" or "table" in original_name.lower():
                            if download_path and os.path.exists(download_path):
                                # Find .table.json files
                                for filename in os.listdir(download_path):
                                    if filename.endswith(".table.json"):
                                        table_file = os.path.join(download_path, filename)
                                        try:
                                            # Load table from JSON
                                            with open(table_file) as f:
                                                table_data = json.load(f)
                                            
                                            # Create wandb.Table from the data
                                            columns = table_data.get("columns", [])
                                            data = table_data.get("data", [])
                                            table = wandb.Table(columns=columns, data=data)
                                            
                                            # Extract table key from filename
                                            table_key = filename.replace(".table.json", "")
                                            new_run.log({table_key: table})
                                            tables_logged += 1
                                        except Exception as te:
                                            wandb.termwarn(f"Failed to log table {filename}: {te}")
                    except Exception as e:
                        wandb.termwarn(f"Failed to process artifact {original_name}: {e}")
                
                if tables_logged > 0:
                    wandb.termlog(f"  Logged {tables_logged} tables")
            finally:
                new_run.finish()

            return True
        except Exception as e:
            wandb.termerror(f"Failed to migrate run {run.run_id()}: {e}")
            return False

    def migrate_runs(
        self,
        entity: str,
        project: str,
        dst_entity: str,
        dst_project: str,
        tag: Optional[str] = None,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        max_workers: int = 1,  # Sequential by default to avoid reinit issues
    ) -> Dict[str, int]:
        """Migrate multiple runs with progress tracking."""
        # Get existing run IDs in destination to skip
        skip_ids = self._get_existing_run_ids(dst_entity, dst_project)

        runs = list(self.collect_runs(
            entity, project, tag=tag, limit=limit,
            start_date=start_date, skip_ids=skip_ids
        ))

        if not runs:
            wandb.termlog("No runs to migrate.")
            return {"success": 0, "failed": 0, "skipped": len(skip_ids)}

        wandb.termlog(f"Migrating {len(runs)} runs (skipping {len(skip_ids)} existing)...")

        success, failed = 0, 0

        # Run sequentially to avoid wandb.init() conflicts
        for run in tqdm(runs, desc="Migrating runs", unit="run"):
            try:
                if self.migrate_run(run, dst_entity, dst_project):
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                wandb.termerror(f"Migration error: {e}")
                failed += 1

        return {"success": success, "failed": failed, "skipped": len(skip_ids)}

    def migrate_artifact(
        self,
        artifact_path: str,
        dst_entity: str,
        dst_project: str,
    ) -> bool:
        """Migrate a single artifact to destination."""
        try:
            # Login to source to download
            wandb.login(key=self.src_api_key, host=self.src_base_url, relogin=True)
            
            # Download artifact from source
            art = self.src_api.artifact(artifact_path)
            local_path = art.download()

            # Parse artifact info
            name = art.name.split(":")[0].split("/")[-1]
            art_type = art.type

            # Create new artifact in destination
            new_art = wandb.Artifact(name, type=art_type)
            new_art.add_dir(local_path)

            # Login to destination explicitly
            wandb.login(key=self.dst_api_key, host=self.dst_base_url, relogin=True)

            # Log to destination
            with wandb.init(
                entity=dst_entity,
                project=dst_project,
                job_type="dataset_migration",
                name=f"migrate-{name}",
                reinit=True,
            ) as init_run:
                init_run.log_artifact(new_art)

            wandb.termlog(f"Migrated artifact: {artifact_path}")
            return True

        except Exception as e:
            wandb.termerror(f"Failed to migrate artifact {artifact_path}: {e}")
            return False

    def migrate_artifacts(
        self,
        artifact_paths: List[str],
        dst_entity: str,
        dst_project: str,
    ) -> Dict[str, int]:
        """Migrate multiple artifacts."""
        success, failed = 0, 0

        for art_path in tqdm(artifact_paths, desc="Migrating artifacts", unit="art"):
            if self.migrate_artifact(art_path, dst_entity, dst_project):
                success += 1
            else:
                failed += 1

        return {"success": success, "failed": failed}

    def create_report(
        self,
        dst_entity: str,
        dst_project: str,
        title: str = "Nejumi Leaderboard",
        description: str = "Migrated Nejumi LLM Leaderboard",
    ) -> Optional[str]:
        """Create a new report in destination project with leaderboard_table panel."""
        if not HAS_REPORTS:
            wandb.termwarn("wandb reports API not available. Skipping report creation.")
            return None

        try:
            report = wr.Report(
                project=dst_project,
                entity=dst_entity,
                title=title,
                description=description,
            )

            # Add leaderboard_table panel with full width layout
            # Layout: x=0, y=0, w=24 (full width), h=16 (tall)
            panels = [
                wr.WeavePanelSummaryTable(
                    table_name="leaderboard_table",
                    layout=wr.Layout(x=0, y=0, w=24, h=16),
                ),
            ]

            report.blocks = [
                wr.H1(text=title),
                wr.P(text=description),
                wr.PanelGrid(
                    panels=panels,
                    runsets=[
                        wr.Runset(
                            project=dst_project,
                            entity=dst_entity,
                            filters="tags in ['leaderboard']",
                        ),
                    ],
                ),
            ]

            report.save()
            wandb.termlog(f"Created report: {report.url}")
            return report.url

        except Exception as e:
            wandb.termerror(f"Failed to create report: {e}")
            return None

    def _get_existing_run_ids(self, entity: str, project: str) -> List[str]:
        """Get IDs of runs already in destination project."""
        if self.dst_api is None:
            return []
        try:
            runs = self.dst_api.runs(f"{entity}/{project}")
            return [run.id for run in runs]
        except Exception:
            return []

    def incremental_update(
        self,
        entity: str,
        project: str,
        dst_entity: str,
        dst_project: str,
        tag: Optional[str] = None,
        state_file: str = ".last_sync.txt",
    ) -> Dict[str, int]:
        """Perform incremental update based on last sync time."""
        last_sync = None
        state_path = Path(state_file)

        if state_path.exists():
            with open(state_path) as f:
                last_sync = f.read().strip()
            wandb.termlog(f"Syncing runs created after: {last_sync}")
        else:
            wandb.termlog("First sync - migrating all runs...")

        now = dt.now().isoformat()

        result = self.migrate_runs(
            entity=entity,
            project=project,
            dst_entity=dst_entity,
            dst_project=dst_project,
            tag=tag,
            start_date=last_sync,
        )

        # Save sync time
        with open(state_path, "w") as f:
            f.write(now)

        return result


@click.group()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.pass_context
def cli(ctx, config: str):
    """Copy W&B Leaderboard to your own environment."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config(config)


@cli.command()
@click.option("--src-api-key", envvar="WANDB_SRC_API_KEY", help="Source W&B API key")
@click.option("--dst-api-key", envvar="WANDB_DST_API_KEY", help="Destination W&B API key")
@click.option("--tag", default=None, help="Filter runs by tag")
@click.option("--limit", type=int, default=None, help="Limit number of runs")
@click.pass_context
def migrate_runs(ctx, src_api_key: str, dst_api_key: str, tag: str, limit: int):
    """Migrate runs from source to destination."""
    config = ctx.obj["config"]
    src = config.source
    dst = config.destination
    opts = config.options

    migrator = LeaderboardMigrator(
        src_base_url=src.get("base_url", "https://api.wandb.ai"),
        src_api_key=src_api_key,
        dst_base_url=dst.get("base_url", "https://api.wandb.ai"),
        dst_api_key=dst_api_key,
    )

    result = migrator.migrate_runs(
        entity=src["entity"],
        project=src["project"],
        dst_entity=dst["entity"],
        dst_project=dst["project"],
        tag=tag or src.get("run_tag"),
        limit=limit or opts.get("limit"),
        start_date=opts.get("start_date"),
        max_workers=opts.get("max_workers", 1),
    )

    click.echo(f"\nMigration complete: {result}")


@cli.command()
@click.option("--src-api-key", envvar="WANDB_SRC_API_KEY", help="Source W&B API key")
@click.option("--dst-api-key", envvar="WANDB_DST_API_KEY", help="Destination W&B API key")
@click.pass_context
def migrate_artifacts(ctx, src_api_key: str, dst_api_key: str):
    """Migrate artifacts from source to destination."""
    config = ctx.obj["config"]
    dst = config.destination

    migrator = LeaderboardMigrator(
        src_base_url=config.source.get("base_url", "https://api.wandb.ai"),
        src_api_key=src_api_key,
        dst_base_url=dst.get("base_url", "https://api.wandb.ai"),
        dst_api_key=dst_api_key,
    )

    result = migrator.migrate_artifacts(
        artifact_paths=config.artifacts,
        dst_entity=dst["entity"],
        dst_project=dst["project"],
    )

    click.echo(f"\nArtifact migration complete: {result}")


@cli.command()
@click.option("--dst-api-key", envvar="WANDB_DST_API_KEY", help="Destination W&B API key")
@click.option("--title", default="Nejumi Leaderboard", help="Report title")
@click.pass_context
def create_report(ctx, dst_api_key: str, title: str):
    """Create a new leaderboard report with leaderboard_table panel."""
    config = ctx.obj["config"]
    dst = config.destination

    # Login to destination explicitly
    dst_base_url = dst.get("base_url", "https://api.wandb.ai")
    
    # Set environment variables that the Reports API may use
    os.environ["WANDB_API_KEY"] = dst_api_key
    os.environ["WANDB_BASE_URL"] = dst_base_url
    
    wandb.login(key=dst_api_key, host=dst_base_url, relogin=True)

    # Create report directly without full migrator initialization
    if not HAS_REPORTS:
        click.echo("wandb reports API not available.")
        return

    try:
        # Create report with leaderboard_table panel (full width)
        report = wr.Report(
            project=dst["project"],
            entity=dst["entity"],
            title=title,
            description="Migrated Nejumi LLM Leaderboard",
        )

        # Full width layout: w=24, h=16
        panels = [
            wr.WeavePanelSummaryTable(
                table_name="leaderboard_table",
                layout=wr.Layout(x=0, y=0, w=24, h=16),
            ),
        ]

        report.blocks = [
            wr.H1(text=title),
            wr.P(text="Migrated Nejumi LLM Leaderboard"),
            wr.PanelGrid(
                panels=panels,
                runsets=[
                    wr.Runset(
                        project=dst["project"],
                        entity=dst["entity"],
                        filters="tags in ['leaderboard']",
                    ),
                ],
            ),
        ]

        report.save()
        url = report.url
    except Exception as e:
        click.echo(f"Failed to create report: {e}")
        return

    if url:
        click.echo(f"\nReport created: {url}")


@cli.command()
@click.option("--src-api-key", envvar="WANDB_SRC_API_KEY", help="Source W&B API key")
@click.option("--dst-api-key", envvar="WANDB_DST_API_KEY", help="Destination W&B API key")
@click.option("--state-file", default=".last_sync.txt", help="State file for tracking last sync")
@click.pass_context
def sync(ctx, src_api_key: str, dst_api_key: str, state_file: str):
    """Incrementally sync new runs since last update."""
    config = ctx.obj["config"]
    src = config.source
    dst = config.destination

    migrator = LeaderboardMigrator(
        src_base_url=src.get("base_url", "https://api.wandb.ai"),
        src_api_key=src_api_key,
        dst_base_url=dst.get("base_url", "https://api.wandb.ai"),
        dst_api_key=dst_api_key,
    )

    result = migrator.incremental_update(
        entity=src["entity"],
        project=src["project"],
        dst_entity=dst["entity"],
        dst_project=dst["project"],
        tag=src.get("run_tag"),
        state_file=state_file,
    )

    click.echo(f"\nSync complete: {result}")


@cli.command()
@click.option("--src-api-key", envvar="WANDB_SRC_API_KEY", help="Source W&B API key")
@click.option("--dst-api-key", envvar="WANDB_DST_API_KEY", help="Destination W&B API key")
@click.option("--tag", default=None, help="Filter runs by tag")
@click.option("--limit", type=int, default=None, help="Limit number of runs")
@click.pass_context
def full_migration(ctx, src_api_key: str, dst_api_key: str, tag: str, limit: int):
    """Perform full migration: artifacts, runs, and create report."""
    config = ctx.obj["config"]
    src = config.source
    dst = config.destination
    opts = config.options

    migrator = LeaderboardMigrator(
        src_base_url=src.get("base_url", "https://api.wandb.ai"),
        src_api_key=src_api_key,
        dst_base_url=dst.get("base_url", "https://api.wandb.ai"),
        dst_api_key=dst_api_key,
    )

    dst_base_url = dst.get("base_url", "https://api.wandb.ai")

    click.echo("=" * 50)
    click.echo("Step 1/3: Migrating evaluation datasets (artifacts from config)...")
    click.echo("=" * 50)
    art_result = migrator.migrate_artifacts(
        artifact_paths=config.artifacts,
        dst_entity=dst["entity"],
        dst_project=dst["project"],
    )
    click.echo(f"Dataset artifacts: {art_result}")

    click.echo("\n" + "=" * 50)
    click.echo("Step 2/3: Migrating runs with their artifacts...")
    click.echo("  (includes used_artifacts and logged_artifacts)")
    click.echo("=" * 50)
    run_result = migrator.migrate_runs(
        entity=src["entity"],
        project=src["project"],
        dst_entity=dst["entity"],
        dst_project=dst["project"],
        tag=tag or src.get("run_tag"),
        limit=limit or opts.get("limit"),
    )
    click.echo(f"Runs: {run_result}")

    click.echo("\n" + "=" * 50)
    click.echo("Step 3/3: Creating report...")
    click.echo("=" * 50)
    # Set environment and login to destination explicitly
    os.environ["WANDB_API_KEY"] = dst_api_key
    os.environ["WANDB_BASE_URL"] = dst_base_url
    wandb.login(key=dst_api_key, host=dst_base_url, relogin=True)
    
    report_url = migrator.create_report(
        dst_entity=dst["entity"],
        dst_project=dst["project"],
    )

    click.echo("\n" + "=" * 50)
    click.echo("Full migration complete!")
    click.echo(f"Artifacts: {art_result}")
    click.echo(f"Runs: {run_result}")
    if report_url:
        click.echo(f"Report: {report_url}")


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
