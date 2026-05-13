# Copy Leaderboard Tool

A tool to copy the W&B Nejumi Leaderboard to your own W&B environment (such as dedicated cloud and on-premise).

## Features

1. **Run Migration**: Migrate runs with specific tags (e.g., "leaderboard") from source to destination
2. **Artifact Migration**: Copy dataset artifacts to your environment
3. **Report Creation**: Create a new leaderboard report using W&B Reports API
4. **Incremental Updates**: Sync only new runs since the last update

## Installation

### Using uv (recommended)

```bash
cd scripts/copy_leaderboard
uv sync
```

### Using pip

```bash
cd scripts/copy_leaderboard
pip install -e .
```

## Configuration

Copy and edit the configuration file:

```bash
cp config.yaml my_config.yaml
```

Edit `my_config.yaml` with your settings:

```yaml
source:
  base_url: "https://api.wandb.ai"
  entity: "llm-leaderboard"
  project: "nejumi-leaderboard4"
  run_tag: "leaderboard"

destination:
  base_url: "https://api.wandb.ai"  # or your W&B server
  entity: "your-entity"
  project: "your-project"

artifacts:
  - "wandb-japan/llm-leaderboard3/jaster:v6"
  # ... add more artifacts

options:
  max_workers: 4
  skip_existing: true
```

### Private Artifacts Access

Some artifacts (such as toxicity datasets) require special access permissions:

```yaml
artifacts:
  - "wandb-japan/toxicity-dataset-private/toxicity_dataset_full:v3"
  - "wandb-japan/toxicity-dataset-private/toxicity_judge_prompts:v1"
```

To access these private artifacts:

- **Enterprise license holders**: Please contact your designated W&B support engineer who will grant temporary access.
- **Other users**: Please contact support-jp@wandb.com.

*Note: We may not be able to accommodate all access requests. Thank you for your understanding.*

## Usage

### Set API Keys

```bash
# Source W&B API key (for reading from Nejumi leaderboard)
export WANDB_SRC_API_KEY="your-source-api-key"

# Destination W&B API key (for writing to your environment)
export WANDB_DST_API_KEY="your-destination-api-key"
```

### Full Migration

Run complete migration (artifacts, runs, and report creation). Use this for the initial one-time migration:

```bash
uv run python copy_leaderboard.py -c my_config.yaml full-migration
```

### Continuous Migration

For ongoing synchronization after the initial migration, use the sync command. This migrates only new runs since the last update.

#### Using start_date

Set `start_date` in your config to specify the starting point for incremental sync:

```yaml
options:
  start_date: "2024-01-01"  # ISO format: YYYY-MM-DD
  skip_existing: true
```

#### Sync Command

```bash
uv run python copy_leaderboard.py -c my_config.yaml sync
```

The sync command will:
- Only migrate runs created after `start_date` (if specified)
- Skip runs that already exist in the destination (when `skip_existing: true`)
- Automatically track the last sync time for subsequent runs

### Individual Commands

#### Migrate Runs Only

```bash
uv run python copy_leaderboard.py -c my_config.yaml migrate-runs
```

With options:
```bash
uv run python copy_leaderboard.py -c my_config.yaml migrate-runs --tag leaderboard --limit 10
```

#### Migrate Artifacts Only

```bash
uv run python copy_leaderboard.py -c my_config.yaml migrate-artifacts
```

#### Create Report

```bash
uv run python copy_leaderboard.py -c my_config.yaml create-report --title "My Leaderboard"
```

**Note**: This command creates a basic report with a `WeavePanelSummaryTable` that displays `runs.summary['leaderboard_table']`. The generated report is intentionally simple and serves as a starting point.

If you want to customize the report (e.g., add additional panels, adjust column widths, add filters, or include custom visualizations), please edit the report manually through the W&B UI after creation.

## Architecture

```
copy_leaderboard/
├── pyproject.toml          # Package and dependencies
├── config.yaml             # Default configuration
├── copy_leaderboard.py     # Main script
└── README.md               # This file
```

## API Reference

### LeaderboardMigrator

Main class for handling migrations.

```python
from copy_leaderboard import LeaderboardMigrator

migrator = LeaderboardMigrator(
    src_base_url="https://api.wandb.ai",
    src_api_key="your-src-key",
    dst_base_url="https://api.wandb.ai",
    dst_api_key="your-dst-key",
)

# Migrate runs
result = migrator.migrate_runs(
    entity="llm-leaderboard",
    project="nejumi-leaderboard4",
    dst_entity="your-entity",
    dst_project="your-project",
    tag="leaderboard",
)
```

### Config

Configuration loader.

```python
from copy_leaderboard import Config

config = Config("config.yaml")
print(config.source)       # Source settings
print(config.destination)  # Destination settings
print(config.artifacts)    # Artifact paths to migrate
```

## Troubleshooting

### Common Issues

1. **Authentication Error**: Ensure API keys are set correctly
   ```bash
   export WANDB_SRC_API_KEY="..."
   export WANDB_DST_API_KEY="..."
   ```

2. **Permission Denied**: Verify you have access to both source and destination projects

3. **Artifact Not Found**: Check artifact paths in config match existing artifacts

### Debug Mode

Enable verbose logging:

```bash
WANDB_DEBUG=true uv run python copy_leaderboard.py -c config.yaml migrate-runs
```

## License

Same as the parent llm-leaderboard project.
