# Blender

This feature allows you to blend inference results later or resume a run by carrying over the previous results.

## Overview

The Blender feature provides two main functionalities:

1. **Blend Run**: Merges the results of multiple past runs to analyze the aggregated output.
2. **Resume Run**: Continues a previous run from where it left off, carrying over the results and configurations.

These features are particularly useful in scenarios where you need to combine results from different experiments or when an experiment is interrupted and needs to be resumed without losing the previous progress.

## Blend Run

### Setting up blend_config

1. Copy the template and configure it properly:
   ```bash
   cp -ip blend_run_configs/config_template.yaml blend_run_configs/blend_config.yaml
   ```

2. The blend_config.yaml requires the following settings:
   - `num_few_shots`: Number of few-shot examples to use.
   - `model`: Information about the model.
   - `new_run`: Information for the new run.
   - `old_runs`: Specify the tasks you want to carry over from past runs.

For detailed information on each setting, please refer to the blender Configuration Settings

### Running the Blend Script

3. Run the blend script:
   ```bash
   python3 scripts/blend_run.py
   ```

### Checking the W&B Dashboard

4. The results of the evaluation will be logged to the specified W&B project.

## Resume Run

### Setting up blend_config

1. Copy the template and configure it properly:
   ```bash
   cp -ip blend_run_configs/config_template.yaml blend_run_configs/blend_config.yaml
   ```

2. The blend_config.yaml requires the following settings:
   - `run_chain`: Set to `true`.
   - `old_runs`: Specify the tasks you want to carry over from past runs.

For detailed information on each setting, please refer to the blender-configuration-settings.

### Running the Evaluation Script

3. You can use either the `-c` or `-s` option:
    - **-c (config)**: Specify the config file by its name, e.g.,
      ```bash
      python3 scripts/run_eval.py -c config-gpt-4o-2024-05-13.yaml
      ```
    - **-s (select-config)**: Select from a list of available config files. This option is useful if you have multiple config files.
      ```bash
      python3 scripts/run_eval.py -s
      ```

### Checking the W&B Dashboard

4. The results of the evaluation will be logged to the specified W&B project.

## Blender Configuration Settings

- **run_chain**: Set to `false` to use the blend feature and `true` to use the resume feature.
- **num_few_shots**: Number of few-shot examples to use.
- **model**: Information about the model. (No configuration is required to use the resume feature.)
    - `use_wandb_artifacts`: Set to `true` if you want to use W&B artifacts.
    - `pretrained_model_name_or_path`: Name of the VLLM model.
    - `chat_template`: Path to the chat template file (if needed).
    - `size_category`: Specify "api" to indicate using an API model.
    - `size`: Model size (parameter).
    - `release_date`: Model release date (MM/DD/YYYY).
    - `max_model_len`: Maximum token length of the input (if needed).
- **new_run**: Information used for Weights & Biases (W&B) support. (No configuration is required to use the resume feature.)
    - `entity`: Name of the W&B Entity.
    - `project`: Name of the W&B Project.
    - `run_name`: Name of the W&B run. Please set up the run name in a model-specific config.
- **old_run**: Specify the tasks you want to carry over from past runs. Multiple runs are permissible.
    - `run_path`: Run path of the W&B old_run.
    - `dataset`: The list of tasks to take over. Please comment out tasks that do not need to be taken over.