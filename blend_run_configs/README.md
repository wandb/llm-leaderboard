# Blender

This feature allows you to blend inference results later or resume a run by carrying over the previous results.

## blender Configuration Settings

- **run_chain:** Please select false to use the blend feature and true to use the resume feature.

- **num_few_shots:**  Number of few-shot examples to use.

- **model:** Information about the model. (No configuration is required to use the resume feature.)
    - `use_wandb_artifacts`: Set to true if you want to use wandb artifacts.
    - `pretrained_model_name_or_path`: Name of the VLLM model.
    - `chat_template`: Path to the chat template file (if needed).
    - `size_category`: Specify "api" to indicate using an API model.
    - `size`: Model size (parameter).
    - `release_date`: Model release date (MM/DD/YYYY).
    - `max_model_len`: Maximum token length of the input (if needed).

- **new_run:** Information used for Weights & Biases (W&B) support. (No configuration is required to use the resume feature.)
    - `entity`: Name of the W&B Entity.
    - `project`: Name of the W&B Project.
    - `run_name`: Name of the W&B run. Please set up run name in a model-specific config.

- **old_run:** Please specify the tasks you want to carry over from past runs. Multiple runs are permissible.
    -`run_path`: run path of the W&B old_run. 
     `dataset`: The list of tasks to take over. Please comment out tasks that do not need to be taken over.

### blend run

1. **Setting blend_config:**
Copy the template and configure it properly
   ```bash
   cp -ip blend_run_configs/config_template.yaml blend_run_configs/blend_config.yaml
   ```

2. **Run the blend script:**

   ```bash
   python3 scripts/blend_run.py
   ```

3. **Check the W&B dashboard:** The results of the evaluation will be logged to the specified W&B project.

### resume run

1. **Setting blend_config:**
Copy the template and configure it properly
   ```bash
   cp -ip blend_run_configs/config_template.yaml blend_run_configs/blend_config.yaml
   ```

2. **Run the evaluation script:**

    You can use either `-c` or `-s` option:
    - **-c (config):** Specify the config file by its name, e.g., `python3 scripts/run_eval.py -c config-gpt-4o-2024-05-13.yaml`
    - **-s (select-config):** Select from a list of available config files. This option is useful if you have multiple config files. 
   ```bash
   python3 scripts/run_eval.py -s
   ```

3. **Check the W&B dashboard:** The results of the evaluation will be logged to the specified W&B project.