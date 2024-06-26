
# JBBQ Dataset Preprocessing

This document provides a detailed explanation of the preprocessing steps required for the JBBQ dataset. The preprocessing script processes the raw data and prepares it for evaluation and tuning.

## Preprocessing Steps

1. **Download the JBBQ Dataset**
    - Obtain the JBBQ dataset from the [official repository](https://github.com/ynklab/JBBQ_data).
    - Ensure compliance with the handling instructions listed in the official repository.

2. **Run the Preprocessing Script**
    - Use the provided `upload_jbbq.py` script to preprocess the JBBQ dataset and upload it to WandB.
    - The script will:
        - Parse the raw data files.
        - Extract relevant information and transform it into the required format.
        - Split the data into training, development, and test sets.
        - Save the processed data and upload it to WandB.

### Command to Run the Script

```bash
# Preprocess the JBBQ dataset and upload it to WandB
  python3 scripts/uploader/upload_jbbq.py -d <jbbq dataset path> -e <wandb/entity> -p <wandb/project> -n <dataset name> -v <dataset version>
```

Replace the placeholders with your specific information:
- `<jbbq_dataset_path>`: Path to the directory containing the JBBQ dataset.
- `<wandb_entity>`: Your WandB entity name.
- `<wandb_project>`: Your WandB project name.
- `<dataset_name>`: The name you want to give to the dataset in WandB.
- `<dataset_version>`: The version of the dataset.

## Detailed Preprocessing Explanation
The preprocessing script performs the following steps:

- Identify and define the unknown option from the choices as `unk_label`.
- Identify and define the stereotypical bias option from the choices as `stereotype_label`.
- If the `question_polarity` is `nonneg`, set the `stereotype_label` to the other choice that is not `unknown`.
- Split the data into training, development, and test sets with the ratio 3:3:10 (split by `question_index`).