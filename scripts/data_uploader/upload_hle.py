import csv
import json
import random
import os
import wandb
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser

def create_stratified_split_data(project_root):
    """
    Reads hle-ja.csv, performs a stratified split by category (8:1:1),
    and saves the resulting train, dev, and test sets as JSONL files.
    Returns the path to the output directory.
    """
    input_csv_path = project_root / 'hle-ja.csv'
    output_dir = project_root / 'data' / 'hle-ja'

    samples_by_category = defaultdict(list)
    with open(input_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples_by_category[row['category']].append(row)

    train_samples, dev_samples, test_samples = [], [], []
    random.seed(42)

    for category, samples in samples_by_category.items():
        random.shuffle(samples)
        n_samples = len(samples)
        train_end = int(n_samples * 0.8)
        dev_end = int(n_samples * 0.9)
        train_samples.extend(samples[:train_end])
        dev_samples.extend(samples[train_end:dev_end])
        test_samples.extend(samples[dev_end:])

    random.shuffle(train_samples)
    random.shuffle(dev_samples)
    random.shuffle(test_samples)

    output_dir.mkdir(parents=True, exist_ok=True)

    def _save_as_jsonl(data, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    _save_as_jsonl(train_samples, output_dir / "train.jsonl")
    _save_as_jsonl(dev_samples, output_dir / "dev.jsonl")
    _save_as_jsonl(test_samples, output_dir / "test.jsonl")

    print(f"Data successfully split (stratified by category) and saved in {output_dir}")
    print(f"Train: {len(train_samples)}, Dev: {len(dev_samples)}, Test: {len(test_samples)}")
    return str(output_dir)

def upload_to_wandb(entity, project, dataset_name, dataset_folder, dataset_version):
    """Uploads the dataset to Weights & Biases as an artifact."""
    with wandb.init(entity=entity, project=project, job_type="upload_data") as run:
        dataset_artifact = wandb.Artifact(
            name=dataset_name,
            type="dataset",
            metadata={"version": dataset_version}
        )
        dataset_artifact.add_dir(dataset_folder, name=dataset_name)
        run.log_artifact(dataset_artifact)
        print(f"Successfully uploaded {dataset_name} to WandB.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--entity", type=str, required=True, help="WandB entity")
    parser.add_argument("-p", "--project", type=str, required=True, help="WandB project")
    parser.add_argument("-n", "--dataset_name", type=str, default="hle-ja", help="Name of the dataset artifact")
    parser.add_argument("-v", "--dataset_version", type=str, default="v1", help="Version of the dataset")
    args = parser.parse_args()

    # Assumes the script is in scripts/data_uploader/
    project_root = Path(__file__).resolve().parents[2]
    
    # 1. Create the split data
    output_folder = create_stratified_split_data(project_root)
    
    # 2. Upload to WandB
    upload_to_wandb(args.entity, args.project, args.dataset_name, output_folder, args.dataset_version)
