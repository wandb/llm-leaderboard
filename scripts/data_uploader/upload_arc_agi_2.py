import json
from tempfile import TemporaryDirectory
import wandb
from argparse import ArgumentParser
from wandb import Table
from wandb.apis.public.runs import Run   
import pandas as pd
import os

parser = ArgumentParser()
parser.add_argument(
    "-e",
    "--entity",
    type=str,
    required=True
)
parser.add_argument(
    "-p",
    "--project",
    type=str,
    required=True
)

parser.add_argument(
    "-v",
    "--dataset_version",
    type=str,
    required=True
)
parser.add_argument(
    "-f",
    "--file_path",
    type=str,
    required=True
)
parser.add_argument(
    "-r",
    "--sampling_reference_run",
    type=str,
    default=None,
)
parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=42,
)
parser.add_argument(
    "-n",
    "--num_samples",
    type=int,
    default=50,
)
args = parser.parse_args()

with wandb.init(entity=args.entity, project=args.project, job_type="upload_data") as run:
    dataset_artifact = wandb.Artifact(name=f"arc-agi-2_public-eval_{args.num_samples}",
                                    type="dataset",
                                    metadata={"version":args.dataset_version},
                                    description="This dataset is based on version {}".format(args.dataset_version))

    if args.sampling_reference_run is not None:
        artifacts = wandb.Api().run(args.sampling_reference_run).logged_artifacts()
        for artifact in artifacts:
            if '-arc_agi_2_output_table:' in artifact.name:
                artifact_filepath = artifact.get_entry("arc_agi_2_output_table.table.json").download()
                break
        else:
            raise ValueError(f"No arc_agi_2_output_table artifact found in {args.sampling_reference_run}")

        with open(artifact_filepath) as f:
            df = Table.from_json(json.load(f), source_artifact=artifact)
        df = df.get_dataframe()
        # 全体のスコアを集計
        score_df = df.groupby(['task_id', 'test_example_id']).agg({'correct': 'max'}) # num_attemptsの中で最大値を取る
        score_df = score_df.groupby('task_id').agg({'correct': 'mean'}) # task id単位で平均
        total_score = score_df['correct'].mean()
        # 同じTaskが重複しないようにサンプリング
        correct_df = df[df.task_id.isin((score_df.correct > 0).index)] # task単位で正解があるものを抽出
        correct_df = correct_df[correct_df.correct == True].drop_duplicates(subset=['task_id', 'test_example_id']) # attemptで重複正解しているものを除外
        correct_df = correct_df.sample(frac=1, random_state=args.seed).drop_duplicates(subset=['task_id']) # 同じtaskに複数example正解しているものを除外
        incorrect_df = df[df.task_id.isin((score_df.correct == 0).index)].drop_duplicates(subset=['task_id', 'test_example_id'])
        incorrect_df = incorrect_df.sample(frac=1, random_state=args.seed).drop_duplicates(subset=['task_id'])
        # トータルスコアと同じになるようにサンプリングする
        # オリジナルのデータはTask単位で平均後に全Taskの平均を取るので、Taskごとのスコアの重みが異なる
        num_correct = round(args.num_samples * total_score)
        num_incorrect = args.num_samples - num_correct
        print(f"correct_sampling_rate: {total_score}, num_correct: {num_correct}, num_incorrect: {num_incorrect}")
        correct_df = correct_df.sample(n=num_correct, random_state=args.seed)
        incorrect_df = incorrect_df.sample(n=num_incorrect, random_state=args.seed)
        df = pd.concat([correct_df, incorrect_df])

        with TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "evaluation.txt"), "w") as id_file:
                id_file.write("\n".join(df.task_id.values.tolist()))
            os.makedirs(os.path.join(tmp_dir, "evaluation"))
            for task_id, test_example_id in zip(df.task_id.values, df.test_example_id.values):
                with open(os.path.join(args.file_path, "evaluation", f"{task_id}.json"), "r") as task_org_file:
                    task_def = json.load(task_org_file)
                task_def['test'] = [task_def['test'][test_example_id]]
                with open(os.path.join(tmp_dir, "evaluation", f"{task_id}.json"), "w") as task_write_file:
                    json.dump(task_def, task_write_file, indent=2)
            dataset_artifact.add_dir(tmp_dir)

    else:
        # 全件アップロード
        dataset_artifact.add_file(os.path.join(args.file_path, "evaluation.txt"))
        dataset_artifact.add_dir(os.path.join(args.file_path, "evaluation"), name="evaluation")
    run.log_artifact(dataset_artifact)
