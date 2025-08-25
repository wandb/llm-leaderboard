import json
from pathlib import Path
from typing import List, Optional
from glob import glob
import os

import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
import wandb
import weave
from matplotlib.colors import TABLEAU_COLORS

from config_singleton import WandbConfigSingleton
from .evaluate_utils import LLMAsyncProcessor


# ARC-AGI-2では0-9の値が使用される
COLORMAP = np.array([
    [int(rgb[1:3], 16), int(rgb[3:5], 16), int(rgb[5:7], 16)]
    for rgb in TABLEAU_COLORS.values()
] + [[255, 255, 255], [255, 0, 0]], dtype=np.uint8)  # 白色 + 赤色（エラー用）を追加


PROMPT_TEMPLATE = """\
あなたはパズルを解くコンテストに参加しています。あなたはパズルを解く専門家です。
以下は、入力と出力のペアのリストです。あなたの目標は、トレーニング例における入力から出力へのマッピングを行うためのパターンや変換を特定し、そのパターンをテスト入力に適用して最終的な出力を生成することです。

以下のトレーニング出力例のフォーマットで出力してください。

--トレーニング例--
{training_examples}
--トレーニング例ここまで--

--テスト入力--
{test_input}
--テスト入力ここまで--
"""


def pretty_print_json(tile: List[List[int]]) -> str:
    return json.dumps(tile).replace("[[", "[\n  [").replace("]]", "]\n]").replace("], [", "],\n  [").replace(", ", ",")


def pretty_print_tile( tile: List[List[int]]) -> str:
    return "\n".join((" ".join(str(e) for e in line) for line in tile))


def convert_task_pairs_to_prompt(training_pairs: List[dict], test_input: dict) -> List[dict]:
    """
    Convert the training pairs to a prompt
    Citation: https://github.com/arcprize/arc-agi-benchmarking/blob/main/src/arc_agi_benchmarking/prompts/prompt_manager.py#L18-L34

    Parameters:
        training_pairs (List[dict]): A list of training pairs.
        test_input (dict): The test input.

    Returns:
        List[dict]: The chatprompt.
    """

    training_examples = ""
    for i, pair in enumerate(training_pairs):
        training_examples += f"--例 {i}-- \n\n 入力: \n\n"
        training_examples += pretty_print_json(pair['input']) + "\n\n"
        training_examples += f"出力: \n\n"
        training_examples += pretty_print_json(pair['output']) + "\n\n"

    test_input_str = pretty_print_json(test_input['input'])

    return [
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(training_examples=training_examples, test_input=test_input_str)
        }
    ]


def backscan_json_parser(log_str: str) -> Optional[List[List[int]]]:
    """
    Extract the last valid JSON substring that matches the List[List] structure
    from the given log string by scanning backwards from the end.
    Citation: https://github.com/arcprize/arc-agi-benchmarking/blob/main/src/arc_agi_benchmarking/utils/parsing.py#L9-L62

    Parameters:
        log_str (str): The full log output text.

    Returns:
        The parsed List[List] object if found and valid, otherwise None.
    """
    last_bracket_idx = -1
    closing_bracket = None
    for i in range(len(log_str) - 1, -1, -1):
        char = log_str[i]
        if char in (']', '}'):
            last_bracket_idx = i
            closing_bracket = char
            break

    if last_bracket_idx == -1:
        return None

    opening_bracket = '[' if closing_bracket == ']' else '{'

    bracket_counter = 1 # Start at 1 to account for the found closing bracket
    start_idx = -1

    for i in range(last_bracket_idx - 1, -1, -1):
        char = log_str[i]
        if char == closing_bracket:
            bracket_counter += 1
        elif char == opening_bracket:
            bracket_counter -= 1
            if bracket_counter == 0:
                start_idx = i
                break

    if start_idx == -1:
        return None

    json_candidate = log_str[start_idx:last_bracket_idx+1]

    try:
        parsed_json = json.loads(json_candidate)

        # Validate the structure: must be a non-empty list of lists.
        if isinstance(parsed_json, list) and parsed_json and all(isinstance(row, list) for row in parsed_json):
            return parsed_json
        else:
            return None

    except json.JSONDecodeError:
        return None


def pad_ragged_tile(tile: List[List[int]]) -> List[List[int]]:
    """
    Pad the ragged tile to the maximum length of the rows for visualization
    1 2 3     1 2 3
    4 5    => 4 5 -1
    6 7 8     6 7 8

    Args:
        tile (List[List[int]]): The tile possibly ragged.

    Returns:
        List[List[int]]: The padded tile.
    """
    max_length = max(len(row) for row in tile)
    return [row + [-1] * (max_length - len(row)) for row in tile]


def tile_to_img(expected: List[List[int]], output: Optional[List[List[int]]]) -> np.ndarray:
    """
    Visualize the tile as a wandb.Image with bounding boxes.

    Args:
        expected (List[List[int]]): Ground truth tile
        output (Optional[List[List[int]]]): Predicted tile

    Returns:
        wandb.Image: The visualized tile with bounding boxes
    """
    expected_arr = np.array(pad_ragged_tile(expected), np.int8)
    if output is None:
        output_arr = np.full_like(expected_arr, -1)
    else:
        output_arr = np.array(pad_ragged_tile(output), np.int8)

    expected_padded = np.full((max(expected_arr.shape[0], output_arr.shape[0]), max(expected_arr.shape[1], output_arr.shape[1])), -1, dtype=np.int8)
    output_padded = np.full((max(expected_arr.shape[0], output_arr.shape[0]), max(expected_arr.shape[1], output_arr.shape[1])), -1, dtype=np.int8)
    expected_padded[:expected_arr.shape[0], :expected_arr.shape[1]] = expected_arr
    output_padded[:output_arr.shape[0], :output_arr.shape[1]] = output_arr

    boundary = np.full((expected_padded.shape[0], 1), -1, dtype=np.int8)

    concat_map = np.concatenate([expected_padded, boundary, output_padded], axis=1).repeat(16, axis=0).repeat(16, axis=1)

    bboxes = {
        "predictions": {
            "box_data": [],
            "class_labels": {0: "expected", 1: "output"},
        }
    }
    if output is not None:
        fail_indices = np.where(expected_padded != output_padded)
        for x, y in zip(fail_indices[1].tolist(), fail_indices[0].tolist()):
            if expected_padded[y, x] == -1 or output_padded[y, x] == -1:
                continue

            caption = f"Expected: {expected_padded[y, x].item()}, Output: {output_padded[y, x].item()}"
            bboxes["predictions"]["box_data"].append({
                "position":{
                    "minX": x * 16,
                    "minY": y * 16,
                    "maxX": (x + 1) * 16 - 1,
                    "maxY": (y + 1) * 16 - 1,
                },
                "domain": "pixel",
                "class_id": 0,
                "box_caption": caption,
            })
            offset = (expected_padded.shape[1] + 1) * 16
            bboxes["predictions"]["box_data"].append({
                "position":{
                    "minX": offset + x * 16,
                    "minY": y * 16,
                    "maxX": offset + (x + 1) * 16 - 1,
                    "maxY": (y + 1) * 16 - 1,
                },
                "domain": "pixel",
                "class_id": 1,
                "box_caption": caption,
            })

    # 無効な値（10以上）は赤色で表示
    concat_map_safe = np.clip(concat_map, 0, len(COLORMAP) - 1)
    pil_image = Image.fromarray(COLORMAP[concat_map_safe])
    return wandb.Image(pil_image, boxes=bboxes)


@weave.op(call_display_name=lambda _: "[ARC-AGI] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
def evaluate():
    # Retrieve the instance from WandbConfigSingleton and load the W&B run and configuration
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    # download dataset
    dataset_name = "arc_agi"
    v1_artifact = run.use_artifact(cfg[dataset_name].arc_agi_1_artifacts_path, type="dataset")
    v1_artifact_dir = v1_artifact.download()
    v2_artifact = run.use_artifact(cfg[dataset_name].arc_agi_2_artifacts_path, type="dataset")
    v2_artifact_dir = v2_artifact.download()

    total_scores = []
    for arc_version, artifact_dir in enumerate([v1_artifact_dir, v2_artifact_dir], start=1):
        dataset_dir = Path(artifact_dir)
        if not dataset_dir.exists():
            print(f"skip {dataset_name} because it is not found in {artifact_dir}")
            raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

        # load tasks
        task_files = glob(os.path.join(dataset_dir, "evaluation", "*.json"))
        if cfg.testmode:
            task_files = task_files[:5]

        # create inference inputs
        tasks = []
        all_inputs = []
        for task_file in task_files:
            task_id = os.path.basename(task_file).split('.')[0]
            with open(task_file, 'r') as f:
                task = {'id': task_id, **json.load(f)}
                for test_example_id, test_example in enumerate(task['test']):
                    prompt = convert_task_pairs_to_prompt(task['train'], test_example)
                    for num_attempts in range(cfg[dataset_name].num_attempts):
                        all_inputs.append((prompt, {"max_tokens": cfg[dataset_name].max_output_tokens}))
                        tasks.append({
                            'id': task_id,
                            'test_example_id': test_example_id,
                            'num_attempts': num_attempts,
                            'input': test_example['input'],
                            'output': test_example['output'],
                            'prompt': prompt,
                        })
                    if cfg.testmode: # test modeの場合1task1問のみ
                        break

        # Run inference in parallel
        # YAMLで切替可能: cfg.arc_agi.error_handling.request_failure.mode: soft|hard
        # 互換: cfg.arc_agi.soft_fail_on_error: true|false
        # デフォルト（ARC-AGIのみ）: soft
        soft_fail = True
        try:
            mode = getattr(cfg.arc_agi, "error_handling").get("request_failure", {}).get("mode", None)
            if mode is not None:
                soft_fail = str(mode).lower() == "soft"
        except Exception:
            pass
        try:
            # boolean が直接指定されていれば優先
            sfoe = cfg.arc_agi.get("soft_fail_on_error", None)
            if sfoe is not None:
                soft_fail = bool(sfoe)
        except Exception:
            pass

        llm_ap = LLMAsyncProcessor(llm=llm, inputs=all_inputs, soft_fail_on_error=soft_fail)
        results = llm_ap.get_results()

        # Evaluation
        evaluation_results = []
        for response, task in tqdm(zip(results, tasks), total=len(results), desc=f"Evaluating ARC-AGI-{arc_version}"):
            raw_output = response.content
            y_pred = backscan_json_parser(raw_output)

            if y_pred is not None:
                expected_arr = np.array(task['output'], np.int8)
                try:
                    # まず、y_predが正しい2次元リスト構造かチェック
                    if not isinstance(y_pred, list) or not all(isinstance(row, list) for row in y_pred):
                        correct = False
                    elif len(y_pred) == 0 or any(len(row) == 0 for row in y_pred):
                        correct = False
                    else:
                        # 各行の長さが同じかチェック（矩形配列であることを確認）
                        row_lengths = [len(row) for row in y_pred]
                        if len(set(row_lengths)) != 1:
                            correct = False
                        else:
                            y_pred_arr = np.array(y_pred, np.int8)
                            # ARC-AGI-2の有効な値は0-9のみ
                            if np.any(y_pred_arr < 0) or np.any(y_pred_arr > 9):
                                correct = False
                            else:
                                correct = expected_arr.shape == y_pred_arr.shape and np.all(y_pred_arr == expected_arr) # 完全一致の場合のみ正解
                except (ValueError, TypeError) as e:
                    correct = False
            else:
                correct = False

            try:
                evaluation_results.append({
                    "model_name": cfg.model.pretrained_model_name_or_path,
                    "task": f"arc-agi-{arc_version}",
                    "task_id": task['id'],
                    "test_example_id": task['test_example_id'],
                    "num_attempts": task['num_attempts'],
                    "prompt": task['prompt'][0]['content'],
                    "raw_output": raw_output,
                    "input": pretty_print_tile(task['input']),
                    "reasoning_content": response.reasoning_content,
                    "output": pretty_print_tile(y_pred) if y_pred is not None else None,
                    "expected_output": pretty_print_tile(task['output']),
                    "correct": correct,
                    "visualize": tile_to_img(task['output'], y_pred),
                })
            except Exception as e:
                print(f"Error visualizing tile for task {task['id']}, example {task['test_example_id']}: {e}")
                evaluation_results.append({
                    "model_name": cfg.model.pretrained_model_name_or_path,
                    "task": f"arc-agi-{arc_version}",
                    "task_id": task['id'],
                    "test_example_id": task['test_example_id'],
                    "num_attempts": task['num_attempts'],
                    "prompt": task['prompt'][0]['content'],
                    "raw_output": raw_output,
                    "input": pretty_print_tile(task['input']),
                    "reasoning_content": response.reasoning_content,
                    "output": pretty_print_tile(y_pred) if y_pred is not None else None,
                    "expected_output": pretty_print_tile(task['output']),
                    "correct": correct,
                    "visualize": None,
                })

        output_df = pd.DataFrame(evaluation_results)
        score_df = output_df.groupby(['task_id', 'test_example_id']).agg({'correct': 'max'}) # num_attemptsの中で最大値を取る
        score_df = score_df.groupby('task_id').agg({'correct': 'mean'}) # task id単位で平均
        total_score = score_df['correct'].mean()

        run.log({f"{dataset_name}_{arc_version}_output_table": output_df})
        total_scores.append(total_score)

    leaderboard_table = pd.DataFrame([{
        "model_name": cfg.model.pretrained_model_name_or_path,
        "arc-agi-1": total_scores[0],
        "arc-agi-2": total_scores[1],
        "AVG": (total_scores[0] + total_scores[1]) / 2,
    }])

    run.log({f"{dataset_name}_leaderboard_table": leaderboard_table})

