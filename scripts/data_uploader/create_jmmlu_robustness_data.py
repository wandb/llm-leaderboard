#!/usr/bin/env python3
"""
JMMluデータから2つの派生データセットを作成するスクリプト

Task1: jmmlu_IncorrectChoice.json - 不正解を選択するタスク
Task2: jmmlu_SymbolChoice.json - 選択肢記号を変更したタスク
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any


def load_json_file(file_path: str) -> Dict[str, Any]:
    """JSONファイルを読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """JSONファイルを保存する"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def get_incorrect_choices(correct_answer: str, all_choices: List[str]) -> str:
    """正解以外の選択肢をカンマ区切りで返す"""
    incorrect = [choice for choice in all_choices if choice != correct_answer]
    return ','.join(incorrect)


def replace_choices_in_text(text: str, choice_mapping: Dict[str, str]) -> str:
    """テキスト内の選択肢記号を置換する"""
    for old_choice, new_choice in choice_mapping.items():
        # "選択肢：A." -> "選択肢：$."
        text = text.replace(f"選択肢：{old_choice}.", f"選択肢：{new_choice}.")
        # ",B." -> ",&."
        text = text.replace(f",{old_choice}.", f",{new_choice}.")
    return text


def create_incorrect_choice_data(original_data: Dict[str, Any]) -> Dict[str, Any]:
    """Task1: 不正解選択タスクのデータを作成"""
    new_data = original_data.copy()
    
    # instructionを変更
    new_data["instruction"] = ("与えられた質問と選択肢から、不正解となる回答を全て選択してください。"
                              "なお、回答には選択肢の記号（例：A）のみを含め、回答となるアルファベットは"
                              "カンマ区切りで出力してください。他には何も含めないことを厳守してください。")
    
    # 各サンプルのoutputを変更
    all_choices = original_data["label_list"]
    new_samples = []
    
    for sample in original_data["samples"]:
        new_sample = sample.copy()
        correct_answer = sample["output"]
        new_sample["output"] = get_incorrect_choices(correct_answer, all_choices)
        new_samples.append(new_sample)
    
    new_data["samples"] = new_samples
    return new_data


def create_symbol_choice_data(original_data: Dict[str, Any]) -> Dict[str, Any]:
    """Task2: 記号選択タスクのデータを作成"""
    new_data = original_data.copy()
    
    # A, B, C, D を $, &, #, @ に対応させる
    choice_mapping = {"A": "$", "B": "&", "C": "#", "D": "@"}
    
    # instructionを変更
    new_data["instruction"] = ("与えられた質問と選択肢から、最も適切な回答を選択してください。"
                              "なお、回答には選択肢の記号（例：$）のみを含め、"
                              "他には何も含めないことを厳守してください。")
    
    # label_listを変更
    new_data["label_list"] = ["$", "&", "#", "@"]
    
    # 各サンプルを変更
    new_samples = []
    
    for sample in original_data["samples"]:
        new_sample = sample.copy()
        
        # inputテキスト内の選択肢記号を置換
        new_input = replace_choices_in_text(sample["input"], choice_mapping)
        new_sample["input"] = new_input
        
        # outputの選択肢記号を置換
        old_output = sample["output"]
        new_sample["output"] = choice_mapping.get(old_output, old_output)
        
        new_samples.append(new_sample)
    
    new_data["samples"] = new_samples
    return new_data


def process_folder(folder_path: str) -> None:
    """指定されたフォルダ内のjmmlu.jsonを処理する"""
    jmmlu_path = os.path.join(folder_path, "jmmlu.json")
    
    if not os.path.exists(jmmlu_path):
        print(f"jmmlu.json not found in {folder_path}")
        return
    
    print(f"Processing {jmmlu_path}...")
    
    # 元データを読み込み
    original_data = load_json_file(jmmlu_path)
    
    # Task1: IncorrectChoice版を作成
    incorrect_choice_data = create_incorrect_choice_data(original_data)
    incorrect_choice_path = os.path.join(folder_path, "jmmlu_IncorrectChoice.json")
    save_json_file(incorrect_choice_data, incorrect_choice_path)
    print(f"Created {incorrect_choice_path}")
    
    # Task2: SymbolChoice版を作成
    symbol_choice_data = create_symbol_choice_data(original_data)
    symbol_choice_path = os.path.join(folder_path, "jmmlu_SymbolChoice.json")
    save_json_file(symbol_choice_data, symbol_choice_path)
    print(f"Created {symbol_choice_path}")


def main():
    """メイン処理"""
    # データのパス
    data_root = "jaster"
    folders = ["train", "dev", "test"]
    
    for folder in folders:
        folder_path = os.path.join(data_root, folder)
        if os.path.exists(folder_path):
            process_folder(folder_path)
        else:
            print(f"Folder not found: {folder_path}")


if __name__ == "__main__":
    main()
