#!/usr/bin/env python3
"""
Materialize zh-TW IFEval-style prompts for Nejumi Taiwan.

The dataset is a Traditional Chinese adaptation focused on mechanically
verifiable instructions. It intentionally excludes Japanese-specific M-IFEval
constraints such as furigana, hiragana, katakana, and nominal endings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import wandb


DEFAULT_ARTIFACT_NAME = "ifeval-zh-tw"
DEFAULT_DATASET_DIR_NAME = "ifeval_zh_tw"


def kwargs(**values: Any) -> dict[str, Any]:
    return values


TASKS: list[dict[str, Any]] = [
    {
        "prompt": "請用繁體中文寫一段關於台灣夜市的介紹，並至少包含 2 個像 [地點] 這樣的方括號佔位符。",
        "instruction_id_list": ["zh_tw:detectable_content:number_placeholders"],
        "kwargs": [kwargs(num_placeholders=2)],
        "reference_response": "台灣夜市常見於[城市]的[地點]，集合了小吃與遊戲攤位。",
    },
    {
        "prompt": "請用繁體中文列出 3 個適合週末在台北進行的活動。答案必須剛好包含 3 個項目，且每個項目都要用 Markdown 項目符號開頭。",
        "instruction_id_list": ["zh_tw:detectable_format:number_bullet_lists"],
        "kwargs": [kwargs(num_bullets=3)],
        "reference_response": "- 逛書店\n- 看展覽\n- 騎河濱自行車",
    },
    {
        "prompt": "請用繁體中文列出準備會議的 4 個步驟。答案必須剛好包含 4 個編號項目，格式使用 1. 2. 3. 4.",
        "instruction_id_list": ["zh_tw:detectable_format:number_numbered_lists"],
        "kwargs": [kwargs(num_items=4)],
        "reference_response": "1. 確認議程\n2. 邀請與會者\n3. 準備資料\n4. 會後追蹤",
    },
    {
        "prompt": "請用繁體中文介紹一項新產品，並至少用《》標出 2 個重點片語。",
        "instruction_id_list": ["zh_tw:detectable_format:number_highlighted_sections"],
        "kwargs": [kwargs(num_highlights=2)],
        "reference_response": "這項產品主打《快速部署》與《低維護成本》，適合小型團隊。",
    },
    {
        "prompt": "請用繁體中文寫一份短報告，至少分成 3 個章節。每個章節標題必須以「第1節」、「第2節」這種格式開始。",
        "instruction_id_list": ["zh_tw:detectable_format:multiple_sections"],
        "kwargs": [kwargs(section_spliter="節", num_sections=3)],
        "reference_response": "第1節\n背景\n第2節\n分析\n第3節\n結論",
    },
    {
        "prompt": "請用繁體中文寫 2 個段落介紹高鐵旅遊，段落之間必須用 Markdown 分隔線 *** 隔開。",
        "instruction_id_list": ["zh_tw:length_constraints:number_paragraphs"],
        "kwargs": [kwargs(num_paragraphs=2)],
        "reference_response": "高鐵讓城市移動更有效率。\n***\n旅客可以安排一日往返的行程。",
    },
    {
        "prompt": "請用繁體中文提供一段客服回覆，最後必須加入以「附註：」開頭的補充說明。",
        "instruction_id_list": ["zh_tw:detectable_content:postscript"],
        "kwargs": [kwargs(postscript_marker="附註：")],
        "reference_response": "我們已收到您的需求，會盡快回覆。\n附註：請保留訂單編號。",
    },
    {
        "prompt": "請用繁體中文寫一段關於智慧城市的說明，必須包含「交通」、「能源」、「資料」三個詞。",
        "instruction_id_list": ["zh_tw:keywords:existence"],
        "kwargs": [kwargs(keywords=["交通", "能源", "資料"])],
        "reference_response": "智慧城市整合交通、能源與資料，讓公共服務更即時。",
    },
    {
        "prompt": "請用繁體中文寫一段健康提醒，詞語「睡眠」必須至少出現 3 次。",
        "instruction_id_list": ["zh_tw:keywords:frequency"],
        "kwargs": [kwargs(keyword="睡眠", frequency=3, relation="at least")],
        "reference_response": "睡眠很重要。良好的睡眠能提升專注力，穩定的睡眠也有助健康。",
    },
    {
        "prompt": "請用繁體中文寫一段餐廳推薦，但不得出現「便宜」這個詞。",
        "instruction_id_list": ["zh_tw:keywords:forbidden_words"],
        "kwargs": [kwargs(forbidden_words=["便宜"])],
        "reference_response": "這間餐廳份量充足，服務親切，適合朋友聚餐。",
    },
    {
        "prompt": "請用繁體中文寫一段旅遊建議，必須至少有 3 句。",
        "instruction_id_list": ["zh_tw:length_constraints:number_sentences"],
        "kwargs": [kwargs(num_sentences=3, relation="at least")],
        "reference_response": "先確認天氣。再安排交通。最後預留休息時間。",
    },
    {
        "prompt": "請用繁體中文回答「遠距工作有哪些優點？」答案中的中文字數必須至少 30 個。",
        "instruction_id_list": ["zh_tw:length_constraints:number_letters"],
        "kwargs": [kwargs(num_letters=30, relation="at least")],
        "reference_response": "遠距工作能節省通勤時間，也讓員工更容易安排專注時段與家庭生活，並降低日常移動壓力。",
    },
    {
        "prompt": "請只輸出一個合法 JSON 物件，內容是繁體中文，包含 name 和 city 兩個欄位。",
        "instruction_id_list": ["zh_tw:detectable_format:json_format"],
        "kwargs": [kwargs()],
        "reference_response": "{\"name\":\"小明\",\"city\":\"台北\"}",
    },
    {
        "prompt": "請只用以下三個選項之一回答：是、否、不確定。",
        "instruction_id_list": ["zh_tw:detectable_format:constrained_response"],
        "kwargs": [kwargs(options=["是", "否", "不確定"])],
        "reference_response": "不確定",
    },
    {
        "prompt": "請用繁體中文寫一段通知，最後必須以「以上，謝謝。」結尾，後面不得再加任何文字。",
        "instruction_id_list": ["zh_tw:startend:end_checker"],
        "kwargs": [kwargs(end_phrase="以上，謝謝。")],
        "reference_response": "明天會議改為線上舉行。以上，謝謝。",
    },
    {
        "prompt": "請用繁體中文寫一首短詩，必須包含一個用《》包住的標題。",
        "instruction_id_list": ["zh_tw:detectable_format:title"],
        "kwargs": [kwargs()],
        "reference_response": "《雨後》\n街燈映著水光，夜色慢慢放晴。",
    },
    {
        "prompt": "請用繁體中文寫一段不超過三句的提醒，全篇不得使用句號或英文句點。",
        "instruction_id_list": ["zh_tw:punctuation:no_period"],
        "kwargs": [kwargs()],
        "reference_response": "出門前記得帶傘\n也請確認手機電量",
    },
    {
        "prompt": "請用繁體中文寫一段活動宣傳，全篇不得使用逗號、頓號或英文 comma。",
        "instruction_id_list": ["zh_tw:punctuation:no_comma"],
        "kwargs": [kwargs()],
        "reference_response": "週末市集開放報名。歡迎家庭一起參加。",
    },
    {
        "prompt": "請用繁體中文回答一個鼓勵朋友的句子，整個回答必須用中文引號「」包住。",
        "instruction_id_list": ["zh_tw:startend:quotation"],
        "kwargs": [kwargs()],
        "reference_response": "「你已經很努力了，下一步會更穩。」",
    },
    {
        "prompt": "請用繁體中文回答，且整體內容必須主要是繁體中文，不得混入明顯簡體字。",
        "instruction_id_list": ["zh_tw:language:response_language"],
        "kwargs": [kwargs(language="zh_tw")],
        "reference_response": "這是一段使用繁體中文撰寫的自然回答，適合台灣使用者閱讀。",
    },
    {
        "prompt": "請用繁體中文回答兩段內容，第一段必須以「回覆一：」開頭，第二段必須以「回覆二：」開頭。",
        "instruction_id_list": ["zh_tw:combination:two_responses"],
        "kwargs": [kwargs(first_marker="回覆一：", second_marker="回覆二：")],
        "reference_response": "回覆一：可以先整理需求。\n回覆二：接著安排執行順序。",
    },
    {
        "prompt": "請先逐字重複「台灣的夏天很熱」，接著換行回答一項避暑建議。",
        "instruction_id_list": ["zh_tw:combination:repeat_prompt"],
        "kwargs": [kwargs(prompt_to_repeat="台灣的夏天很熱")],
        "reference_response": "台灣的夏天很熱\n可以安排室內展覽或早晨活動。",
    },
    {
        "prompt": "請用繁體中文寫一段短文，必須包含「捷運」和「悠遊卡」，且不得出現「汽車」。",
        "instruction_id_list": ["zh_tw:keywords:existence", "zh_tw:keywords:forbidden_words"],
        "kwargs": [kwargs(keywords=["捷運", "悠遊卡"]), kwargs(forbidden_words=["汽車"])],
        "reference_response": "搭乘捷運時可以使用悠遊卡，進出站都相當方便。",
    },
    {
        "prompt": "請用繁體中文列出 2 個編號項目，且最後必須以「完成。」結尾。",
        "instruction_id_list": ["zh_tw:detectable_format:number_numbered_lists", "zh_tw:startend:end_checker"],
        "kwargs": [kwargs(num_items=2), kwargs(end_phrase="完成。")],
        "reference_response": "1. 確認資料\n2. 送出申請\n完成。",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize(args: argparse.Namespace) -> Path:
    artifact_root = args.output_dir / args.dataset_dir_name
    artifact_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset, task in enumerate(TASKS):
        row = {
            "key": 2000 + offset,
            "benchmark": "IFEval zh-TW",
            "prompt": task["prompt"],
            "instruction_id_list": task["instruction_id_list"],
            "kwargs": task["kwargs"],
            "reference_response": task["reference_response"],
            "source": "Nejumi zh-TW adaptation of IFEval/M-IFEval verifiable instruction families",
        }
        rows.append(row)
    write_jsonl(artifact_root / "ifeval_zh_tw.jsonl", rows)
    write_jsonl(artifact_root / "reference_responses.jsonl", [
        {
            "key": row["key"],
            "prompt": row["prompt"],
            "response": row["reference_response"],
            "instruction_id_list": row["instruction_id_list"],
            "kwargs": row["kwargs"],
        }
        for row in rows
    ])
    (artifact_root / "README.md").write_text(
        "\n".join(
            [
                "# IFEval zh-TW",
                "",
                "Traditional Chinese mechanically verifiable instruction-following prompts",
                "for Nejumi Taiwan.",
                "",
                "This adaptation keeps objective IFEval-style constraints and excludes",
                "Japanese-specific M-IFEval constraints.",
                "",
                f"Rows: {len(rows)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "dataset_dir_name": args.dataset_dir_name,
        "total_instances": len(rows),
        "source_urls": {
            "ifeval": "https://huggingface.co/datasets/google/IFEval",
            "m_ifeval": "https://github.com/lightblue-tech/M-IFEval",
        },
        "license_note": (
            "Prompt family inspired by Apache-2.0 Google IFEval code/data and "
            "M-IFEval. This artifact contains newly written zh-TW prompts."
        ),
        "files": {},
    }
    for path in sorted(artifact_root.rglob("*")):
        if path.is_file():
            manifest["files"][str(path.relative_to(artifact_root))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    (artifact_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact_root


def upload_artifact(args: argparse.Namespace, artifact_root: Path) -> None:
    if not args.entity or not args.project:
        raise ValueError("--entity and --project are required with --upload")
    wandb.login()
    with wandb.init(
        entity=args.entity,
        project=args.project,
        job_type="data-upload",
        name=f"{args.artifact_name}-upload",
    ) as run:
        artifact = wandb.Artifact(
            args.artifact_name,
            type="dataset",
            metadata={
                "dataset_dir_name": args.dataset_dir_name,
                "total_instances": len(TASKS),
            },
        )
        artifact.add_dir(str(artifact_root), name=args.dataset_dir_name)
        run.log_artifact(artifact, aliases=["latest", "production"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/taiwan"))
    parser.add_argument("--dataset-dir-name", default=DEFAULT_DATASET_DIR_NAME)
    parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--entity")
    parser.add_argument("--project")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = materialize(args)
    print(artifact_root)
    if args.upload:
        upload_artifact(args, artifact_root)


if __name__ == "__main__":
    main()
