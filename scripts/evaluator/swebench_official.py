import json
import logging
import tempfile
import time
import shutil
import re
from pathlib import Path
from typing import Dict, List
import traceback
import pandas as pd
import wandb
from tqdm import tqdm

from config_singleton import WandbConfigSingleton
from .evaluate_utils.llm_async_processor import LLMAsyncProcessor
from swebench.inference.make_datasets.utils import repair_patch
from swebench.inference.make_datasets.utils import extract_minimal_patch as _official_min_patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # change to DEBUG if needed

# ---------- context expansion settings ----------
MIN_CTX = 5  # keep at least 5 context lines before/after change

import re

hunk_header_re = re.compile(r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@")

# Patch extraction patterns
PATCH_PATTERN = re.compile(
    r"(^diff --git.*?(?=^diff --git|^--- |^\+\+\+ |^@@ |$))", 
    re.MULTILINE | re.DOTALL
)
PATCH_FILE_PATTERN = re.compile(
    r"^diff --git a/(.*?) b/(.*?)$", 
    re.MULTILINE
)
PATCH_HUNK_PATTERN = re.compile(
    r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@(.*?)(?=^@@ |^diff |^--- |^\+\+\+ |$)", 
    re.MULTILINE | re.DOTALL
)

def expand_hunk_headers(patch: str, ctx: int = MIN_CTX) -> str:
    """Increase pre_len/post_len by ctx*2 in hunk headers to allow fuzzy apply."""
    new_lines = []
    for line in patch.split("\n"):
        m = hunk_header_re.match(line)
        if m:
            pre_start, pre_len, post_start, post_len = map(int, m.groups())
            pre_len += ctx * 2
            post_len += ctx * 2
            line = f"@@ -{pre_start},{pre_len} +{post_start},{post_len} @@"
        new_lines.append(line)
    return "\n".join(new_lines)

def get_first_idx(charlist: List[str | None]) -> int:
    first_min = charlist.index("-") if "-" in charlist else len(charlist)
    first_plus = charlist.index("+") if "+" in charlist else len(charlist)
    return min(first_min, first_plus)

def get_last_idx(charlist: List[str | None]) -> int:
    char_idx = get_first_idx(charlist[::-1])
    last_idx = len(charlist) - char_idx
    return last_idx

def strip_content(hunk_content: str) -> tuple[str, int]:
    lines = hunk_content.split("\n")
    first_chars = [line[0] if len(line) > 0 else " " for line in lines]
    
    idx_first_change = get_first_idx(first_chars)
    
    # Determine the slice for lines that contain actual changes or are between them
    # Find the start of the last sequence of +/- lines
    _hsplit_rev = lines[::-1]
    _first_chars_rev = [line[0] if len(line) > 0 else " " for line in _hsplit_rev]
    idx_last_change_from_rev = get_first_idx(_first_chars_rev)
    idx_end_slice = len(lines) - idx_last_change_from_rev

    if idx_first_change < idx_end_slice:
        relevant_lines = lines[idx_first_change:idx_end_slice]
    else: # No lines with +/- or hunk is empty/all context
        relevant_lines = []

    processed_lines = [line.rstrip() for line in relevant_lines]
    processed_lines = [line if line.strip() else " " for line in processed_lines]
    
    new_hunk_str = "\n" + "\n".join(processed_lines)
    if processed_lines: 
        new_hunk_str += "\n"

    return new_hunk_str, idx_first_change

def get_hunk_stats(pre_start: int, pre_len_orig: int, post_start_orig: int, post_len_orig: int, 
                   stripped_hunk_content: str, total_delta: int) -> tuple[int, int, int, int, int]:
    stats = {"context": 0, "added": 0, "subtracted": 0}
    # .strip('\n') because stripped_hunk_content starts and ends with \n if not empty
    hunk_body_lines = stripped_hunk_content.strip("\n").split("\n")
    if not stripped_hunk_content.strip(): # Handle empty hunk after stripping
        hunk_body_lines = []

    for line in hunk_body_lines:
        if not line and len(hunk_body_lines) == 1: # single empty line from strip_content for empty hunk
            continue
        if line.startswith("-"):
            stats["subtracted"] += 1
        elif line.startswith("+"):
            stats["added"] += 1
        else:
            stats["context"] += 1
    
    context = stats["context"]
    added = stats["added"]
    subtracted = stats["subtracted"]
    
    new_pre_len = context + subtracted
    new_post_start = pre_start + total_delta # pre_start here is already adjusted
    new_post_len = context + added
    
    current_hunk_delta = added - subtracted
    new_total_delta = total_delta + current_hunk_delta
    
    return pre_start, new_pre_len, new_post_start, new_post_len, new_total_delta

def extract_minimal_patch(model_patch_str: str | None) -> str:
    if model_patch_str is None:
        return ""
    
    model_patch_str = model_patch_str.lstrip("\n")
    final_reconstructed_patch = ""
    
    for patch_segment in PATCH_PATTERN.findall(model_patch_str):
        single_file_patch_str = ""
        total_delta_for_file = 0 
        
        file_header_match = PATCH_FILE_PATTERN.search(patch_segment)
        if not file_header_match:
            logger.debug(f"Skipping segment, no file header: {patch_segment[:100]}...")
            continue
        file_header = file_header_match.group(0)
        single_file_patch_str += file_header + "\n"

        hunk_matches = list(PATCH_HUNK_PATTERN.finditer(patch_segment))
        if not hunk_matches:
             # This could be a file creation/deletion with no content changes (e.g. empty file) or just mode changes
             # If it has a valid file header, we keep it. The patch command can handle this.
             if file_header.strip(): 
                 logger.debug(f"Segment has file header but no hunks: {file_header}")
             # else: (No file header was already skipped)
                 # It means the patch segment itself was probably just the file header

        for hunk_match in hunk_matches:
            orig_pre_start_str, orig_pre_len_str, orig_post_start_str, orig_post_len_str, captured_hunk_content = hunk_match.groups()
            
            try:
                orig_pre_start, orig_pre_len, orig_post_start, orig_post_len = map(int, [
                    orig_pre_start_str, orig_pre_len_str, orig_post_start_str, orig_post_len_str
                ])
            except ValueError:
                logger.warning(f"Could not parse hunk line numbers: {hunk_match.group(0)}")
                continue # Skip malformed hunk
            
            stripped_content, num_leading_context_in_capture = strip_content(captured_hunk_content)
            
            # pre_start for get_hunk_stats should be the true start of modifications within the original file
            adjusted_pre_start = orig_pre_start + num_leading_context_in_capture
            
            _, new_pre_len, new_post_start, new_post_len, total_delta_for_file = get_hunk_stats(
                adjusted_pre_start, 
                orig_pre_len, # Original length, get_hunk_stats recalculates based on stripped_content
                orig_post_start,
                orig_post_len,
                stripped_content, 
                total_delta_for_file 
            )
            
            reconstructed_hunk_header = f"@@ -{adjusted_pre_start},{new_pre_len} +{new_post_start},{new_post_len} @@"
            
            single_file_patch_str += reconstructed_hunk_header
            # stripped_content includes its own leading/trailing newlines if content exists
            # if it's just "\n", it means an empty hunk body after stripping
            if stripped_content == "\n" and not (new_pre_len == 0 and new_post_len == 0) : # an empty hunk body
                 single_file_patch_str += "\n" # Ensure the hunk ends with a newline
            else:
                 single_file_patch_str += stripped_content # which is "\n" or "\n actual content \n"

        if single_file_patch_str.strip():
            final_reconstructed_patch += single_file_patch_str

    if not final_reconstructed_patch.strip() and model_patch_str.strip():
        logger.debug(f"Original patch resulted in empty minimal patch. Original: {model_patch_str[:200]}...")
        return "" 
    return final_reconstructed_patch

def extract_diff(response: str | None) -> str | None:
    """
    公式SWE-benchのextract_diff関数（完全複製）
    Extracts the diff from a response formatted in different ways
    """
    if response is None:
        return None
    diff_matches = []
    other_matches = []
    pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    if diff_matches:
        return diff_matches[0]
    if other_matches:
        return other_matches[0]
    return response.split("</s>")[0]

def format_problem_statement(instance: Dict) -> str:
    """プロンプト生成（公式スタイルに近づける）"""
    problem_statement = instance["problem_statement"]
    hints_text = instance.get("hints_text", "")
    
    prompt = f"""You are a software engineer tasked with fixing a bug in a Python codebase.

<issue>
{problem_statement}
</issue>"""

    if hints_text and hints_text.strip():
        prompt += f"""

<hints>
{hints_text}
</hints>"""

    prompt += """

Please provide the **unified diff** that fixes the bug, enclosed in one of the following:

<patch>
...diff here...
</patch>

または

```diff
...diff here...
```

必要最小限の変更のみを含めてください。"""
    
    return prompt

def build_prompt(instance: Dict) -> str:
    """プロンプト構築（公式データセット対応）"""
    # 公式データセットの'text'フィールドを優先使用
    if "text" in instance and instance["text"].strip():
        return instance["text"]
    # フォールバック
    return format_problem_statement(instance)

def generate_predictions(samples: List[Dict], llm, generator_config, output_file: Path):
    """パッチ生成とJSONL保存"""
    print(f"Generating patches for {len(samples)} samples...")
    
    for i, sample in enumerate(tqdm(samples, desc="Generating predictions")):
        instance_id = sample["instance_id"]
        
        try:
            # プロンプト作成
            prompt = build_prompt(sample)
            messages = [{"role": "user", "content": prompt}]
            
            # LLM呼び出し
            llm_ap = LLMAsyncProcessor(
                llm=llm,
                inputs=[[messages, generator_config]],
            )
            responses = llm_ap.get_results()
            
            if not responses or len(responses) == 0:
                logger.error(f"No response for {instance_id}")
                continue
                
            response = responses[0]
            raw_output = response.content if hasattr(response, 'content') else str(response)
            
            logger.debug(f"LLM raw output (first 300 chars): {raw_output[:300].replace(chr(10),' ')}")
            extracted_patch_str = extract_diff(raw_output)
            logger.debug(f"extract_diff result (first 200 chars): {str(extracted_patch_str)[:200].replace(chr(10),' ')}")

            # 直接 repair_patch をかけてコンテキストを維持したまま整形
            repaired = repair_patch(extracted_patch_str) if extracted_patch_str else ""
            # ファイルパスが途中で改行されているケースを補修
            repaired = _fix_split_headers(repaired) if repaired else ""
            # 最小化パッチも試作し、短い方を採用（行番号再計算で適用成功率↑）
            minimal_patch = _official_min_patch(repaired) if repaired else ""
            final_patch_to_save = minimal_patch if (minimal_patch and len(minimal_patch) <= len(repaired)) else repaired

            # skip if still empty
            if not final_patch_to_save or not final_patch_to_save.strip():
                logger.warning(f"No usable patch for {instance_id}; skipping.")
                continue

            # 末尾に改行を追加（patch コマンド要求）
            if not final_patch_to_save.endswith('\n'):
                final_patch_to_save += '\n'

            pred_data = {
                "instance_id": instance_id,
                "model_patch": final_patch_to_save,
                "model_name_or_path": "gpt-4.1-2025-04-14"
            }
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(pred_data, ensure_ascii=False) + '\n')
                f.flush()
            
            if (i + 1) % 10 == 0: # このインデントは維持
                print(f"Progress: {i + 1}/{len(samples)} completed")
        except Exception as e:
            logger.error(f"Error processing {instance_id}: {e}")
            logger.error(traceback.format_exc()) #詳細なトレースバックをログに記録
            # エラー時も空のパッチを保存する（オプション）
            # pred_data = { ... "model_patch": "" ... } ... f.write ...
            continue # 次のサンプルへ

def run_swebench_evaluation(predictions_file: Path, max_workers: int = 4) -> Dict:
    """公式SWE-bench評価実行"""
    from swebench.harness.run_evaluation import main as run_evaluation
    
    # run_id生成
    run_id = f"nejumi_{int(time.time())}"
    
    print(f"Running SWE-bench evaluation (run_id: {run_id})...")
    
    # 公式評価実行
    result = run_evaluation(
        dataset_name="princeton-nlp/SWE-bench_Verified",
        split="test",
        instance_ids=None,
        predictions_path=str(predictions_file),
        max_workers=max_workers,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id=run_id,
        timeout=1800,
        namespace="swebench",
        rewrite_reports=False,
        modal=False,
        instance_image_tag="latest",
        report_dir="."
    )
    
    print(f"Evaluation result type: {type(result)}")
    
    # 結果がファイルパスの場合は読み込み
    if isinstance(result, (str, Path)):
        result_file = Path(result)
        if result_file.exists():
            with open(result_file, 'r') as f:
                return json.load(f)
        else:
            raise RuntimeError(f"Result file not found: {result_file}")
    
    return result

def evaluate():
    """SWE-bench評価メイン関数"""
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    llm = instance.llm

    dataset_name = "swebench"
    
    # データセットダウンロード
    artifact = run.use_artifact(cfg[dataset_name].artifacts_path, type="dataset")
    artifact_dir = artifact.download()
    dataset_dir = Path(artifact_dir) / cfg[dataset_name].dataset_dir
    
    # Arrow形式データ読み込み
    from datasets import load_from_disk
    hf_dataset = load_from_disk(str(dataset_dir))
    
    # 'test' split取得
    if hasattr(hf_dataset, 'keys') and 'test' in hf_dataset:
        hf_dataset = hf_dataset['test']
    
    # データを辞書形式で取得
    task_data = []
    for sample in hf_dataset:
        task_data.append(sample)
    
    print(f"Loaded {len(task_data)} samples from dataset")
    
    # サンプル数制限
    max_samples = cfg.swebench.get("max_samples", 500)
    if cfg.testmode:
        max_samples = 2 # テストモード時はサンプル数を減らす
    
    samples = task_data[:max_samples]
    print(f"Processing {len(samples)} samples")
    
    # 一時ディレクトリ
    temp_dir = Path(tempfile.mkdtemp(prefix="swebench_official_"))
    predictions_file = temp_dir / "predictions.jsonl"
    logger.info(f"Predictions will be saved to: {predictions_file.resolve()}") # フルパスを出力
    
    try:
        # 生成パラメータ
        max_tokens = cfg.swebench.get("max_tokens", 32768)
        model_name = cfg.model.pretrained_model_name_or_path
        
        if model_name in ["o1", "o1-preview", "o1-mini", "o3", "o3-mini"]:
            generator_config = {}
        else:
            generator_config = {"max_tokens": max_tokens}
        
        # パッチ生成
        generate_predictions(samples, llm, generator_config, predictions_file)
        
        # 公式評価実行
        max_workers = cfg.swebench.get("max_workers", 4)
        results = run_swebench_evaluation(predictions_file, max_workers)
        
        print(f"=== SWE-Bench Results ===")
        print(f"Total samples: {len(samples)}")
        
        if isinstance(results, dict):
            resolved = results.get("resolved_instances", results.get("resolved", 0))
            total = results.get("total_instances", results.get("total", len(samples)))
            if total == 0 and len(samples) > 0 : # totalが0だがサンプル処理した場合
                total = len(samples)

            print(f"Resolved: {resolved}")
            print(f"Total processed for eval: {total}") # 評価が実際に処理した数

            # WandB logging
            # totalが0の場合の除算エラーを避ける
            resolution_rate = resolved / total if total > 0 else 0.0
            print(f"Resolution rate: {resolution_rate:.3f}")

            leaderboard_data = {
                "model_name": model_name,
                "total_samples": total,
                "issues_resolved": resolved,
                "resolution_rate": resolution_rate,
                "total_samples_in_script": len(samples)
            }
            
            leaderboard_table = pd.DataFrame([leaderboard_data])
            run.log({"swebench_leaderboard_table": wandb.Table(dataframe=leaderboard_table)})
            run.log({"swebench_results": results})
        
        # -------- per-instance table (inputs / outputs / status) --------
        try:
            # 予測 JSONL を読み込み
            predictions_map = {}
            with open(predictions_file, "r", encoding="utf-8") as pf:
                for line in pf:
                    obj = json.loads(line)
                    predictions_map[obj["instance_id"]] = obj

            # ステータス集合
            resolved_set = set(results.get("resolved_ids", []))
            unresolved_set = set(results.get("unresolved_ids", []))
            error_set = set(results.get("error_ids", []))
            empty_set = set(results.get("empty_patch_ids", []))

            table_rows = []
            for sample in samples:
                iid = sample["instance_id"]
                if iid in resolved_set:
                    status = "resolved"
                elif iid in unresolved_set:
                    status = "unresolved"
                elif iid in empty_set:
                    status = "empty_patch"
                elif iid in error_set:
                    status = "error"
                else:
                    status = "not_submitted"

                # 入力は text 優先、なければ problem_statement
                inp = sample.get("text") or sample.get("problem_statement", "")
                patch_str = predictions_map.get(iid, {}).get("model_patch", "")

                table_rows.append({
                    "instance_id": iid,
                    "status": status,
                    "input": inp,
                    "patch": patch_str,
                })

            instance_df = pd.DataFrame(table_rows)
            run.log({"swebench_instance_table": wandb.Table(dataframe=instance_df)})
        except Exception as e:
            logger.warning(f"Failed to log per-instance table: {e}")
        
        print("SWE-Bench evaluation completed!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(traceback.format_exc()) # エラー発生時にトレースバックを出力
        # raise # デバッグ中は再raiseしない方がファイル確認しやすい場合もある
    
    finally:
        # クリーンアップ
        # try:
        #     shutil.rmtree(temp_dir) # ★★★ この行を一時的にコメントアウト ★★★
        #     logger.info(f"Successfully cleaned up temp directory: {temp_dir}")
        # except Exception as e:
        #     logger.warning(f"Failed to cleanup temp directory: {e}")
        logger.info(f"Temporary directory {temp_dir} was NOT cleaned up for debugging.") # コメントアウトしたことを明記 

# ---------- new helper to fix broken header line breaks ----------

def _fix_split_headers(patch: str) -> str:
    """Join broken file header lines where path got split by newline.

    Example of broken header::
        --- a/astropy/modeling/separable.py
        +++ b/astropy
        /modeling/separable.py

    The second header line should be a single line. We detect header lines
    that start with '--- ' or '+++ ' but *do not* contain a file separator
    ('.py', '.c', etc.) after the initial marker and join with subsequent
    lines that continue the path (they typically start with '/' because of
    the split).
    """
    if not patch:
        return patch

    lines: List[str] = patch.split("\n")
    fixed_lines: List[str] = []
    i = 0
    header_start = ("--- ", "+++ ")
    while i < len(lines):
        line = lines[i]
        if line.startswith(header_start):
            # Heuristic: if the line does not end with a known extension and the next
            # line continues the path (starts with '/') then join them.
            # We'll keep appending subsequent lines that look like they belong
            # to the path until we encounter a line that starts a new diff marker
            # or hunk ("@@ ").
            if not re.search(r"\.[a-zA-Z0-9]+$", line) and (i + 1) < len(lines):
                j = i + 1
                joined = line
                while j < len(lines):
                    next_line = lines[j]
                    if next_line.startswith(("@@ ", "diff ", "--- ", "+++ ")):
                        break
                    # Likely continuation of the path (common cases start with '/')
                    joined += next_line.strip("\n")
                    j += 1
                fixed_lines.append(joined)
                i = j
                continue
        fixed_lines.append(line)
        i += 1
    return "\n".join(fixed_lines) 

# --- extend patch apply commands to allow higher fuzz ---
try:
    from swebench.harness.run_evaluation import GIT_APPLY_CMDS
    extra_cmds = [
        "patch --batch --fuzz=10 -p1 -i",
        "patch --batch --fuzz=20 -p1 -i",
    ]
    for cmd in extra_cmds:
        if cmd not in GIT_APPLY_CMDS:
            GIT_APPLY_CMDS.append(cmd)
except ImportError:
    # SWE-benchがインストールされていない場合は警告のみ
    logger.warning("Failed to extend patch commands: swebench not installed") 