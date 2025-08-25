import json
import logging
import tempfile
import time
import shutil
import re
from pathlib import Path
from typing import Dict, List
import traceback
from functools import partial
import atexit
import threading
import multiprocessing as mp
import pandas as pd
import wandb
import weave
from tqdm import tqdm
import subprocess
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

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

Please provide the **unified diff** that fixes the bug. The diff MUST follow the standard unified diff format with proper line numbers.

Example of correct unified diff format:
```diff
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -10,7 +10,7 @@
 def function_name():
     context_line1
     context_line2
-    old_line_to_remove
+    new_line_to_add
     context_line3
     context_line4
```

Important requirements:
1. Each hunk MUST start with `@@ -start,count +start,count @@` where:
   - First pair (-start,count) refers to the original file
   - Second pair (+start,count) refers to the modified file
   - start = starting line number
   - count = number of lines in the hunk
2. Include at least 3 lines of context before and after changes
3. Use `-` prefix for removed lines and `+` prefix for added lines
4. Context lines have no prefix (just a space)

Enclose your diff in one of the following:

<patch>
...your unified diff here...
</patch>

OR

```diff
...your unified diff here...
```

Include only the minimal changes necessary to fix the bug."""
    
    return prompt

def build_prompt(instance: Dict) -> str:
    """プロンプト構築（公式データセット対応）"""
    # 公式データセットの'text'フィールドを優先使用
    if "text" in instance and instance["text"].strip():
        return instance["text"]
    # フォールバック
    return format_problem_statement(instance)

def generate_predictions(samples: List[Dict], llm, generator_config, output_file: Path, model_name: str):
    """パッチ生成とJSONL保存（並列処理版）"""
    print(f"Generating patches for {len(samples)} samples...")
    
    # 全サンプルのプロンプトを事前に準備
    all_inputs = []
    sample_data = []
    
    # Read fc_enabled from cfg
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config
    fc_enabled = bool(getattr(cfg.swebench, 'fc_enabled', False)) if hasattr(cfg, 'swebench') else False

    # Predefine tool schema if FC enabled
    submit_patch_tool = None
    system_prefix = None
    if fc_enabled:
        # Choose tool schema based on API family (Responses vs Chat Completions)
        api_type = getattr(cfg, 'api', 'openai')
        if api_type == 'openai_responses':
            # OpenAI Responses API: tools have top-level name/parameters
            submit_patch_tool = [
                {
                    "type": "function",
                    "name": "submit_patch",
                    "description": "Submit the unified diff (patch) that fixes the bug.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patch": {
                                "type": "string",
                                "description": "Unified diff string (use ```diff or <patch> ... </patch>)."
                            }
                        },
                        "required": ["patch"]
                    }
                }
            ]
        else:
            # Chat Completions style
            submit_patch_tool = [
                {
                    "type": "function",
                    "function": {
                        "name": "submit_patch",
                        "description": "Submit the unified diff (patch) that fixes the bug.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "patch": {
                                    "type": "string",
                                    "description": "Unified diff string (use ```diff or <patch> ... </patch>)."
                                }
                            },
                            "required": ["patch"]
                        }
                    }
                }
            ]
        system_prefix = "You are a software engineer. Produce ONLY the unified diff and return it via the function call 'submit_patch'. Do not include explanations."

    for sample in samples:
        instance_id = sample["instance_id"]
        prompt = build_prompt(sample)
        if fc_enabled and system_prefix:
            messages = [
                {"role": "system", "content": system_prefix},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Inject tools/tool_choice when FC is enabled
        if fc_enabled and submit_patch_tool is not None:
            api_type = getattr(cfg, 'api', 'openai')
            if api_type == 'openai_responses':
                tool_choice = {"type": "function", "name": "submit_patch"}
            else:
                tool_choice = {"type": "function", "function": {"name": "submit_patch"}}
            fc_kwargs = {**generator_config, "tools": submit_patch_tool, "tool_choice": tool_choice}
            all_inputs.append([messages, fc_kwargs])
        else:
            all_inputs.append([messages, generator_config])
        sample_data.append({
            "instance_id": instance_id,
            "sample": sample
        })
    
    # 並列処理でLLM呼び出し
    llm_ap = LLMAsyncProcessor(
        llm=llm,
        inputs=all_inputs,
    )
    responses = llm_ap.get_results()
    
    # 結果を処理
    for i, (response, sample_info) in enumerate(zip(responses, sample_data)):
        instance_id = sample_info["instance_id"]
        
        try:
            if not response:
                logger.error(f"No response for {instance_id}")
                continue
                
            # Prefer function call output when FC enabled
            raw_output = response.content if hasattr(response, 'content') else str(response)
            if fc_enabled and hasattr(response, 'tool_calls') and response.tool_calls:
                # Find submit_patch call
                for tc in response.tool_calls:
                    try:
                        if getattr(tc, 'name', '') == 'submit_patch' and isinstance(tc.arguments, dict):
                            candidate = tc.arguments.get('patch')
                            if isinstance(candidate, str) and candidate.strip():
                                raw_output = candidate
                                break
                    except Exception:
                        continue
            
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
                "model_name_or_path": model_name  # 設定から取得したモデル名を使用
            }
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(pred_data, ensure_ascii=False) + '\n')
                f.flush()
            
            if (i + 1) % 10 == 0: # このインデントは維持
                print(f"Progress: {i + 1}/{len(samples)} completed")
        except Exception as e:
            logger.error(f"Error processing {instance_id}: {e}")
            logger.error(traceback.format_exc()) #詳細なトレースバックをログに記録
            continue # 次のサンプルへ

def _api_http_json(method: str, url: str, body_obj=None, headers=None, timeout: float = 300.0):
    """HTTP JSON クライアント（タイムアウト/一時エラーに対してリトライ）"""
    import time as _time
    from urllib.error import URLError
    data = None
    if body_obj is not None:
        data = json.dumps(body_obj).encode("utf-8")
    attempt = 0
    backoff_sec = 2.0
    last_err = None
    
    # POSTリクエストは副作用があるため、リトライしない
    max_attempts = 1 if method == "POST" else 3
    
    while attempt < max_attempts:
        attempt += 1
        try:
            req = Request(url=url, data=data, method=method)
            req.add_header("Content-Type", "application/json")
            if headers:
                for k, v in (headers or {}).items():
                    req.add_header(k, v)
            with urlopen(req, timeout=timeout) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                text = resp.read().decode(charset)
                return json.loads(text) if text else {}
        except (TimeoutError, URLError, HTTPError) as e:  # network/transient
            # 524 (Gateway Timeout) などの一時的なエラーはリトライ（GETのみ）
            if isinstance(e, HTTPError) and e.code not in [524, 502, 503, 504]:
                raise  # 回復不可能なHTTPエラーは即座に失敗
            last_err = e
            if attempt >= max_attempts:
                raise
            print(f"[API] {method} request failed (attempt {attempt}/{max_attempts}): {e}. Retrying in {backoff_sec}s...")
            _time.sleep(backoff_sec)
            backoff_sec *= 2
    # 保険
    if last_err:
        raise last_err
    return {}


def run_swebench_evaluation(predictions_file: Path, max_workers: int = 4, instance_ids: List[str] = None, samples: List[Dict] = None) -> Dict:
    from swebench.harness.run_evaluation import main as run_evaluation
    instance = WandbConfigSingleton.get_instance()
    cfg = instance.config

    api_cfg = cfg.swebench.get("api_server", {}) if hasattr(cfg, 'swebench') else {}
    use_api = bool(api_cfg.get("enabled", False))

    if use_api:
        # Submit each instance to API server and aggregate results
        endpoint = (api_cfg.get("endpoint") or os.getenv("SWE_API_ENDPOINT") or "http://127.0.0.1:8000").rstrip("/")
        api_key = api_cfg.get("api_key") or os.getenv("SWE_API_KEY")
        # Effective image namespace/tag (simplified)
        # Priority: api_server.namespace/tag > images.namespace/tag > defaults
        images_cfg = getattr(cfg.swebench, 'images', None)
        ns = (
            api_cfg.get("namespace")
            or (images_cfg.get("namespace") if images_cfg else None)
            or "swebench"
        )
        tag = (
            api_cfg.get("tag")
            or (images_cfg.get("tag") if images_cfg else None)
            or "latest"
        )
        timeout_sec = int(api_cfg.get("timeout_sec", 1200))

        headers = {"X-API-Key": api_key} if api_key else {}
        # load predictions file
        instances = []
        with open(predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                instances.append(obj)

        resolved_ids = []
        unresolved_ids = []
        error_ids = []
        empty_patch_ids = []

        logger.info(f"Submitting {len(instances)} jobs to API server: {endpoint}")
        print(f"Submitting {len(instances)} jobs to API server: {endpoint}")

        # --- parallel submission/polling settings ---
        # 4コア環境では過負荷を避けるため、CPU数に基づく安全な既定値
        try:
            cpu_count = os.cpu_count() or 4
        except Exception:
            cpu_count = 4
        default_concurrency = max(1, min(int(cpu_count * 0.75), 8))
        concurrency = int(api_cfg.get("concurrency", default_concurrency))
        concurrency = max(1, min(concurrency, 32))

        # Simple bounded-concurrency loop
        in_flight = []  # list of dict(job_id, iid, start)
        pending = list(instances)

        def submit(pred):
            iid = pred["instance_id"]
            patch = pred.get("model_patch", "")
            payload = {
                "instance_id": iid,
                "patch_diff": patch,
                "namespace": ns,
                "tag": tag,
                "model_name_or_path": pred.get("model_name_or_path") or cfg.model.pretrained_model_name_or_path,
                "timeout_sec": timeout_sec,
            }
            job = _api_http_json("POST", f"{endpoint}/v1/jobs", body_obj=payload, headers=headers, timeout=300)
            return {"job_id": job.get("job_id"), "iid": iid, "start": time.time()}

        # prime submit
        while pending and len(in_flight) < concurrency:
            in_flight.append(submit(pending.pop(0)))

        processed = 0
        while in_flight:
            time.sleep(2)
            # poll all inflight
            new_in_flight = []
            for jf in in_flight:
                job_id = jf["job_id"]
                iid = jf["iid"]
                try:
                    j = _api_http_json("GET", f"{endpoint}/v1/jobs/{job_id}", headers=headers, timeout=300)
                except Exception as e:
                    logger.warning(f"poll failed for {iid}: {e}")
                    new_in_flight.append(jf)
                    continue
                status = j.get("status")
                if status in {"finished", "failed"}:
                    final_class = status
                    if status == "finished":
                        try:
                            rep = _api_http_json("GET", f"{endpoint}/v1/jobs/{job_id}/report", headers=headers, timeout=300)
                            if rep.get("error_instances") == 1 or iid in (rep.get("error_ids") or []):
                                final_class = "error"
                            elif iid in (rep.get("resolved_ids") or []):
                                final_class = "resolved"
                            elif iid in (rep.get("unresolved_ids") or []):
                                final_class = "unresolved"
                            elif iid in (rep.get("empty_patch_ids") or []):
                                final_class = "empty_patch"
                            else:
                                final_class = "error"
                        except Exception as e:
                            logger.warning(f"Failed to get report for {iid}: {e}")
                            final_class = "error"

                    if final_class == "resolved":
                        resolved_ids.append(iid)
                    elif final_class == "unresolved":
                        unresolved_ids.append(iid)
                    elif final_class == "empty_patch":
                        empty_patch_ids.append(iid)
                    else:
                        error_ids.append(iid)

                    processed += 1
                    if processed % 10 == 0 or processed == len(instances):
                        print(f"Progress: {processed}/{len(instances)} jobs completed")
                else:
                    new_in_flight.append(jf)

            in_flight = new_in_flight
            while pending and len(in_flight) < concurrency:
                in_flight.append(submit(pending.pop(0)))

        total = len(instances)
        return {
            "total_instances": total,
            "submitted_instances": total,
            "completed_instances": len(resolved_ids) + len(unresolved_ids) + len(error_ids) + len(empty_patch_ids),
            "resolved_instances": len(resolved_ids),
            "unresolved_instances": len(unresolved_ids),
            "empty_patch_instances": len(empty_patch_ids),
            "error_instances": len(error_ids),
            "completed_ids": resolved_ids + unresolved_ids + error_ids + empty_patch_ids,
            "incomplete_ids": [],
            "empty_patch_ids": empty_patch_ids,
            "submitted_ids": [p["instance_id"] for p in instances],
            "resolved_ids": resolved_ids,
            "unresolved_ids": unresolved_ids,
            "error_ids": error_ids,
            "schema_version": 2,
        }

    # Fallback to local official harness
    # run_id生成
    run_id = f"nejumi_{int(time.time())}"
    
    print(f"Running SWE-bench evaluation (run_id: {run_id})...")
    
    if samples is not None:
        # Arrow形式データセットから直接評価用データを構築
        print("Building evaluation dataset from Arrow format data...")
        
        # 一時的な評価用データセットを作成
        import tempfile
        
        temp_dir = Path(tempfile.mkdtemp(prefix="swebench_eval_"))
        eval_dataset_file = temp_dir / "eval_dataset.jsonl"
        
        # サンプルを評価用形式に変換
        with open(eval_dataset_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                # 公式SWE-bench形式に変換
                eval_sample = {
                    "repo": sample["repo"],
                    "instance_id": sample["instance_id"],
                    "base_commit": sample["base_commit"],
                    "patch": sample["patch"],
                    "test_patch": sample["test_patch"],
                    "problem_statement": sample["problem_statement"],
                    "hints_text": sample.get("hints_text", ""),
                    "created_at": sample["created_at"],
                    "version": sample["version"],
                    "FAIL_TO_PASS": sample["FAIL_TO_PASS"],
                    "PASS_TO_PASS": sample["PASS_TO_PASS"],
                    "environment_setup_commit": sample.get("environment_setup_commit", ""),
                }
                f.write(json.dumps(eval_sample, ensure_ascii=False) + '\n')
        
        # ローカルファイルパスを使用（公式ハーネスがサポートしている場合）
        print(f"Using local dataset file: {eval_dataset_file}")
        
        # ----------------------------------------------------------------------
        # ★★★ 事前にDockerイメージをビルドする処理を追加 ★★★
        # ----------------------------------------------------------------------
        # 設定で有効な場合のみ事前ビルドを実行
        instance = WandbConfigSingleton.get_instance()
        cfg = instance.config
        
        if cfg.swebench.get("prebuild_images", False):
            print("Pre-build images is enabled. Attempting to pre-build Docker images for the dataset...")
            try:
                # Dockerクライアントを作成してbuild_instance_images関数を呼び出す
                import docker
                from swebench.harness.docker_build import build_instance_images
                
                # eval_dataset_fileからデータセットを読み込む
                dataset_for_build = []
                with open(eval_dataset_file, 'r') as f:
                    for line in f:
                        dataset_for_build.append(json.loads(line))
                
                print(f"Pre-building Docker images for {len(dataset_for_build)} instances...")
                
                # Dockerクライアントを作成
                client = docker.from_env()
                
                # Docker Hubにログイン（環境変数から認証情報を取得）
                docker_username = os.environ.get('DOCKER_USERNAME')
                docker_password = os.environ.get('DOCKER_PASSWORD')
                
                if docker_username and docker_password:
                    try:
                        print(f"Logging in to Docker Hub as {docker_username}...")
                        client.login(username=docker_username, password=docker_password)
                        print("Successfully logged in to Docker Hub")
                    except Exception as e:
                        logger.warning(f"Failed to login to Docker Hub: {e}")
                        logger.warning("Proceeding without authentication (may hit rate limits)")
                else:
                    logger.warning("DOCKER_USERNAME or DOCKER_PASSWORD not set. Proceeding without Docker Hub authentication.")
                
                # build_instance_images関数を呼び出す
                # force_rebuild=Falseで既存のイメージは再利用
                
                # プライベートレジストリの設定を取得
                private_registry = cfg.swebench.get("private_registry", None)
                namespace = private_registry if private_registry else "swebench"
                
                build_instance_images(
                    client=client,
                    dataset=dataset_for_build,
                    force_rebuild=False,
                    max_workers=max_workers,
                    namespace=namespace,  # プライベートレジストリまたはデフォルト
                    tag="latest"  # instance_image_tagを追加
                )
                
                print("Successfully pre-built all necessary Docker images.")
                
            except ImportError as e:
                logger.warning(
                    f"Could not import required modules for pre-building: {e}. "
                    "Skipping pre-build. Images will be built on-demand during evaluation."
                )
            except Exception as e:
                logger.error(f"An error occurred during image pre-building: {e}")
                logger.error(traceback.format_exc())
                # ビルドエラーは警告として扱い、評価は続行する（オンデマンドビルドに期待）
                logger.warning("Image pre-build failed. Proceeding with evaluation (images will be built on-demand).")
        else:
            print("Pre-build images is disabled. Images will be built on-demand during evaluation.")

        try:
            result = run_evaluation(
                dataset_name=str(eval_dataset_file),  # ローカルファイルパス
                split="train",
                instance_ids=instance_ids,
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
        except Exception as e:
            logger.error(f"Local file approach failed: {e}")
            raise RuntimeError("Local dataset evaluation failed. No fallback to internet-based dataset.")
        
        # 一時ファイルをクリーンアップ（成功時のみ）
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass  # デバッグのため保持
        
    else:
        # インターネット経由の評価は削除（データの一元管理のため）
        raise RuntimeError("Internet-based dataset evaluation is disabled for data consistency.")
    
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

@weave.op(call_display_name=lambda _: "[SWE-Bench] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
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
        
        generator_config = {"max_tokens": max_tokens}
        
        # パッチ生成
        generate_predictions(samples, llm, generator_config, predictions_file, model_name)
        
        # 公式評価実行
        max_workers_config = cfg.swebench.get("max_workers", 4)
        
        # max_workersが"auto"の場合は自動計算
        if max_workers_config == "auto":
            import os
            cpu_count = os.cpu_count()
            # 推奨値: min(0.75 * cpu_count, 24) に安全マージン0.8を掛ける
            max_workers = min(int(0.75 * cpu_count * 0.8), 20)  # 24の代わりに20に制限
            print(f"Auto-calculated max_workers: {max_workers} (from {cpu_count} CPUs)")
        else:
            max_workers = max_workers_config
        
        # サンプリングしたIDを取得
        instance_ids = [sample["instance_id"] for sample in samples]

        if cfg.swebench.background_eval:
            # バックグラウンドで評価プロセスを起動し、完了時に集計するコールバックを返す
            ctx = mp.get_context("fork")
            result_queue = ctx.Queue(1)
            eval_process = ctx.Process(
                target=run_evaluation_proc,
                args=(predictions_file, max_workers, instance_ids, samples, result_queue)
            )
            eval_process.start()

            print("SWE-Bench evaluation started in background process. You can continue other benchmarks.")

            # 呼び出し側が終了前に明示的に結果集計できるようコールバックを返す
            def wait_and_log_metrics():
                try:
                    calculate_metrics(samples, result_queue, temp_dir)
                finally:
                    try:
                        eval_process.join(timeout=5)
                    except Exception:
                        pass

            return wait_and_log_metrics
        else:
            results = run_swebench_evaluation(predictions_file, max_workers, instance_ids, samples)
            return calculate_metrics(samples, results, temp_dir)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(traceback.format_exc()) # エラー発生時にトレースバックを出力
        raise  # エラーを再発生させて上位で処理させる

def run_evaluation_proc(predictions_file, max_workers, instance_ids, samples, result_queue):
    """SWE-Bench評価実行（バックグラウンド実行用）のラッパー関数"""
    # 進捗をリアルタイムで表示（プレフィックスは print 時に追加）
    try:
        print("[SWE-Bench] Starting evaluation...")
        result = run_swebench_evaluation(predictions_file, max_workers, instance_ids, samples)
        print("[SWE-Bench] Evaluation completed. Putting result to queue...")
        result_queue.put((result, ""))  # ログは既にリアルタイム出力済み
        print("[SWE-Bench] Result queued successfully.")
    except Exception as e:
        # 例外はメインプロセス側でraiseする
        print(f"[SWE-Bench] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put((e, ""))
        print("[SWE-Bench] Exception queued.")
    finally:
        # Queueを確実にフラッシュしてから自然終了
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass

def calculate_metrics(samples, results_or_queue, temp_dir):
    """SWE-Bench評価結果の集計"""
    instance = WandbConfigSingleton.get_instance()
    run = instance.run
    cfg = instance.config
    model_name = cfg.model.pretrained_model_name_or_path

    if cfg.swebench.background_eval:
        print("Waiting for SWE-Bench evaluation result...")
        # キューから結果とログを取得しログを表示
        try:
            results, log = results_or_queue.get(timeout=1800)
            print(f"Got result from queue. Type: {type(results)}")
            if log:
                print(f"Background process log:\n{log}")
            if isinstance(results, Exception):
                logger.error(f"Background evaluation failed with exception: {results}")
                raise results
        except Exception as e:
            logger.error(f"Failed to get results from queue: {e}")
            logger.error(f"Queue empty: {results_or_queue.empty()}")
            raise RuntimeError(f"Background evaluation failed or timed out: {e}")
    else:
        results = results_or_queue

    try:
        print(f"=== SWE-Bench Results ===")
        print(f"Total samples: {len(samples)}")
        print(f"Results type: {type(results)}")
        print(f"Results content: {results}")
        
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
        print("Logging per-instance output table to WandB...")
        try:
            # 予測 JSONL を読み込み
            predictions_map = {}
            predictions_file = temp_dir / "predictions.jsonl"
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
            missing_images = []  # 不足しているイメージを記録
            
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

                # イメージ存在確認（新規追加）
                # SWE-benchのイメージ名規則に従う
                # 設定からnamespaceを取得（後方互換を維持）
                instance = WandbConfigSingleton.get_instance()
                cfg = instance.config
                images_cfg = cfg.swebench.get("images", {}) if hasattr(cfg, 'swebench') else {}
                namespace = (images_cfg.get("namespace") or "swebench")
                
                # "__" を "_1776_" に変換（SWE-benchの特殊な命名規則）
                # 注: 現在のシステムでは namespace="swebench" でも変換されている
                iid_formatted = iid.replace("__", "_1776_")
                image_name = f"{namespace}/sweb.eval.x86_64.{iid_formatted.lower()}:latest"
                try:
                    import docker
                    client = docker.from_env()
                    client.images.get(image_name)
                    image_available = True
                except (docker.errors.ImageNotFound, ImportError, Exception):
                    image_available = False
                    missing_images.append(image_name)

                table_rows.append({
                    "instance_id": iid,
                    "status": status,
                    "input": inp,
                    "patch": patch_str,
                    "image_available": image_available,  # 新規追加
                    "image_name": image_name,           # 新規追加
                })

            # 不足しているイメージがあれば警告を出力
            if missing_images:
                logger.error("🚨 MISSING DOCKER IMAGES DETECTED!")
                logger.error(f"❌ {len(missing_images)}/{len(samples)} images are missing:")
                for img in missing_images:
                    logger.error(f"   - {img}")
                
                # Rate limitの可能性を判定
                if len(missing_images) > len(samples) * 0.3:  # 30%以上が不足
                    logger.error("🚨 DOCKER HUB RATE LIMIT LIKELY REACHED!")
                    logger.error("💡 Consider using private registry or waiting for rate limit reset")
                
                # WandBにも記録
                try:
                    run.log({
                        "missing_images_count": len(missing_images),
                        "total_instances": len(samples),
                        "image_coverage_rate": (len(samples) - len(missing_images)) / len(samples),
                        "rate_limit_suspicion": len(missing_images) > len(samples) * 0.3
                    })
                except Exception as e:
                    logger.warning(f"Failed to log image availability metrics: {e}")
            else:
                logger.info("✅ All Docker images are available")

            instance_df = pd.DataFrame(table_rows)
            run.log({"swebench_output_table": wandb.Table(dataframe=instance_df)})
            print("Per-instance table logged to WandB.")
        except Exception as e:
            logger.warning(f"Failed to log per-instance table: {e}")
        
        print("SWE-Bench evaluation completed and logged!")
        
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