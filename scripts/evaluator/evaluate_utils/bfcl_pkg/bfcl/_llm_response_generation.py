import argparse
import json
import time
import traceback
import asyncio
from datetime import datetime
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy

from .constants.category_mapping import (
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    TEST_FILE_MAPPING,
)
from .constants.eval_config import *
from .eval_checker.eval_runner_helper import load_file
from .constants.model_config import MODEL_CONFIG_MAPPING
from .model_handler.api_inference.openrouter import OpenRouterHandler
from .model_handler.model_style import ModelStyle
from .utils import is_multi_turn, parse_test_category_argument, sort_key
from tqdm import tqdm

RETRY_LIMIT = 3
# 60s for the timer to complete. But often we find that even with 60 there is a conflict. So 65 is a safe no.
RETRY_DELAY = 65  # Delay in seconds


class BFCLStalledError(RuntimeError):
    """Raised when BFCL generation is making too little progress to continue safely."""


def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _test_case_id(test_case):
    if isinstance(test_case, dict):
        return test_case.get("id", "<unknown>")
    return "<unknown>"


def _positive_float_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _nonnegative_int(value, default=0):
    if value is None:
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _build_case_timeout_result(test_case, elapsed_sec, case_timeout_sec, attempts, retries):
    case_id = _test_case_id(test_case)
    return {
        "id": case_id,
        "result": (
            "Error during inference: BFCL case timeout after "
            f"{elapsed_sec:.1f}s (limit {case_timeout_sec:.1f}s)"
        ),
        "error": "bfcl_case_timeout",
        "timeout": True,
        "timeout_attempts": attempts,
        "timeout_retries": retries,
        "latency": elapsed_sec,
        "input_token_count": 0,
        "output_token_count": 0,
    }


def _effective_case_timeout_sec(args):
    case_timeout_sec = _positive_float_or_none(getattr(args, "case_timeout_sec", None))
    return case_timeout_sec


def _retryable_failed_result(entry):
    if not isinstance(entry, dict):
        return False
    if entry.get("error") == "bfcl_case_timeout":
        return True
    result = entry.get("result")
    return isinstance(result, str) and result.startswith("Error during inference:")


def _is_case_timeout_result(entry):
    return isinstance(entry, dict) and entry.get("error") == "bfcl_case_timeout"


def get_args():
    parser = argparse.ArgumentParser()
    # Refer to model_choice for supported models.
    parser.add_argument("--model", type=str, default="gorilla-openfunctions-v2", nargs="+")
    # Refer to test_categories for supported categories.
    parser.add_argument("--test-category", type=str, default="all", nargs="+")

    # Parameters for the model that you want to test.
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--include-input-log", action="store_true", default=False)
    parser.add_argument("--exclude-state-log", action="store_true", default=False)
    parser.add_argument("--num-threads", default=1, type=int)
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--backend", default="vllm", type=str, choices=["vllm", "sglang"])
    parser.add_argument("--gpu-memory-utilization", default=0.9, type=float)
    parser.add_argument("--result-dir", default=None, type=str)
    parser.add_argument("--run-ids", action="store_true", default=False)
    parser.add_argument("--allow-overwrite", "-o", action="store_true", default=False)
    # Add the new skip_vllm argument
    parser.add_argument(
        "--skip-server-setup",
        action="store_true",
        default=False,
        help="Skip vLLM/SGLang server setup and use existing endpoint specified by the VLLM_ENDPOINT and VLLM_PORT environment variables."
    )
    # Optional local model path
    parser.add_argument(
        "--local-model-path",
        type=str,
        default=None,
        help="Specify the path to a local directory containing the model's config/tokenizer/weights for fully offline inference. Use this only if the model weights are stored in a location other than the default HF_HOME directory.",
    )
    args = parser.parse_args()

    return args

def build_handler(model_name, temperature):
    config = MODEL_CONFIG_MAPPING[model_name]
    handler = config.model_handler(model_name, temperature)
    # Propagate config flags to the handler instance
    handler.is_fc_model = config.is_fc_model
    return handler

def get_involved_test_entries(test_category_args, run_ids, samples_per_category=None,artifacts_path=None):
    all_test_file_paths, all_test_categories, all_test_entries_involved = [], [], []
    if run_ids:
        with open(TEST_IDS_TO_GENERATE_PATH) as f:
            test_ids_to_generate = json.load(f)
        for category, test_ids in test_ids_to_generate.items():
            if len(test_ids) == 0:
                continue
            test_file_path = TEST_FILE_MAPPING[category]
            all_test_entries_involved.extend(
                [
                    entry
                    for entry in load_file(artifacts_path + PROMPT_PATH + test_file_path)
                    if entry["id"] in test_ids
                ]
            )
            all_test_categories.append(category)
            all_test_file_paths.append(test_file_path)

    else:
        all_test_file_paths, all_test_categories = parse_test_category_argument(test_category_args)
        # Make a copy here since we are removing list elemenets inside the for loop
        for test_category, file_to_open in zip(
            all_test_categories[:], all_test_file_paths[:]
        ):
            # Load all entries for the category
            category_entries = load_file(artifacts_path + PROMPT_PATH + file_to_open)
            if category_entries:
                if samples_per_category is not None and samples_per_category > 0:
                    # Take specified number of samples from each category
                    all_test_entries_involved.extend(category_entries[:samples_per_category])
                else:
                    # Use all samples if samples_per_category is None or 0
                    all_test_entries_involved.extend(category_entries)

    return (
        all_test_file_paths,
        all_test_categories,
        all_test_entries_involved,
    )


def collect_test_cases(
    args, model_name, all_test_categories, all_test_file_paths, all_test_entries_involved
):
    model_name_dir = model_name.replace("/", "_")
    model_result_dir = args.result_dir / model_name_dir

    existing_result = []
    retry_failed_cases = bool(getattr(args, "retry_failed_cases", True))
    for test_category, file_to_open in zip(all_test_categories, all_test_file_paths):

        result_file_path = model_result_dir / file_to_open.replace(".json", "_result.json")
        if result_file_path.exists():
            # Not allowing overwrite, we will load the existing results
            if not args.allow_overwrite:
                existing_result.extend(load_file(result_file_path))
            # Allow overwrite and not running specific test ids, we will delete the existing result file before generating new results
            elif not args.run_ids:
                result_file_path.unlink()
            # Allow overwrite and running specific test ids, we will do nothing here
            else:
                pass

        if retry_failed_cases:
            existing_result = [
                entry for entry in existing_result if not _retryable_failed_result(entry)
            ]

        existing_ids = [entry["id"] for entry in existing_result]

    test_cases_to_generate = [
        test_case
        for test_case in all_test_entries_involved
        if test_case["id"] not in existing_ids
    ]
    test_cases_to_generate = process_multi_turn_test_case(test_cases_to_generate,args.artifacts_path)

    return sorted(test_cases_to_generate, key=sort_key)

def process_multi_turn_test_case(test_cases,artifacts_path):
    """
    Multi-turn test cases don't have the function doc in the prompt. We need to add them here.
    """
    for entry in test_cases:
        if not is_multi_turn(entry["id"]):
            continue
        involved_classes = entry["involved_classes"]
        entry["function"] = []
        for func_collection in involved_classes:
            # func_doc is a list of dict
            func_doc = load_file(
                artifacts_path + MULTI_TURN_FUNC_DOC_PATH + MULTI_TURN_FUNC_DOC_FILE_MAPPING[func_collection]
            )
            entry["function"].extend(func_doc)

        # Handle Miss Func category; we need to remove the holdout function doc
        if "missed_function" in entry:
            for turn_index, missed_func_names in entry["missed_function"].items():
                entry["missed_function"][turn_index] = []
                for missed_func_name in missed_func_names:
                    for i, func_doc in enumerate(entry["function"]):
                        if func_doc["name"] == missed_func_name:
                            # Add the missed function doc to the missed_function list
                            entry["missed_function"][turn_index].append(func_doc)
                            # Remove it from the function list
                            entry["function"].pop(i)
                            break

    return test_cases


def multi_threaded_inference(handler, test_case, include_input_log, exclude_state_log):

    assert type(test_case["function"]) is list

    retry_count = 0

    while True:
        try:
            result, metadata = handler.inference(
                deepcopy(test_case), include_input_log, exclude_state_log
            )
            break  # Success, exit the loop
        except Exception as e:
            # TODO: It might be better to handle the exception in the handler itself rather than a universal catch block here, as each handler use different ways to call the endpoint.
            # OpenAI has openai.RateLimitError while Anthropic has anthropic.RateLimitError. It would be more robust in the long run.
            if retry_count < RETRY_LIMIT and (
                "rate limit reached" in str(e).lower()
                or (hasattr(e, "status_code") and (e.status_code in {429, 503, 500}))
            ):
                print(
                    f"Rate limit reached. Sleeping for 65 seconds. Retry {retry_count + 1}/{RETRY_LIMIT}"
                )
                time.sleep(RETRY_DELAY)
                retry_count += 1
            else:
                # This is usually the case when the model getting stuck on one particular test case.
                # For example, timeout error or FC model returning invalid JSON response.
                # Since temperature is already set to 0.001, retrying the same test case will not help.
                # So we continue the generation process and record the error message as the model response
                print("-" * 100)
                print(
                    "❗️❗️ Error occurred during inference. Maximum reties reached for rate limit or other error. Continuing to next test case."
                )
                print(f"❗️❗️ Test case ID: {test_case['id']}, Error: {str(e)}")
                traceback.print_exc()
                print("-" * 100)

                return {
                    "id": test_case["id"],
                    "result": f"Error during inference: {str(e)}",
                }

    result_to_write = {
        "id": test_case["id"],
        "result": result,
    }

    result_to_write.update(metadata)

    return result_to_write


async def async_inference(
    handler,
    test_case,
    include_input_log,
    exclude_state_log,
    case_timeout_sec=None,
    case_timeout_retries=1,
):

    assert type(test_case["function"]) is list

    retry_count = 0
    timeout_attempts = 0
    case_timeout_sec = _positive_float_or_none(case_timeout_sec)
    case_timeout_retries = _nonnegative_int(case_timeout_retries, default=1)
    case_id = _test_case_id(test_case)
    if is_multi_turn(case_id):
        case_timeout_retries = 0
    case_started_at = time.monotonic()

    while True:
        started_at = time.monotonic()
        try:
            inference_task = handler.inference_async(
                deepcopy(test_case), include_input_log, exclude_state_log
            )
            if case_timeout_sec is not None:
                result, metadata = await asyncio.wait_for(
                    inference_task,
                    timeout=case_timeout_sec,
                )
            else:
                result, metadata = await inference_task
            break  # Success, exit the loop
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - started_at
            timeout_attempts += 1
            case_id = _test_case_id(test_case)
            if timeout_attempts <= case_timeout_retries:
                print(
                    f"[{_ts()}] BFCL async case timed out; retrying: "
                    f"{case_id} attempt {timeout_attempts}/"
                    f"{case_timeout_retries} "
                    f"({elapsed:.1f}s > {case_timeout_sec:.1f}s)",
                    flush=True,
                )
                continue

            result = _build_case_timeout_result(
                test_case,
                elapsed,
                case_timeout_sec,
                timeout_attempts,
                case_timeout_retries,
            )
            print(
                f"[{_ts()}] BFCL async case timed out; recording inference "
                f"error and continuing: {result['id']} "
                f"after {timeout_attempts} timed-out attempts "
                f"({elapsed:.1f}s > {case_timeout_sec:.1f}s)",
                flush=True,
            )
            return result
        except Exception as e:
            # TODO: It might be better to handle the exception in the handler itself rather than a universal catch block here, as each handler use different ways to call the endpoint.
            # OpenAI has openai.RateLimitError while Anthropic has anthropic.RateLimitError. It would be more robust in the long run.
            if retry_count < RETRY_LIMIT and (
                "rate limit reached" in str(e).lower()
                or (hasattr(e, "status_code") and (e.status_code in {429, 503, 500}))
            ):
                case_elapsed = time.monotonic() - case_started_at
                if case_timeout_sec is not None:
                    remaining = case_timeout_sec - case_elapsed
                    if remaining <= 0:
                        result = _build_case_timeout_result(
                            test_case,
                            case_elapsed,
                            case_timeout_sec,
                            timeout_attempts + 1,
                            case_timeout_retries,
                        )
                        print(
                            f"[{_ts()}] BFCL async case timed out during "
                            f"provider retry handling: {result['id']} "
                            f"({case_elapsed:.1f}s > {case_timeout_sec:.1f}s)",
                            flush=True,
                        )
                        return result
                    retry_delay = min(RETRY_DELAY, remaining)
                else:
                    retry_delay = RETRY_DELAY
                print(
                    f"[{_ts()}] Rate limit reached. Sleeping for "
                    f"{retry_delay:.1f} seconds. Retry {retry_count + 1}/{RETRY_LIMIT}",
                    flush=True,
                )
                await asyncio.sleep(retry_delay)
                if case_timeout_sec is not None and retry_delay < RETRY_DELAY:
                    case_elapsed = time.monotonic() - case_started_at
                    result = _build_case_timeout_result(
                        test_case,
                        case_elapsed,
                        case_timeout_sec,
                        timeout_attempts + 1,
                        case_timeout_retries,
                    )
                    print(
                        f"[{_ts()}] BFCL async case timed out during provider "
                        f"retry sleep: {result['id']} "
                        f"({case_elapsed:.1f}s > {case_timeout_sec:.1f}s)",
                        flush=True,
                    )
                    return result
                retry_count += 1
            else:
                # This is usually the case when the model getting stuck on one particular test case.
                # For example, timeout error or FC model returning invalid JSON response.
                # Since temperature is already set to 0.001, retrying the same test case will not help.
                # So we continue the generation process and record the error message as the model response
                print("-" * 100)
                print(
                    "❗️❗️ Error occurred during inference. Maximum reties reached for rate limit or other error. Continuing to next test case."
                )
                print(f"❗️❗️ Test case ID: {test_case['id']}, Error: {str(e)}")
                traceback.print_exc()
                print("-" * 100)

                return {
                    "id": test_case["id"],
                    "result": f"Error during inference: {str(e)}",
                }

    result_to_write = {
        "id": test_case["id"],
        "result": result,
    }

    result_to_write.update(metadata)

    return result_to_write

async def async_generate_results(args, handler, test_cases_total):
    max_concurrency = max(1, int(getattr(args, "num_threads", 1) or 1))
    case_timeout_sec = _effective_case_timeout_sec(args)
    case_timeout_retries = _nonnegative_int(
        getattr(args, "case_timeout_retries", 1),
        default=1,
    )
    watchdog_interval_sec = (
        _positive_float_or_none(getattr(args, "watchdog_interval_sec", None)) or 30.0
    )
    stall_fail_fast_sec = (
        _positive_float_or_none(getattr(args, "stall_fail_fast_sec", None)) or 900.0
    )
    stall_fail_fast_min_completed = _nonnegative_int(
        getattr(args, "stall_fail_fast_min_completed", 5),
        default=5,
    )
    consecutive_timeout_fail_fast = _nonnegative_int(
        getattr(args, "consecutive_timeout_fail_fast", 5),
        default=5,
    )
    consecutive_failure_fail_fast = _nonnegative_int(
        getattr(args, "consecutive_failure_fail_fast", consecutive_timeout_fail_fast),
        default=consecutive_timeout_fail_fast,
    )
    semaphore = asyncio.Semaphore(max_concurrency)
    print(
        f"[{_ts()}] BFCL async generation limits: "
        f"max_concurrency={max_concurrency}, "
        f"case_timeout_sec={case_timeout_sec}, "
        f"request_timeout_sec={getattr(args, 'request_timeout_sec', None)}, "
        f"provider_min_request_interval_sec={getattr(args, 'provider_min_request_interval_sec', None)}, "
        f"provider_request_jitter_sec={getattr(args, 'provider_request_jitter_sec', None)}, "
        f"case_timeout_retries={case_timeout_retries}, "
        f"watchdog_interval_sec={watchdog_interval_sec}, "
        f"stall_fail_fast_sec={stall_fail_fast_sec}",
        f"consecutive_failure_fail_fast={consecutive_failure_fail_fast}",
        flush=True,
    )
    in_flight = {}
    progress = {
        "done": 0,
        "timeouts": 0,
        "failures": 0,
        "consecutive_timeouts": 0,
        "consecutive_failures": 0,
        "last_completed_at": time.monotonic(),
        "started_at": time.monotonic(),
    }
    stop_watchdog = asyncio.Event()
    stall_state = {"error": None}

    async def run_one(test_case):
        async with semaphore:
            case_id = _test_case_id(test_case)
            in_flight[case_id] = {
                "started_at": time.monotonic(),
                "case_timeout_retries": 0 if is_multi_turn(case_id) else case_timeout_retries,
            }
            try:
                return await async_inference(
                    handler,
                    test_case,
                    args.include_input_log,
                    args.exclude_state_log,
                    case_timeout_sec=case_timeout_sec,
                    case_timeout_retries=case_timeout_retries,
                )
            finally:
                in_flight.pop(case_id, None)

    async def watchdog(tasks):
        while not stop_watchdog.is_set():
            try:
                await asyncio.wait_for(stop_watchdog.wait(), timeout=watchdog_interval_sec)
                return
            except asyncio.TimeoutError:
                now = time.monotonic()
                in_flight_text = []
                for case_id, state in sorted(in_flight.items()):
                    elapsed = now - state["started_at"]
                    retries = state.get("case_timeout_retries", case_timeout_retries)
                    in_flight_text.append(
                        f"{case_id} ({elapsed:.1f}s, retries={retries})"
                    )
                if not in_flight_text:
                    in_flight_text.append("none")
                print(
                    f"[BFCL watchdog {_ts()}] in-flight: "
                    f"{'; '.join(in_flight_text)}; "
                    f"done {progress['done']}/{len(test_cases_total)}, "
                    f"timeouts {progress['timeouts']}, "
                    f"failures {progress['failures']}, "
                    f"seconds_since_last_done {now - progress['last_completed_at']:.1f}",
                    flush=True,
                )

                if (
                    progress["done"] < stall_fail_fast_min_completed
                    and now - progress["started_at"] > stall_fail_fast_sec
                ):
                    stall_state["error"] = BFCLStalledError(
                        "BFCL stalled: fewer than "
                        f"{stall_fail_fast_min_completed} cases completed in "
                        f"{stall_fail_fast_sec:.0f}s"
                    )
                elif (
                    consecutive_failure_fail_fast > 0
                    and progress["consecutive_failures"] >= consecutive_failure_fail_fast
                ):
                    stall_state["error"] = BFCLStalledError(
                        "BFCL stalled: "
                        f"{progress['consecutive_failures']} consecutive inference failures"
                    )

                if stall_state["error"] is not None:
                    print(f"[{_ts()}] {stall_state['error']}", flush=True)
                    for task in tasks:
                        task.cancel()
                    stop_watchdog.set()
                    return

    tasks = [asyncio.create_task(run_one(test_case)) for test_case in test_cases_total]
    watchdog_task = asyncio.create_task(watchdog(tasks))
    try:
        with tqdm(total=len(test_cases_total), desc=f"Generating results for {args.model_name}") as pbar:
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                except asyncio.CancelledError:
                    if stall_state["error"] is not None:
                        raise stall_state["error"]
                    raise
                handler.write(
                    result, result_dir=args.result_dir, update_mode=True
                )  # Always use update_mode=True to prevent duplicate entries for the same test case
                progress["done"] += 1
                progress["last_completed_at"] = time.monotonic()
                if _is_case_timeout_result(result):
                    progress["timeouts"] += 1
                    progress["consecutive_timeouts"] += 1
                else:
                    progress["consecutive_timeouts"] = 0
                if _retryable_failed_result(result):
                    progress["failures"] += 1
                    progress["consecutive_failures"] += 1
                else:
                    progress["consecutive_failures"] = 0
                pbar.update()
                if (
                    consecutive_failure_fail_fast > 0
                    and progress["consecutive_failures"] >= consecutive_failure_fail_fast
                ):
                    stall_state["error"] = BFCLStalledError(
                        "BFCL stalled: "
                        f"{progress['consecutive_failures']} consecutive inference failures"
                    )
                    for pending_task in tasks:
                        if not pending_task.done():
                            pending_task.cancel()
                    raise stall_state["error"]
                if stall_state["error"] is not None:
                    raise stall_state["error"]
    finally:
        stop_watchdog.set()
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

def generate_results(args, model_name, test_cases_total, handler=None):
    # Always use update_mode=True to prevent duplicate entries for the same test case
    update_mode = True
    handler = handler or build_handler(model_name, args.temperature)

    if handler.model_style == ModelStyle.OSSMODEL or isinstance(handler, OpenRouterHandler):
        if hasattr(handler, "setup_tokenizer"):
            handler.setup_tokenizer(args.local_model_path)
        asyncio.run(async_generate_results(args, handler, test_cases_total))
    else:
        max_workers = max(1, int(getattr(args, "num_threads", 1) or 1))
        case_timeout_sec = _effective_case_timeout_sec(args)
        case_timeout_retries = _nonnegative_int(
            getattr(args, "case_timeout_retries", 1),
            default=1,
        )
        progress_poll_sec = (
            _positive_float_or_none(getattr(args, "progress_poll_sec", None)) or 5.0
        )
        print(
            f"[{_ts()}] BFCL threaded generation limits: "
            f"max_workers={max_workers}, "
            f"case_timeout_sec={case_timeout_sec}, "
            f"request_timeout_sec={getattr(args, 'request_timeout_sec', None)}, "
            f"provider_min_request_interval_sec={getattr(args, 'provider_min_request_interval_sec', None)}, "
            f"provider_request_jitter_sec={getattr(args, 'provider_request_jitter_sec', None)}, "
            f"case_timeout_retries={case_timeout_retries}, "
            f"progress_poll_sec={progress_poll_sec}",
            flush=True,
        )
        pending_cases = iter(test_cases_total)
        requeued_cases = deque()
        timeout_attempts_by_case = {}
        futures = {}
        executor = ThreadPoolExecutor(max_workers=max_workers)

        def submit_next_case():
            if requeued_cases:
                test_case = requeued_cases.popleft()
            else:
                try:
                    test_case = next(pending_cases)
                except StopIteration:
                    return False
            future = executor.submit(
                multi_threaded_inference,
                handler,
                test_case,
                args.include_input_log,
                args.exclude_state_log,
            )
            futures[future] = (test_case, time.monotonic())
            return True

        try:
            for _ in range(max_workers):
                if not submit_next_case():
                    break

            with tqdm(
                total=len(test_cases_total), desc=f"Generating results for {model_name}"
            ) as pbar:
                while futures:
                    done, _ = wait(
                        list(futures.keys()),
                        timeout=progress_poll_sec,
                        return_when=FIRST_COMPLETED,
                    )

                    if not done:
                        if case_timeout_sec is not None:
                            now = time.monotonic()
                            expired = [
                                (future, test_case, now - started_at)
                                for future, (test_case, started_at) in futures.items()
                                if now - started_at > case_timeout_sec
                            ]
                            if expired:
                                expired_futures = {future for future, _, _ in expired}
                                retry_cases = [
                                    test_case
                                    for future, (test_case, _) in futures.items()
                                    if future not in expired_futures
                                ]
                                expired_retry_cases = []
                                for future in futures:
                                    future.cancel()
                                executor.shutdown(wait=False, cancel_futures=True)
                                futures.clear()
                                executor = ThreadPoolExecutor(max_workers=max_workers)

                                for _, test_case, elapsed in expired:
                                    case_id = _test_case_id(test_case)
                                    attempts = timeout_attempts_by_case.get(case_id, 0) + 1
                                    timeout_attempts_by_case[case_id] = attempts
                                    effective_retries = (
                                        0 if is_multi_turn(case_id) else case_timeout_retries
                                    )
                                    if attempts <= effective_retries:
                                        expired_retry_cases.append(test_case)
                                        print(
                                            f"[{_ts()}] BFCL case timed out; retrying: "
                                            f"{case_id} attempt {attempts}/"
                                            f"{effective_retries} "
                                            f"({elapsed:.1f}s > {case_timeout_sec:.1f}s)",
                                            flush=True,
                                        )
                                        continue

                                    result = _build_case_timeout_result(
                                        test_case,
                                        elapsed,
                                        case_timeout_sec,
                                        attempts,
                                        effective_retries,
                                    )
                                    print(
                                        f"[{_ts()}] BFCL case timed out; recording inference "
                                        f"error and continuing: {result['id']} "
                                        f"after {attempts} timed-out attempts "
                                        f"({elapsed:.1f}s > {case_timeout_sec:.1f}s)",
                                        flush=True,
                                    )
                                    handler.write(
                                        result,
                                        result_dir=args.result_dir,
                                        update_mode=True,
                                    )
                                    pbar.update()

                                if retry_cases or expired_retry_cases:
                                    already_requeued = list(requeued_cases)
                                    requeued_cases.clear()
                                    requeued_cases.extend(expired_retry_cases)
                                    requeued_cases.extend(retry_cases)
                                    requeued_cases.extend(already_requeued)

                                while len(futures) < max_workers:
                                    if not submit_next_case():
                                        break
                                continue
                        continue

                    for future in done:
                        test_case, _ = futures.pop(future)
                        case_id = _test_case_id(test_case)
                        try:
                            result = future.result()
                        except Exception as exc:
                            for pending_future in futures:
                                pending_future.cancel()
                            raise RuntimeError(
                                f"BFCL generation failed for test case {case_id}"
                            ) from exc
                        handler.write(
                            result, result_dir=args.result_dir, update_mode=True
                        )  # Always use update_mode=True to prevent duplicate entries for the same test case
                        pbar.update()
                        submit_next_case()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    return handler

def main(args):

    if type(args.test_category) is not list:
        args.test_category = [args.test_category]

    (
        all_test_file_paths,
        all_test_categories,
        all_test_entries_involved,
    ) = get_involved_test_entries(args.test_category, args.run_ids, args.samples_per_category,args.artifacts_path)

    if args.model_name not in MODEL_CONFIG_MAPPING:
        raise ValueError(
                    f"Unknown model_name '{args.model_name}'.\n"
                    "• For officially supported models, please refer to `SUPPORTED_MODELS.md`.\n"
                    "• For running new models, please refer to `README.md` and `CONTRIBUTING.md`."
                )
    print(f"Generating results for {args.model_name}")
    if args.run_ids:
        print("Running specific test cases. Ignoring `--test-category` argument.")
    else:
        print(f"Running full test cases for categories: {all_test_categories}.")

    if args.result_dir is not None:
        args.result_dir = PROJECT_ROOT / args.result_dir
    else:
        args.result_dir = RESULT_PATH

    test_cases_total = collect_test_cases(
        args,
        args.model_name,
        all_test_categories,
        all_test_file_paths,
        all_test_entries_involved,
    )
    handler = build_handler(args.model_name, args.temperature)

    if len(test_cases_total) == 0:
        print(
            f"All selected test cases have been previously generated for {args.model_name}. No new test cases to generate."
        )
    else:
        handler = generate_results(args, args.model_name, test_cases_total, handler=handler)
    
    return handler
