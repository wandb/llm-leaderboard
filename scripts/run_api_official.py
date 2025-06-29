#!/usr/bin/env python3
"""
Official-like inference script adapted from SWE-bench `run_api.py`.
It supports OpenAI (GPT) models and Anthropic Claude models, writes predictions in JSONL,
and extracts diff patches using `extract_diff`.
The goal is to run this script separately to verify patch generation quality without
interfering with the leaderboard pipeline.

Usage example:
    python scripts/run_api_official.py \
        --dataset_name_or_path princeton-nlp/SWE-bench_Verified \
        --split test \
        --model_name_or_path gpt-4-0613 \
        --output_dir ./predictions \
        --model_args temperature=0.2,top_p=0.95

The output file path will be printed at start; you can pass that path to
`swebench.harness.run_evaluation` for scoring.
"""

import json, os, time, traceback, dotenv
from pathlib import Path
from argparse import ArgumentParser
import logging
from typing import List, Dict, Any, Set

import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, load_from_disk

import openai
import tiktoken
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from tenacity import retry, stop_after_attempt, wait_random_exponential

from swebench.inference.make_datasets.utils import extract_diff

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# --- Constants copied from official script ---
MODEL_LIMITS = {
    "gpt-4-0613": 8192,
    "gpt-4-32k-0613": 32768,
    "gpt-3.5-turbo-16k-0613": 16385,
    "gpt-3.5-turbo-1106": 16385,
    "gpt-4-1106-preview": 128000,
    "gpt-4-0125-preview": 128000,
    "claude-2": 100000,
    "claude-instant-1": 100000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "gpt-4.1-2025-04-14": 32768,
    "gpt-4.1": 32768,
}

ENGINES = {
    "gpt-3.5-turbo-16k-0613": "gpt-35-turbo-16k",
    "gpt-4-0613": "gpt-4",
    "gpt-4-32k-0613": "gpt-4-32k",
}

# Simple cost tables (can be extended) ----------------------------------
MODEL_COST_PER_INPUT = {k: 0.0 for k in MODEL_LIMITS}
MODEL_COST_PER_OUTPUT = {k: 0.0 for k in MODEL_LIMITS}

# ----------------------------------------------------------------------

def calc_cost(model_name: str, input_tok: int, output_tok: int) -> float:
    return MODEL_COST_PER_INPUT.get(model_name, 0) * input_tok + MODEL_COST_PER_OUTPUT.get(model_name, 0) * output_tok

# ---------------- OpenAI helpers ----------------

@retry(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))
def call_openai(model: str, system_msg: str, user_msg: str, **chat_kwargs):
    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        **chat_kwargs,
    )
    return resp

# --------------- Anthropic helpers --------------
@retry(wait=wait_random_exponential(min=60, max=600), stop=stop_after_attempt(6))
def call_anthropic(anthropic: Anthropic, model: str, prompt: str, max_tokens: int, **kwargs):
    resp = anthropic.completions.create(model=model, prompt=prompt, max_tokens_to_sample=max_tokens, **kwargs)
    return resp

# --------------- Token utilities ---------------

def gpt_token_len(text: str, enc) -> int:
    return len(enc.encode(text))

def claude_token_len(text: str, api: Anthropic):
    return api.count_tokens(text)

# --------------- Inference main -----------------

def run_openai(dataset, model_name: str, out_path: Path, model_kwargs: Dict[str, Any], max_cost: float | None):
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY missing")

    enc = tiktoken.encoding_for_model(model_name)
    dataset = dataset.filter(lambda x: gpt_token_len(x["text"], enc) <= MODEL_LIMITS[model_name], load_from_cache_file=False)

    temperature = float(model_kwargs.pop("temperature", 0.2))
    top_p = float(model_kwargs.pop("top_p", 0.95 if temperature > 0 else 1))

    total_cost = 0.0
    with out_path.open("a", encoding="utf-8") as f:
        for row in tqdm(dataset, desc=f"OpenAI-{model_name}"):
            system_msg, user_msg = row["text"].split("\n", 1)
            resp = call_openai(model_name, system_msg, user_msg, temperature=temperature, top_p=top_p, **model_kwargs)
            completion = resp.choices[0].message.content
            input_tok = resp.usage.prompt_tokens
            out_tok = resp.usage.completion_tokens
            total_cost += calc_cost(model_name, input_tok, out_tok)

            data = {
                "instance_id": row["instance_id"],
                "model_name_or_path": model_name,
                "full_output": completion,
                "model_patch": extract_diff(completion),
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            if max_cost and total_cost >= max_cost:
                logger.info("Reached max cost limit, exiting.")
                break


def run_anthropic(dataset, model_name: str, out_path: Path, model_kwargs: Dict[str, Any], max_cost: float | None):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    anthropic = Anthropic(api_key=api_key)
    dataset = dataset.filter(lambda x: claude_token_len(x["text"], anthropic) <= MODEL_LIMITS[model_name], load_from_cache_file=False)

    temperature = float(model_kwargs.pop("temperature", 0.2))
    top_p = float(model_kwargs.pop("top_p", 0.95 if temperature > 0 else 1))

    total_cost = 0.0
    with out_path.open("a", encoding="utf-8") as f:
        for row in tqdm(dataset, desc=f"Anthropic-{model_name}"):
            prompt = f"{HUMAN_PROMPT} {row['text']}\n\n{AI_PROMPT}"
            resp = call_anthropic(anthropic, model_name, prompt, 4096, temperature=temperature, top_p=top_p, **model_kwargs)
            completion = resp.completion
            input_tok = anthropic.count_tokens(prompt)
            out_tok = anthropic.count_tokens(completion)
            total_cost += calc_cost(model_name, input_tok, out_tok)
            data = {
                "instance_id": row["instance_id"],
                "model_name_or_path": model_name,
                "full_output": completion,
                "model_patch": extract_diff(completion),
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            if max_cost and total_cost >= max_cost:
                logger.info("Reached max cost limit, exiting.")
                break


# -------------------- CLI -----------------------

def parse_args():
    p = ArgumentParser()
    p.add_argument("--dataset_name_or_path", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--model_name_or_path", required=True, choices=list(MODEL_LIMITS))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_args", default=None, help="k=v,k2=v2")
    p.add_argument("--max_cost", type=float, default=None)
    p.add_argument("--max_instances", type=int, default=None, help="Process only the first N instances after filtering")
    return p.parse_args()


def main():
    args = parse_args()
    model_kwargs = {}
    if args.model_args:
        for kv in args.model_args.split(","):
            k, v = kv.split("=")
            model_kwargs[k] = v

    if Path(args.dataset_name_or_path).exists():
        dset = load_from_disk(args.dataset_name_or_path)[args.split]
    else:
        dset = load_dataset(args.dataset_name_or_path, split=args.split)

    # limit instances if requested
    if args.max_instances is not None and args.max_instances > 0:
        dset = dset.select(range(min(args.max_instances, len(dset))))
        logger.info(f"Keeping only first {len(dset)} instances due to --max_instances")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{Path(args.model_name_or_path).name}__{args.dataset_name_or_path.split('/')[-1]}__{args.split}.jsonl"
    logger.info(f"Writing predictions to {out_file}")

    if args.model_name_or_path.startswith("gpt"):
        run_openai(dset, args.model_name_or_path, out_file, model_kwargs, args.max_cost)
    elif args.model_name_or_path.startswith("claude"):
        run_anthropic(dset, args.model_name_or_path, out_file, model_kwargs, args.max_cost)
    else:
        raise ValueError("Unsupported model")

    logger.info("Done")

if __name__ == "__main__":
    main() 