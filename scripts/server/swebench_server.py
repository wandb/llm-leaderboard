#!/usr/bin/env python3
"""
Minimal SWE-Bench Scoring API Server (single-node)

Endpoints:
- POST /v1/jobs: submit a scoring job for one instance_id with a patch
- GET  /v1/jobs/{job_id}: get status and (if finished) result summary
- GET  /v1/jobs/{job_id}/logs: get simple log text (best-effort)

Security:
- Optional API key via header: X-API-Key. If env SWE_API_KEY is set, it is enforced.

Notes:
- This server loads the instance from HuggingFace dataset (SWE-bench Verified test split)
- It runs the official harness (swebench.harness.run_evaluation) on a temp dataset
- Requires Docker daemon access on this host
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# ------------------------------
# Models
# ------------------------------

class JobCreate(BaseModel):
    instance_id: str = Field(..., description="SWE-bench Verified instance_id")
    patch_diff: str = Field(..., description="Unified diff to apply (model_patch)")
    namespace: str = Field("swebench", description="Docker registry namespace")
    tag: str = Field("latest", description="Instance image tag")
    timeout_sec: int = Field(1800, description="Evaluation timeout in seconds")
    model_name_or_path: str = Field(
        default="nejumi-api",
        description="Identifier recorded in predictions (used by official harness)",
    )


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


@dataclass
class InternalJob:
    job_id: str
    req: JobCreate
    created_at: float = field(default_factory=lambda: time.time())
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    status: str = "queued"  # queued | running | finished | failed
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    log_buffer: list[str] = field(default_factory=list)
    work_dir: Optional[Path] = None


# ------------------------------
# Auth
# ------------------------------

def get_api_key(x_api_key: Optional[str] = Header(default=None)) -> Optional[str]:
    return x_api_key


def require_api_key(x_api_key: Optional[str] = Depends(get_api_key)) -> None:
    load_dotenv()
    expected = os.getenv("SWE_API_KEY")
    if expected:
        if not x_api_key or x_api_key != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")


# ------------------------------
# App & State
# ------------------------------

app = FastAPI(title="SWE-Bench Scoring API", version="0.1.0")

JOB_STORE: dict[str, InternalJob] = {}
JOB_QUEUE: "asyncio.Queue[str]" = asyncio.Queue()


# ------------------------------
# Utilities
# ------------------------------

def _log(job: InternalJob, msg: str) -> None:
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    job.log_buffer.append(f"[{ts}] {msg}")


def _format_instance_id_for_image(instance_id: str) -> str:
    return instance_id.replace("__", "_1776_").lower()


def _load_verified_instance(instance_id: str) -> Dict[str, Any]:
    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    for item in ds:
        if item["instance_id"] == instance_id:
            return dict(item)
    raise KeyError(f"instance_id not found: {instance_id}")


def _run_single_evaluation(job: InternalJob) -> Dict[str, Any]:
    """Run official harness for a single instance using a temp dataset+pred file."""
    req = job.req
    job.started_at = time.time()
    _log(job, "Loading instance from HF dataset...")
    instance = _load_verified_instance(req.instance_id)

    # Prepare temp working directory
    work_dir = Path(tempfile.mkdtemp(prefix=f"swebench_job_{job.job_id}_"))
    job.work_dir = work_dir
    dataset_file = work_dir / "eval_dataset.jsonl"
    predictions_file = work_dir / "predictions.jsonl"

    # Write single-instance dataset
    with open(dataset_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "repo": instance["repo"],
            "instance_id": instance["instance_id"],
            "base_commit": instance["base_commit"],
            "patch": instance["patch"],
            "test_patch": instance["test_patch"],
            "problem_statement": instance["problem_statement"],
            "hints_text": instance.get("hints_text", ""),
            "created_at": instance["created_at"],
            "version": instance["version"],
            "FAIL_TO_PASS": instance["FAIL_TO_PASS"],
            "PASS_TO_PASS": instance["PASS_TO_PASS"],
            "environment_setup_commit": instance.get("environment_setup_commit", ""),
        }, ensure_ascii=False) + "\n")

    # DEBUG: Temporarily disable patch processing to identify the issue
    patch_text = req.patch_diff
    _log(job, f"Original patch: {repr(patch_text[:200])}")
    
    # Just ensure 'diff --git' header exists (without any other processing)
    if patch_text and "diff --git" not in patch_text:
        lines = patch_text.splitlines()
        a_line = next((ln for ln in lines if ln.startswith("--- ")), None)
        b_line = next((ln for ln in lines if ln.startswith("+++ ")), None)
        if a_line and b_line:
            try:
                a_path = a_line.split()[1]
                b_path = b_line.split()[1]
                header = f"diff --git {a_path} {b_path}\n"
                patch_text = header + patch_text
                _log(job, "Inserted missing 'diff --git' header")
            except Exception:
                pass
    
    # Ensure trailing newline
    if patch_text and not patch_text.endswith('\n'):
        patch_text = patch_text + '\n'
        _log(job, "Added trailing newline to patch")

    # Write predictions (model patch)
    with open(predictions_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "instance_id": req.instance_id,
            "model_name_or_path": req.model_name_or_path,
            "model_patch": patch_text,
        }, ensure_ascii=False) + "\n")

    _log(job, "Starting official harness evaluation...")
    from swebench.harness.run_evaluation import main as run_evaluation

    run_id = f"api_{int(time.time())}"
    formatted = _format_instance_id_for_image(req.instance_id)
    namespace = req.namespace
    image_tag = req.tag

    # Run evaluation for one instance
    res = run_evaluation(
        dataset_name=str(dataset_file),
        split="train",
        instance_ids=[req.instance_id],
        predictions_path=str(predictions_file),
        max_workers=1,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id=run_id,
        timeout=req.timeout_sec,
        namespace=namespace,
        rewrite_reports=False,
        modal=False,
        instance_image_tag=image_tag,
        report_dir=str(work_dir),
    )

    # Normalize result into a dict
    result_dict: Dict[str, Any]
    try:
        if isinstance(res, dict):
            result_dict = res
        elif isinstance(res, (str, Path)):
            result_dict = {"report_path": str(res)}
        else:
            result_dict = {"result_repr": repr(res)}
    except Exception:
        result_dict = {"result_repr": repr(res)}

    # Attach helpful references
    result_dict["work_dir"] = str(work_dir)
    result_dict["image"] = f"{namespace}/sweb.eval.x86_64.{formatted}:{image_tag}"
    # Best-effort: log a short summary if possible
    try:
        report_path = result_dict.get("report_path")
        if report_path and Path(report_path).exists():
            with open(report_path, "r", encoding="utf-8") as rf:
                rep = json.load(rf)
            iid = req.instance_id
            resolved_ids = set(rep.get("resolved_ids", []))
            is_resolved = iid in resolved_ids
            _log(job, f"Result for {iid}: resolved: {is_resolved}")
    except Exception:
        pass
    return result_dict


# NOTE: expand_hunk_headers is removed to maintain compatibility with local evaluator


def _fix_split_headers(patch: str) -> str:
    """Join broken file header lines where the path got split by newline.

    Mirrors evaluator behavior to increase robustness.
    """
    if not patch:
        return patch
    lines = patch.split("\n")
    fixed_lines: list[str] = []
    i = 0
    header_start = ("--- ", "+++ ")
    while i < len(lines):
        line = lines[i]
        if line.startswith(header_start):
            if not re.search(r"\.[a-zA-Z0-9]+$", line) and (i + 1) < len(lines):
                j = i + 1
                joined = line
                while j < len(lines):
                    next_line = lines[j]
                    if next_line.startswith(("@@ ", "diff ", "--- ", "+++ ")):
                        break
                    joined += next_line.strip("\n")
                    j += 1
                fixed_lines.append(joined)
                i = j
                continue
        fixed_lines.append(line)
        i += 1
    return "\n".join(fixed_lines)


# --- extend patch apply commands in official harness to allow higher fuzz ---
try:
    from swebench.harness.run_evaluation import GIT_APPLY_CMDS  # type: ignore
    _EXTRA_CMDS = [
        "patch --batch --fuzz=10 -p1 -i",
        "patch --batch --fuzz=20 -p1 -i",
    ]
    for _cmd in _EXTRA_CMDS:
        if _cmd not in GIT_APPLY_CMDS:
            GIT_APPLY_CMDS.append(_cmd)
except Exception:
    pass


async def worker_loop():
    """並列実行用のワーカー。
    注意: 評価処理は同期関数なので、イベントループをブロックしないようスレッドにオフロードする。
    """
    while True:
        job_id = await JOB_QUEUE.get()
        job = JOB_STORE.get(job_id)
        if not job:
            JOB_QUEUE.task_done()
            continue
        job.status = "running"
        try:
            # ブロッキングな評価処理をスレッドにオフロード
            res = await asyncio.to_thread(_run_single_evaluation, job)
            job.result = res
            job.status = "finished"
            job.finished_at = time.time()
            _log(job, "Evaluation finished")
        except Exception:
            job.status = "failed"
            job.finished_at = time.time()
            err = traceback.format_exc()
            job.error = err
            _log(job, f"Evaluation failed: {err}")
        finally:
            JOB_QUEUE.task_done()


@app.on_event("startup")
async def _startup():
    # Start N workers (env SWE_WORKERS)
    try:
        n_workers = int(os.environ.get("SWE_WORKERS", "1"))
    except Exception:
        n_workers = 1
    n_workers = max(1, min(n_workers, 32))
    for _ in range(n_workers):
        asyncio.create_task(worker_loop())


# ------------------------------
# Routes
# ------------------------------

@app.post("/v1/jobs", response_model=JobStatus, dependencies=[Depends(require_api_key)])
async def create_job(req: JobCreate):
    job_id = f"job_{int(time.time() * 1000)}"
    job = InternalJob(job_id=job_id, req=req)
    JOB_STORE[job_id] = job
    await JOB_QUEUE.put(job_id)
    return JobStatus(
        job_id=job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@app.get("/v1/jobs/{job_id}", response_model=JobStatus, dependencies=[Depends(require_api_key)])
async def get_job(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatus(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        result=job.result,
    )


@app.get("/v1/jobs/{job_id}/logs", response_class=PlainTextResponse, dependencies=[Depends(require_api_key)])
async def get_logs(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return "\n".join(job.log_buffer)


@app.get("/v1/jobs/{job_id}/report", response_class=JSONResponse, dependencies=[Depends(require_api_key)])
async def get_report(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job or not job.result:
        raise HTTPException(404, "Job not found or not finished")
    report_path = (job.result or {}).get("report_path")
    if not report_path:
        raise HTTPException(404, "Report not available")
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(404, "Report file not found on server")


@app.get("/v1/summary", response_class=JSONResponse)
async def summary() -> Dict[str, Any]:
    # cluster-wide simple summary
    return {
        "status": "ok",
        "workers": int(os.environ.get("SWE_WORKERS", "1")),
        "jobs": {
            "queued": sum(1 for j in JOB_STORE.values() if j.status == "queued"),
            "running": sum(1 for j in JOB_STORE.values() if j.status == "running"),
            "finished": sum(1 for j in JOB_STORE.values() if j.status == "finished"),
            "failed": sum(1 for j in JOB_STORE.values() if j.status == "failed"),
        },
        "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def main():
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    # Import string経由ではなく、アプリインスタンスを直接渡して起動
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()

