import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def http_json(method: str, url: str, body_obj: Optional[Dict] = None, headers: Optional[Dict] = None, timeout: float = 300.0) -> Dict:
    data = None
    if body_obj is not None:
        data = json.dumps(body_obj).encode("utf-8")
    req = Request(url=url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            text = resp.read().decode(charset)
            return json.loads(text) if text else {}
    except HTTPError as e:
        try:
            err_text = e.read().decode("utf-8")
        except Exception:
            err_text = str(e)
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {err_text}")
    except URLError as e:
        raise RuntimeError(f"URL Error: {e}")


def infer_is_swebench(row: Dict[str, str]) -> bool:
    iid = (row.get("instance_id") or "").strip()
    img = (row.get("image_name") or "").strip()
    patch = (row.get("patch") or "").strip()
    # Heuristics: has instance_id with repo__issue, or image_name from swebench registry, and has a patch
    return ("__" in iid or img.startswith("swebench/")) and bool(patch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay SWE-Bench CSV via API server and compare results")
    parser.add_argument("--csv", required=True, help="Path to wandb-exported CSV containing instance_id, patch, status")
    parser.add_argument("--endpoint", default=os.getenv("SWE_API_ENDPOINT", "http://127.0.0.1:8000"), help="API server base URL")
    parser.add_argument("--api-key", default=os.getenv("SWE_API_KEY"), help="API key for X-API-Key header")
    parser.add_argument("--namespace", default=os.getenv("SWE_NAMESPACE", "swebench"))
    parser.add_argument("--tag", default=os.getenv("SWE_IMAGE_TAG", "latest"))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("SWE_JOB_TIMEOUT", "1200")), help="Job timeout seconds when polling")
    parser.add_argument("--max", type=int, default=0, help="Limit number of rows to replay (0 = no limit)")
    parser.add_argument("--out", default="replay_results.csv", help="Output CSV path")
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    headers = {}
    if args.api_key:
        headers["X-API-Key"] = args.api_key

    # Read CSV
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if infer_is_swebench(row):
                rows.append(row)
    if args.max > 0:
        rows = rows[: args.max]

    print(f"Found {len(rows)} SWE-Bench-like rows to replay")
    if not rows:
        return

    # Prepare output
    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "instance_id", "orig_status", "api_status", "match", "job_id", "report_path"
        ])

        for idx, row in enumerate(rows, start=1):
            instance_id = (row.get("instance_id") or "").strip()
            patch = (row.get("patch") or "").strip()
            orig_status = (row.get("status") or "").strip()

            if not instance_id or not patch:
                continue

            payload = {
                "instance_id": instance_id,
                "patch_diff": patch,
                "namespace": args.namespace,
                "tag": args.tag,
                "model_name_or_path": "nejumi-replay-csv",
            }
            try:
                job = http_json("POST", f"{endpoint}/v1/jobs", body_obj=payload, headers=headers, timeout=60)
            except Exception as e:
                print(f"[{idx}/{len(rows)}] {instance_id}: submit failed: {e}")
                writer.writerow([instance_id, orig_status, "submit_error", False, "", ""])
                continue

            job_id = job.get("job_id")
            print(f"[{idx}/{len(rows)}] {instance_id}: submitted as {job_id}")

            # Poll
            start = time.time()
            api_status = ""
            report_path = ""
            while True:
                time.sleep(2)
                try:
                    j = http_json("GET", f"{endpoint}/v1/jobs/{job_id}", headers=headers, timeout=60)
                except Exception as e:
                    print(f"  poll error: {e}")
                    if time.time() - start > args.timeout:
                        api_status = "poll_error"
                        break
                    else:
                        continue
                st = j.get("status")
                if st in {"finished", "failed"}:
                    api_status = st
                    res = j.get("result") or {}
                    report_path = res.get("report_path") or ""
                    break
                if time.time() - start > args.timeout:
                    api_status = "timeout"
                    break

            # Classify per report JSON
            final_class = api_status
            if api_status == "finished" and report_path:
                try:
                    with open(report_path, "r", encoding="utf-8") as rf:
                        rep = json.load(rf)
                    iid = instance_id
                    if rep.get("error_instances") == 1 or iid in (rep.get("error_ids") or []):
                        final_class = "error"
                    elif iid in (rep.get("resolved_ids") or []):
                        final_class = "resolved"
                    elif iid in (rep.get("unresolved_ids") or []):
                        final_class = "unresolved"
                    elif iid in (rep.get("empty_patch_ids") or []):
                        final_class = "empty_patch"
                    else:
                        # fallback: treat as error when unrecognized
                        final_class = "error"
                except Exception as e:
                    print(f"  report parse error: {e}")
                    final_class = "report_error"

            match = (orig_status.lower() == final_class.lower()) if orig_status else ""
            writer.writerow([instance_id, orig_status, final_class, match, job_id, report_path])

    print(f"Replay finished. Results saved to: {out_path}")


if __name__ == "__main__":
    main()