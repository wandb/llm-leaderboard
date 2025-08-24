# SWE-Bench Evaluation Guide (Nejumi LLM Leaderboard 4)

**TL;DR**

* **What do we measure?**
  For real OSS “bug-fixing tasks,” the model generates a unified diff. We apply it and automatically grade based on whether all official tests pass.
* **Verified subset**
  We use a stable subset with curated inputs, environments, and expected tests. From the Japanese version `nejumi/swe-bench-verified-ja`, we sample 80 instances under 7,000 input tokens.
* **Most critical output constraint**
  The unified diff must include line numbers in the @@ hunk headers (`@@ -start,count +start,count @@`). Missing them causes patch application to fail.

---

## 1. What is SWE-Bench?

* **Overview**: An end-to-end benchmark based on real bug fixes in OSS repositories.
  Input (Issue/PR context, relevant file snippets, reproduction tests, etc.) → Output (the unified diff patch).
  After applying the patch and running tests, the instance is considered Resolved if all tests pass.

* **Verified**:
  A stable subset with curated inputs, environments, and expected tests for higher reproducibility and grading reliability.

---

## 2. How we build this 80-instance evaluation set

* **Objective**: Make it feasible for local LLMs (~8k context) by ensuring inputs are under 7,000 tokens, while preserving the original (500 instances) difficulty distribution and staying close to GPT-4.1’s resolved ratio.

* **Base**: `nejumi/swe-bench-verified-ja` (500 instances, `problem_statement` and `hint_text` translated into Japanese with Qwen/Qwen3-235B-A22B)

* **Filter**: Approximate token counts using `hf-internal-testing/llama-tokenizer` (equivalent to Llama 2), and keep only `num_tokens < 7000`.

* **Sampling**: Stratified by `(difficulty, status)`

  * `difficulty`: 4 levels
  * `status`: resolved / not resolved

* **Prompt shaping**:
  We embed a CRITICAL sentence before the string "I need you to solve the provided issue" within `text`, enforcing "line numbers required in @@ hunk header".

* **Distribution**: W&B Artifact (Arrow format)

  * Name: `swebench_verified_official` (versioned)
  * This evaluation uses the 80-instance subset

---

## 3. Distribution comparison (Original 500 vs 80)

| Metric            | Original (500) | 80 (<7k) |
| ----------------- | -------------: | -------: |
| <15 min fix       |          38.8  |     46.2 |
| 15 min – 1 hour   |          52.2  |     50.0 |
| 1–4 hours         |           8.4  |      3.8 |
| >4 hours          |           0.6  |      0.0 |
| Resolved rate     |          34.6  |     37.5 |

**Notes**: Slight increase in short fixes (<15 min), and reductions in 1–4 hours and >4 hours.
The **Resolved rate** is similar (34.6% → 37.5%), maintaining overall trends well.

---

## 4. End-to-end evaluation flow (overview)

```
┌────────────────────────────┐
│ Dataset (80 items)         │
└───────────────┬────────────┘
                │ Input (Issue/PR context, relevant snippets,
                │         reproduction/expected tests, constraints)
                ▼
┌────────────────────────────────────────┐
│ Generation (LLM)                       │
│  - Prompt shaping (CRITICAL sentence)  │
│  - fc_enabled: enforce unified diff    │
└───────────────────┬────────────────────┘
                    │ Unified diff (line numbers in @@ required)
                    ▼
┌────────────────────────────────────────┐
│ Preprocessing (expansion & normalization) │
│  - Extract minimal patch                │
│  - Hunk header expansion                │
│  - Filename normalization / merge dups  │
└───────────────────┬────────────────────┘
                    │ Apply patch (git apply / patch --fuzz)
                    ▼
┌────────────────────────────────────────┐
│ Evaluation runner                      │
│  - Official or Docker                  │
│  - Prebuilt images optional            │
│  - Run unit tests                      │
└───────────────────┬────────────────────┘
                    │ Pass/Fail
                    ▼
            Resolved / Not Resolved
                    ↓
             SWE-Bench Score
           (Resolved rate = pass rate)
```

---

## 5. I/O specification (example)

### Input

* Task description (Japanese), relevant file snippets, reproduction/expected tests, constraints
* Evaluation meta (e.g., `evaluation_method: official|docker`, `prebuild_images: true`)

### Output (unified diff)

```diff
--- a/astropy/modeling/separable.py
+++ b/astropy/modeling/separable.py
@@ -245,1 +245,1 @@
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right
```

⚠️ **CRITICAL**

* **Line numbers are required in the @@ hunk header**
* Follow the format `@@ -start,count +start,count @@`

---

## 6. Execution modes and environment

* **Execution environment**: Runs on a fixed Docker image
* **Generation control**: Adjust temperature and max tokens with `generator.*`
* **fc_enabled: true** enforces a unified diff output format

---

## 7. Quickstart

### Dependencies

```bash
./myenv/bin/pip install fastapi "uvicorn[standard]" swebench
```

### Start the API server

```bash
nohup ./myenv/bin/python scripts/server/swebench_server.py \
  --host 0.0.0.0 --port 8000 \
  >/tmp/swebench_server.out 2>&1 & disown
```

### Submit a job (example)

```bash
PATCH_FILE=patch.diff
INSTANCE_ID=astropy__astropy-12907
curl -s -H "Content-Type: application/json" \
     -H "X-API-Key: $SWE_API_KEY" \
     -d @<(jq -n --arg iid "$INSTANCE_ID" --arg patch "$(cat "$PATCH_FILE")" \
       '{instance_id:$iid, patch_diff:$patch, namespace:"swebench", tag:"latest", model_name_or_path:"nejumi-api"}') \
     http://127.0.0.1:8000/v1/jobs
```

---

## 8. Key configuration items

```yaml
swebench:
  artifacts_path: llm-leaderboard/nejumi-leaderboard4/swebench_verified_official:production
  dataset_dir: swebench_verified_official
  max_samples: 80
  max_tokens: 2048
  max_workers: 4
  evaluation_method: docker
  prebuild_images: true
  fc_enabled: true
  api_server:
    enabled: false
```

---

## 9. Extensions beyond the original SWE-Bench

* **Hunk header expansion**

  * Increase `pre_len` / `post_len` in hunk headers by `ctx*2` (default ctx=5)
  * We do not insert extra context lines into the patch body; we only increase counts to relax application tolerance
  * If `git apply` fails, we progressively try `patch --fuzz=10` / `--fuzz=20`

* **Minimal patch extraction**

  * Remove extraneous text and irrelevant diffs; recompute header line numbers
  * Reduce changes to the minimum necessary for tests to pass

* **CRITICAL sentence insertion**

  * Explicitly enforce "line numbers required in @@ hunk header" to stabilize outputs

---

## 10. Best practices

* **Output only the diff** (no explanatory text mixed in)
* **Line numbers are mandatory in @@ headers**
* **Minimal patch principle**
* Enable `fc_enabled: true`

---

## 11. FAQ

* **Q. What determines Resolved?**
  → After applying the generated patch, the instance is Resolved if all tests pass.

* **Q. Why hunk header expansion?**
  → To allow `git apply` to succeed despite minor drifts or whitespace differences.
  → We increase only the line counts in the header, not the number of actual context lines in the body.

* **Q. When do we use fuzz?**
  → We try `patch --fuzz=10,20` if `git apply` fails.

