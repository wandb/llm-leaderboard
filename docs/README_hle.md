# HLE-JA Evaluation Guide

**TL;DR**

*   **What does it measure?** A benchmark designed to test the limits of AI capabilities, containing tasks that current AI models find difficult to solve.
*   **Evaluation Set**: The Japanese version of "Humanity's Last Exam" (HLE). Based on `hle-ja.csv`, stratified by category and split into training, development, and test sets (8:1:1).
*   **Output Format**: The model is required to generate text containing three elements: "Explanation," "Answer," and "Confidence."

---

## 1. What is HLE-JA?

*   **Overview**: HLE-JA is the Japanese version of the benchmark "Humanity's Last Exam" (HLE), created by the Center for AI Safety and Scale AI. It was developed as a more challenging evaluation because existing benchmarks like MMLU have become too easy for high-performing LLMs.
*   **Characteristics**: The questions are deliberately designed to be unanswerable via simple web searches. The original dataset includes 2,500 questions spanning a wide range of domains, about 14% of which are multimodal (text and images). Even top models currently score as low as 3-14%, illustrating its high difficulty level.

---

## 2. How the Evaluation Set is Created

*   **Objective**: To create a challenging benchmark for Japanese language models, based on the HLE framework.
*   **Source File**: `hle-ja.csv`
*   **Process**:
    1.  Stratified sampling of data based on the `category` column.
    2.  Split into three sets: Training (80%), Development (10%), and Test (10%).
    3.  Save as `train.jsonl`, `dev.jsonl`, and `test.jsonl`.
*   **Distribution Format**: W&B Artifact (`wandb.Artifact`)
    *   **Name**: `hle-ja` (version-controlled)
    *   **Upload Script**: `scripts/data_uploader/upload_hle.py`

---

## 3. Breakdown of Question Categories

The original HLE dataset consists of the following categories. The sampling for `hle-ja` maintains the same category proportions.

| Category                  | Count  |
| ------------------------- | -----: |
| Biology/Medicine          |    280 |
| Chemistry                 |    165 |
| Computer Science/AI       |    241 |
| Engineering               |    111 |
| Humanities/Social Science |    219 |
| Math                      |  1,021 |
| Other                     |    233 |
| Physics                   |    230 |
| **Total**                 | **2500** |

---

## 4. Comparison Between Original HLE and Japanese Version

`hle-ja` is an adaptation of the original HLE for evaluating Japanese LLMs. The main differences are:

| Item              | Details                                                                                                |
| ----------------- | ------------------------------------------------------------------------------------------------------ |
| **Translation**   | All questions and options were machine-translated into Japanese using the `qwen/qwen3-235b-a22b` model. |
| **Content Changes** | Multimodal questions containing images in the original dataset are **excluded** from this evaluation. |
| **Sampling**      | Sampled from text-only questions while maintaining the original category ratios.                       |

---

## 5. Evaluation Flow

1.  **Dataset Retrieval**: Download the specified version of the `hle-ja` dataset from W&B Artifacts.
2.  **Answer Generation**: For each question in the `dev` or `test` set, the target model generates an answer. The prompt instructs the model to output in a format containing "Explanation," "Answer," and "Confidence."
3.  **Judgment (Judge)**: The generated answers are evaluated by an independent judge model (e.g., `o3-mini-2025-01-31`). The judge compares the model's answer with the reference answer and outputs structured data including `correct: "yes" or "no"`.
4.  **Metric Calculation**: Based on all judgments, final scores such as accuracy (`accuracy`) and calibration error (`calibration_error`) are calculated.
5.  **Result Logging**: Final scores and detailed results for each question (including generated answers and reasoning for judgments) are recorded as W&B tables.

---

## 6. Input/Output Specifications (Examples)

### Input (Dataset)

*   **Format**: JSONL
*   **Key Fields**: `question` (question text), `answer` (correct answer), `category`, etc.
*   **Note**: Some questions may include images, but the evaluation script will warn if used on models that do not support images.

### Output (Target Model)

```text
Explanation: {Explanation for the chosen answer}
Answer: {Chosen answer}
Confidence: {Confidence score from 0% to 100%}
```

### Output (Judge Model)

The judge model analyzes the target model's response and outputs a JSON containing:

```json
{
  "extracted_final_answer": "Extracted final answer",
  "reasoning": "Reason for judgment compared to the correct answer",
  "correct": "yes" or "no",
  "confidence": "Extracted confidence score"
}
```

---

## 7. Execution and Environment

*   **Evaluation Script**: Entry point is the `evaluate()` function in `scripts/evaluator/hle.py`.
*   **Asynchronous Processing**: Evaluation is executed asynchronously using `asyncio`, parallelizing model requests and judgment tasks.
*   **Inference Control**: Model parameters like temperature and token limit are configured under `generator.*`.

---

## 8. Key Configuration Items

Configurations are managed in a YAML file under the `hle` key:

```yaml
hle:
  artifact_path: "path/to/your/wandb_artifact:version"  # Path to the dataset artifact used for evaluation
  max_samples: 100                                    # Maximum number of samples to evaluate (applied to dev set)
  judge:
    model: "o3-mini-2025-01-31"                       # Judge model used for evaluation
    parallel: 32                                      # Parallelism for judgment tasks
  generator_config:                                   # Parameters for answer generation
    temperature: 0.7
    max_tokens: 1024
```

---

## 9. FAQ

*   **Q. How is correctness determined?**
    → An independent **judge model** compares the target model's answer to the correct answer and returns `yes` or `no`. This approach helps absorb variations in wording and provides more human-like evaluation.

*   **Q. What is `calibration_error`?**
    → It measures how closely the model's reported confidence aligns with its actual accuracy. A lower value indicates the model's confidence estimates are more reliable.
