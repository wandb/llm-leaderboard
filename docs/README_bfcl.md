# BFCL Evaluation Guide (Nejumi LLM Leaderboard 4 Edition)

**TL;DR**

BFCL (Berkeley Function Calling Leaderboard) is a comprehensive benchmark for evaluating LLM function calling capabilities. Nejumi Leaderboard uses **BFCL v3 with Japanese localization** featuring:

🚀 **Unified OSS Handlers**: `unified-oss-fc` (FC-compatible) and `unified-oss-jsonschema` (FC-incompatible) greatly simplify OSS model evaluation

📊 **Evaluation Scope**: Function calling from single-turn to multi-turn/multi-step (function selection, parallel execution, multi-step reasoning, etc.). However, Nejumi Leaderboard excludes parallel categories

⚙️ **Configuration**: Simply specify `bfcl_model_id` in model config (see [SUPPORTED_MODELS.md](../scripts/evaluator/evaluate_utils/bfcl_pkg/SUPPORTED_MODELS.md) for details)

---

## 1. What is BFCL

**BFCL (Berkeley Function Calling Leaderboard)** is a comprehensive benchmark for evaluating LLM function calling capabilities. It measures a wide range of function calling skills from single-turn to multi-turn and real-world scenarios.

### BFCL Version Evolution

| Version | Key Features | Evaluation Focus |
|---------|--------------|------------------|
| **V1** | Expert-curated single-turn function calling | Accuracy verification through AST (Abstract Syntax Tree) evaluation |
| **V2** | Large-scale collection of real function documentation and queries | Enhanced resistance to bias and data contamination, real-world diversity |
| **V3** ⭐ | **Multi-turn/Multi-step** function calling introduction | Continuous function calling and state-based verification |
| **V4** | Transition to agent-like capability evaluation | Web search, memory management, format robustness |

⭐ **Currently Used**: Nejumi Leaderboard uses BFCL v3 with Japanese localization

### Evaluation Targets and Criteria

BFCL evaluates the following multifaceted skill sets:

#### 🎯 **Basic Function Calling Capabilities**
- **Function Selection Ability** (Multiple, Parallel Multiple): Selecting appropriate functions from multiple options
- **Parallel Calling Ability** (Parallel): Simultaneous execution of multiple functions
- **Syntax Accuracy**: Function calling with correct arguments and types

#### 🧠 **Advanced Judgment and Reasoning Abilities**
- **Relevance Detection**: Judgment ability to identify inappropriate functions and avoid calling them
- **Multi-turn Support**: Context maintenance and appropriate function selection in continuous dialogue
- **Multi-step Planning**: Logical function calling planning across multiple steps

#### 🔄 **Agent-like Capabilities** (Extended in V4, not yet in V3)
- **Multi-stage Reasoning**: Handling multi-hop questions in web search
- **State Management**: Memory access and read/write capabilities
- **Format Robustness**: Adaptability to function, prompt, and format changes

### Reference Links

- 📊 **[BFCL Function-calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)**: Latest evaluation results and model performance comparison
- 📝 **[BFCL V3 Blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)**: Detailed explanation of multi-turn/multi-step features

---

## 2. Creating Japanese Evaluation Dataset

BFCL(v3) problems come in several types. For details, refer to the BFCL [blog](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html).

- **Translation**
    - Base translation using qwen/qwen3-235b-a22b with manual corrections
    - Used llm-leaderboard/scripts/translation/bfcl_translation.py
    - **Rule**: Function names and code-related content are excluded from translation
- **Extraction**
    - Extracted problems with 3 turns or fewer to enable evaluation of models with small max tokens
        - Used llm-leaderboard/scripts/translation/bfcl_multi_turn_count.py to calculate turn counts
        - Used llm-leaderboard/scripts/translation/sort_bfcl_file.py for sorting
    - Randomly extracted 30 problems
        - Randomized to avoid similar problems when extracting 30 problems in order of smallest ID
        - All problems used if fewer than 30 problems available
        - For categories with 50+ problems, limited to 50 for quality assurance of manual translation verification, saved to W&B artifacts [link](https://wandb.ai/llm-leaderboard/nejumi-leaderboard4/artifacts/dataset/bfcl)
    - Removed parallel problems
        - Removed parallel problems as some models don't support parallel processing
- **Other Detailed Processing**
    - Removed problems containing English questions in problem text
    - Randomly extracted 30 problems per category (used all problems if fewer than 30)
        - BFCL_v3_live_irrelevance.json (882→30 problems)
        - BFCL_v3_irrelevance.json (240→30 problems)
        - BFCL_v3_simple.json (400→30 problems)
        - And more categories...
    - Added Japanese options to possible answers
    - Added "Answer in English" instruction to problems requiring language specification

---

## 4. Extensions from Original BFCL

This section provides detailed explanations of specific changes made to integrate BFCL into Nejumi Leaderboard.

### Major Updates

#### Inference System
- **Challenge**: Tool usage implementations differ by model, and BFCL required defining classes for each model. This creates overhead when adding new models.
- **Solutions**:
    - Created "OpenAIResponsesHandler-FC" as unified OpenAI model ID
    - Added OpenRouter support "OpenRouter-FC"
    - Added unified handlers for OSS models:
        - **UnifiedOSSFCHandler**: For vLLM tool call compatible models
        - **UnifiedOSSJsonSchemaHandler**: For tool call incompatible models using prompt engineering

#### STEP Limit Reduction
- Changed STEP limit from 20 to 10
- STEP = Number of attempts
- To accommodate models with small max tokens

### Other Minor Updates
- Added API model handlers for Gemini and Mistral
- Integrated BFCL evaluation into `scripts/run_eval.py`
- Changed from package download to relative imports
- Updated dependency management to uv-based system
- Enhanced WandB integration and reporting

---

## 5. End-to-End Evaluation Flow

```
┌──────────────────────────┐
│ Config Merge & Init      │
│  - Get default config   │
│  - Merge user config    │
│  - Handle testmode      │
└──────────┬───────────────┘
           │ Config Ready
           ▼
┌──────────────────────────┐
│ WandB Artifact Fetch     │
│  - Get datasets         │
│  - Download locally     │
└──────────┬───────────────┘
           │ BFCL_v3_*.json
           ▼
┌──────────────────────────┐
│ Inference (generation)   │
│  - Select unified handler│
│  - FC/JsonSchema support │
│  - Generate function calls│
└──────────┬───────────────┘
           │ Inference Results (JSONL)
           ▼
┌──────────────────────────┐
│ Evaluation (evaluation)  │
│  - AST analysis & exec check│
│  - Category-wise scoring │
│  - Calculate overall accuracy│
└──────────┬───────────────┘
           │ Score Files
           ▼
┌──────────────────────────┐
│ Result Processing & WandB│
│  - Leaderboard table    │
│  - Radar chart         │
│  - Detailed log table  │
└──────────────────────────┘
           ↓
      BFCL Overall Score
   (Overall Accuracy)
```

### Detailed Steps

1. **Config Merge & Initialization**
   - Get BFCL default settings with `get_default_config()`
   - Merge with user settings (`cfg.bfcl`) to generate final configuration
   - Limit to `samples_per_category=2` in testmode

2. **WandB Artifact Retrieval**
   - Download datasets from configured `artifacts_path`
   - Retrieve `BFCL_v3_*.json` files for each category

3. **Inference Execution (generation)**
   - Select appropriate unified handler based on model
     - **UnifiedOSSFCHandler**: tool call compatible models
     - **UnifiedOSSJsonSchemaHandler**: tool call incompatible models
   - Generate function calls for each test case

4. **Evaluation Execution (evaluation)**
   - **AST Analysis Phase**: Syntax analysis and accuracy checking
   - **Execution Check Phase**: Actual function execution and result comparison
   - **Parallel and Multi-turn Support**: Handle complex execution scenarios
   - Calculate category-wise accuracy (Non-Live, Live, Multi-Turn, etc.)

5. **Result Processing & WandB Recording**
   - Leaderboard table (Overall Acc, etc.)
   - Radar chart (category-wise accuracy)
   - Detailed log table (individual test case results)

### Output File Structure

- **Result Files**: `{result_dir}/{model_name}/BFCL_v3_{category}.json`
  - Inference results for each test case (JSONL format)
- **Score Files**: `{score_dir}/{model_name}/BFCL_v3_{category}_score.json`
  - Evaluation results and detailed error information

## 6. Main Configuration Items

### Specifying `bfcl_model_id` in Model Configuration

Each model's config file must include `bfcl_model_id`. Please select available model IDs from [SUPPORTED_MODELS.md](../scripts/evaluator/evaluate_utils/bfcl_pkg/SUPPORTED_MODELS.md).

#### Recommended Settings by Model Type

| Model Type | Recommended `bfcl_model_id` | Notes |
|-------------|---------------------------|-------|
| **OSS Models (FC-compatible)** | `unified-oss-fc` | Tokenizer natively supports tools arguments |
| **OSS Models (FC-incompatible)** | `unified-oss-jsonschema` | Prompt engineering + JSONSchema constraints |
| **OpenRouter** | `OpenRouter-FC` | Function calling via OpenRouter |
| **Provider APIs** | Provider-specific IDs | Dedicated handlers for Gemini, Mistral, Claude, OpenAI, etc. |

### Basic Configuration Example

```yaml
bfcl:
  test_category: "java javascript live_irrelevance live_multiple live_relevance live_simple multi_turn_base multi_turn_miss_func multi_turn_miss_param simple multiple irrelevance"
  temperature: 0.01
  num_threads: 2
  artifacts_path: 'llm-leaderboard/nejumi-leaderboard4/bfcl:production'
  generator_config:
    max_tokens: 8096
    temperature: 0.01
    top_p: 1.0
  handler_config:
    unified_oss_jsonschema:
      execution_result_include_call_str: true
      execution_result_include_call_id: false
      execution_result_join_parallel_calls: true
      execution_result_role: "tool"
```

---

## 7. FAQ

* **Why do I get JSON decode errors when output strings are cut off midway?**
    
    This error is mainly caused by **generation interruption due to token limits**.

    1. **max_tokens Setting Limitations**
       - BFCL has `generator.max_tokens` (default 128) configured
       - When input prompts are long, remaining token count becomes small

    2. **Dynamic Token Calculation**
       - Calculate input token count with `_estimate_leftover_tokens_count` function
       - Limited by `min(self.max_tokens, self.max_context_length - input_token_count - 2)`

    3. **LLM Generation Process**
       - When token limit is reached, LLM stops generation midway
       - Function call JSON ends in incomplete state

    ### Solutions
    - **Increase max_tokens**: Set sufficient token count for BFCL
    - **Optimize prompts**: Reduce input token count
    - **Improve error handling**: Enhance functionality to repair incomplete JSON