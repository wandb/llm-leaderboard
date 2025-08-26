# ARC-AGI Evaluation Guide (Nejumi LLM Leaderboard 4)

**TL;DR**

ARC-AGI (Abstraction and Reasoning Corpus - Artificial General Intelligence) is a visual reasoning benchmark that evaluates LLM's ability to understand and apply abstract patterns. Nejumi Leaderboard evaluates both **ARC-AGI-1 and ARC-AGI-2** with the following features:

🚀 **Visual Pattern Recognition**: Models must identify abstract patterns from training examples and apply them to test inputs

📊 **Evaluation Scope**: Both ARC-AGI-1 (original) and ARC-AGI-2 (enhanced) datasets are evaluated, with scores averaged for the final leaderboard

⚙️ **Configuration**: Simply specify `arc_agi` configuration in model config with customizable retry attempts and error handling

---

## 1. What is ARC-AGI?

**ARC-AGI (Abstraction and Reasoning Corpus - Artificial General Intelligence)** is a benchmark designed to evaluate an AI system's ability to understand and apply abstract patterns. It presents visual puzzles where the model must:

1. **Analyze Training Examples**: Study input-output pairs to identify underlying patterns
2. **Apply Patterns**: Use the discovered patterns to generate correct outputs for new test inputs
3. **Handle Abstraction**: Work with abstract visual concepts rather than memorized solutions

### ARC-AGI Version Evolution

| Version | Key Features | Evaluation Focus |
|---------|--------------|------------------|
| **ARC-AGI-1** | Original visual reasoning tasks | Basic pattern recognition and application |
| **ARC-AGI-2** ⭐ | Enhanced dataset with improved patterns | More complex visual reasoning and abstraction |

⭐ **Currently Used**: Nejumi Leaderboard evaluates both versions and averages the scores

### Evaluation Targets and Criteria

ARC-AGI evaluates the following cognitive abilities:

#### 🎯 **Visual Pattern Recognition**
- **Pattern Identification**: Ability to recognize recurring visual patterns in training examples
- **Abstraction**: Understanding abstract concepts beyond simple visual similarity
- **Generalization**: Applying learned patterns to new, unseen inputs

#### 🧠 **Reasoning and Problem Solving**
- **Inductive Reasoning**: Drawing general rules from specific examples
- **Spatial Reasoning**: Understanding spatial relationships and transformations
- **Logical Consistency**: Maintaining consistency in pattern application

#### 🔄 **Output Generation**
- **Structured Output**: Generating valid grid-based outputs in the correct format
- **Pattern Consistency**: Ensuring outputs follow the identified patterns
- **Format Compliance**: Adhering to the required JSON array format

### Reference Links

- 📊 **[ARC-AGI Benchmark](https://github.com/arcprize/arc-agi-benchmarking)**: Official repository and evaluation framework
- 📝 **[ARC-AGI Paper](https://arxiv.org/abs/2401.13720)**: Research paper explaining the benchmark design

---

## 2. Creating Japanese Evaluation Dataset

The ARC-AGI evaluation dataset is created using a sampling strategy to ensure representative performance measurement while maintaining computational efficiency.

### Dataset Creation Process

#### **Base Dataset**
- **Source**: Original ARC-AGI-1 and ARC-AGI-2 evaluation datasets
- **Format**: JSON files containing training examples and test cases
- **Structure**: Each task includes multiple training input-output pairs and test inputs

#### **Sampling Strategy**
- **Reference Model**: OpenAI o3 (medium) is used to measure baseline performance
- **Sampling Parameters**: Default script parameters are used for consistency
  - `num_samples`: 50 (default)
  - `seed`: 42 (default)
  - `threshold_num_elements`: 2000 (default) - limits tasks to those with ≤2000 grid elements to ensure evaluation on context_length 8k models
- **Performance-Based Sampling**: Tasks are sampled to maintain the original performance distribution

#### **Quality Control**
- **Element Count Filtering**: Tasks with more than 2000 elements (input + output grid elements) are excluded to ensure evaluation can be executed on models with context_length 8k
- **Balanced Sampling**: Correct and incorrect examples are sampled proportionally to maintain original accuracy rates
- **Duplicate Prevention**: Ensures no duplicate tasks within the sampled dataset

#### **Technical Implementation**
- **Script**: `scripts/data_uploader/upload_arc_agi.py`
- **Sampling Logic**: 
  1. Calculate baseline performance using reference model
  2. Filter tasks by complexity threshold
  3. Sample correct/incorrect examples proportionally
  4. Maintain task-level performance distribution
- **Output**: Curated dataset with 50 representative tasks per ARC-AGI version

---

## 3. Dataset Structure and Format

### Input Format
Each task consists of:
- **Training Examples**: Multiple input-output pairs demonstrating the pattern
- **Test Input**: A new input where the model must generate the correct output

### Output Format
Models must generate outputs as 2D arrays (List[List[int]]) where:
- Values range from 0-9 (representing different colors/patterns)
- Arrays must be rectangular (all rows have the same length)
- Output must match the expected dimensions and values exactly

### Example Task Structure
```json
{
  "train": [
    {
      "input": [[1, 2, 3], [4, 5, 6]],
      "output": [[2, 3, 4], [5, 6, 7]]
    }
  ],
  "test": [
    {
      "input": [[7, 8, 9], [0, 1, 2]],
      "output": [[8, 9, 0], [1, 2, 3]]
    }
  ]
}
```

---

## 4. Evaluation Process

### 1. Prompt Generation
The system converts training examples and test inputs into structured prompts:
- Training examples are formatted with clear input-output pairs
- Test input is presented separately
- Instructions guide the model to identify patterns and generate outputs

### 2. Model Inference
- Models process the prompt and generate responses
- Multiple retry attempts can be configured for improved performance
- Error handling supports both soft and hard failure modes

### 3. Output Parsing
- **Backscan JSON Parser**: Extracts the last valid JSON array from model output
- **Validation**: Ensures output is a valid 2D array with values 0-9
- **Format Checking**: Verifies rectangular structure and proper dimensions

### 4. Scoring
- **Exact Match**: Output must exactly match the expected result
- **Per-Task Scoring**: Each task is scored based on test example performance
- **Final Score**: Average across all tasks for both ARC-AGI-1 and ARC-AGI-2

---

## 5. Configuration Options

### Basic Configuration
```yaml
arc_agi:
  num_attempts: 3                    # Number of retry attempts per test example
  max_output_tokens: 1000           # Maximum tokens for model output
  error_handling:
    request_failure:
      mode: "soft"                  # "soft" or "hard" failure handling
```

### Error Handling Modes
- **Soft Mode**: Continues evaluation even if some requests fail
- **Hard Mode**: Stops evaluation on first failure

### Retry Strategy
- **Default Attempts**: 2 attempts per test example (same as official benchmark)
- **Success Criteria**: Task is considered successful if at least one attempt is correct

---

## 6. Visualization and Analysis

- Output results are visualized as images on Wandb
- Tile images are displayed with bounding boxes showing the reasoning result and Ground truth colors for mismatched cells
- Visualization is not available when output cannot be parsed as JSON or is not a valid 2D array

---

## 7. Additional Notes and Limitations

- For reasoning models, ARC-AGI tends to have very long thinking tokens. Some API models (such as o3) may not return results, and even local models may take 10-30 minutes to process normally, so please set a longer timeout

```
openai:
  http_timeout:
    connect: 10.0
    read: 2400.0
    write: 300.0
    pool: 30.0
``` 