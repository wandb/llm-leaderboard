# TWBias Taiwan Bias Evaluation

TWBias evaluates social bias in Traditional Chinese with a Taiwan cultural lens.
The upstream benchmark is published with the EMNLP 2024 paper:
https://aclanthology.org/2024.findings-emnlp.507/

## Artifact

Materialize the upstream GitHub data locally:

```bash
python3 scripts/data_uploader/prepare_twbias.py --output-dir data/taiwan
```

The upstream repository did not include a LICENSE file when this integration was
implemented. For that reason, the prepared manifest records `license: unknown`,
and the evaluator refuses to run unless `twbias.allow_unknown_license=true` is
set explicitly for internal verification.

Do not publish `llm-leaderboard/tc-leaderboard/twbias:production` until
distribution rights are confirmed.

## Evaluation

The implemented evaluator follows the upstream perplexity-comparison setup:

- build target replacements from `target_gender.csv` and `target_ethnicity.csv`
- compute PPL for the original biased sentence
- compute PPL for target-replaced sentences
- mark a biased preference when the original biased sentence has lower PPL than
  the replacement set

The leaderboard metric is:

```text
unbiased_response_rate = 1 - bias_preference_rate
```

This is an HF/local-model evaluator. It requires token-level likelihood from
`transformers.AutoModelForCausalLM`; it is not equivalent to a generation-only
OpenAI-style API evaluation.

Minimal internal smoke settings:

```yaml
run:
  twbias: true
twbias:
  artifacts_path: 'llm-leaderboard/tc-leaderboard/twbias:internal'
  allow_unknown_license: true
  prompt_ids: ['0']
  max_samples_per_task: 2
```
