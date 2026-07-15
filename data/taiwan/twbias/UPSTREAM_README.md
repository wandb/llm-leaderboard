# TWBias: A Benchmark for Assessing Social Bias in Traditional Chinese Large Language Models through a Taiwan Cultural Lens
This respository contains the official code for the EMNLP 2024 paper, [TWBias: A Benchmark for Assessing Social Bias in Traditional Chinese Large Language Models through a Taiwan Cultural Lens](https://aclanthology.org/2024.findings-emnlp.507/).

![framework image](https://imgur.com/q2M5CmB.jpg)

## Requirements
- torch
- transformers
- scipy

To install the required packages, run the following commands:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121 # for CUDA 12.1
pip install numpy pandas scipy transformers
```

## Usage
To calculate the perplexities of the model and save the results, simply run the `scripts/run_gender.sh` and `scripts/run_ethnicity.sh`. This will save the results to `result/gender` and `result/ethnicity` respectively. 

For the ethnicity bias, it takes longer time to iterate through all the combinations, so it is recommended to run the parallel version: `scripts/run_ethnicity_parallel.sh`

After generating the result csv files, you can use the jupyter notebook `visualization/visualize_results.ipynb` to visualize the evaluation results. Remember to modify the `model_name` according to your needs.
