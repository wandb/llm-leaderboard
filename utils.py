
import numpy as np
import wandb
from wandb.integration.langchain import WandbTracer
import torch
from langchain.chains import SequentialChain
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr, spearmanr
import re

def eval_MARC_ja(dataset,llm_chain):
    y_trues = []
    y_preds = []
    overall_chain = SequentialChain(
                        chains=[llm_chain], 
                        input_variables = llm_chain.input_keys+['label'],
                        output_variables = llm_chain.output_keys+["label"],
                        verbose=True
                    )

    for i in range(len(dataset['validation'])):
        sentence = dataset['validation'][i]['sentence']
        y_true = dataset['validation'][i]['label']
        label = str(y_true)  + [" (positive/ポジティブ)", " (negative/ネガティブ)"][y_true]
        y_pred = overall_chain({'sentence':sentence, 'label':label},callbacks=[WandbTracer()])['output']
        y_pred = y_pred.strip().lower()
        
        y_trues.append(y_true)
        y_preds.append(y_pred)
        torch.cuda.empty_cache()


    y_trues=np.array(y_trues)
    y_preds=np.array(y_preds)

    # Replace 'ネガティブ' with 'negative'
    y_preds[y_preds == 'ネガティブ'] = 'negative'
    # Replace 'ポジティブ' with 'positive'
    y_preds[y_preds == 'ポジティブ'] = 'positive'
    # simple post processing
    y_preds = np.where(np.char.find(y_preds, 'negative') >= 0, 1, 0) 
    
    marc_ja_score = accuracy_score(y_trues, y_preds)
    return marc_ja_score


def eval_JSTS(dataset,llm_chain):
    overall_chain = SequentialChain(
                        chains=[llm_chain], 
                        input_variables = llm_chain.input_keys+['label'],
                        output_variables = llm_chain.output_keys+["label"],
                        verbose=True
                    )

    y_trues = []
    y_preds = []

    for i in range(len(dataset['validation'])):
        sentence1, sentence2 = dataset['validation'][i]['sentence1'], dataset['validation'][i]['sentence2']
        y_true = dataset['validation'][i]['label']
        y_pred = overall_chain({'sentence1':sentence1, 'sentence2':sentence2, 'label':y_true},callbacks=[WandbTracer()])['output'].strip()
        
        y_trues.append(y_true)
        y_preds.append(y_pred)
        torch.cuda.empty_cache()
    
    y_preds = [parse_float(pred) for pred in y_preds]
    y_preds = np.nan_to_num(y_preds, nan=2)
    jsts_peason = pearsonr(y_trues, np.array(y_preds).astype(float).clip(0,5))[0]
    jsts_spearman = spearmanr(y_trues, np.array(y_preds).astype(float).clip(0,5))[0]
    return jsts_peason, jsts_spearman

def parse_float(input_str):
    cleaned_str = re.sub(r'[^0-9.]', '', input_str)
    try:
        return float(cleaned_str)
    except ValueError:
        return 2.0



def eval_JNLI(dataset,llm_chain):
    overall_chain = SequentialChain(
        chains=[llm_chain], 
        input_variables = llm_chain.input_keys+['label'],
        output_variables = llm_chain.output_keys+["label"],
        verbose=True
    )
    y_trues = []
    y_preds = []

    for i in tqdm(range(len(dataset['validation']))):
        sentence1, sentence2 = dataset['validation'][i]['sentence1'], dataset['validation'][i]['sentence2']
        y_true = dataset['validation'][i]['label']
        label = str(y_true) + ' (%s)'%['entailment', 'contradiction', 'neutral'][y_true]
        y_pred = overall_chain({'premise':sentence1, 'hypothesis':sentence2, 'label':label},callbacks=[WandbTracer()])['output'].strip().lower()
        
        y_trues.append(y_true)
        y_preds.append(y_pred)
        torch.cuda.empty_cache()

    y_trues=np.array(y_trues)
    y_preds=np.array(y_preds)
    conditions = [y_preds == 'entailment', y_preds == 'contradiction', y_preds == 'neutral']
    choices = [0, 1, 2]
    y_preds = np.select(conditions, choices, default=0)
    y_preds = np.nan_to_num(y_preds, nan=0)
    jnli_score = accuracy_score(y_trues, y_preds)

    return jnli_score

