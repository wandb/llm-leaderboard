
import numpy as np
import wandb
from wandb.integration.langchain import WandbTracer
import torch
from langchain.chains import SequentialChain
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from scipy.stats import pearsonr, spearmanr
import re
from fuzzywuzzy import fuzz
import unicodedata
from tqdm.notebook import tqdm


def eval_MARC_ja(dataset,llm_chain):
    y_trues = []
    y_preds = []
    overall_chain = SequentialChain(
        chains=[llm_chain], 
        input_variables = llm_chain.input_keys+['label'],
        output_variables = llm_chain.output_keys+["label"],
        verbose=False
    )
    for i in tqdm(range(len(dataset['validation']))):
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
        verbose=False
    )

    y_trues = []
    y_preds = []

    for i in tqdm(range(len(dataset['validation']))):
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
    jsts_peason=np.nan_to_num(jsts_peason, nan=0)
    jsts_spearman=np.nan_to_num(jsts_spearman, nan=0)
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
        verbose=False
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

def eval_JSQuAD(dataset,llm_chain):

    exact_match_scores = []
    f1_scores = []
    y_trues = []
    y_preds = []
    overall_chain = SequentialChain(
        chains=[llm_chain], 
        input_variables = llm_chain.input_keys+['label'],
        output_variables = llm_chain.output_keys+["label"],
        verbose=False
    )
    max_new_tokens = 25

    for i in tqdm(range(len(dataset['validation']))):
        if len(max(dataset['validation'][i]['answers']['text']))<=max_new_tokens*1.5:
            sentence1, sentence2 = dataset['validation'][i]['context'], dataset['validation'][i]['question']
            y_true = dataset['validation'][i]['answers']
            y_pred = overall_chain({'context':sentence1, 'question':sentence2, 'label':str(y_true['text'])},callbacks=[WandbTracer()])['output'].strip()

            y_trues.append(y_true)
            y_preds.append(y_pred)
            torch.cuda.empty_cache()
 

    y_trues=np.array(y_trues)
    y_preds=np.array(y_preds)

    vec_clean_normalize_string = np.vectorize(clean_normalize_string)
    y_preds = vec_clean_normalize_string(y_preds)

    for y_true, y_pred in zip(y_trues, y_preds):
        exact_match, max_f1_score = compute_scores(y_true, y_pred)
        exact_match_scores.append(exact_match)
        f1_scores.append(max_f1_score)

    exact_match_scores = np.array(exact_match_scores)
    f1_scores = np.array(f1_scores)

    JSQuAD_EM = np.mean(exact_match_scores)
    JSQuAD_F1 = np.mean(f1_scores)

    return JSQuAD_EM, JSQuAD_F1

def compute_scores(y_true, y_pred):
    # Exact Match Score
    exact_match = y_pred in y_true['text']

    # F1 Score
    f1_scores = [fuzz.token_sort_ratio(y_pred, true_text) for true_text in y_true['text']]
    max_f1_score = max(f1_scores) / 100.0  # Normalize to [0, 1] range

    return exact_match, max_f1_score

def clean_normalize_string(s):
    s = s.strip("「」『』。")
    return unicodedata.normalize('NFKC', s)

def eval_JCommonsenseQA(dataset,llm_chain):
    overall_chain = SequentialChain(
        chains=[llm_chain], 
        input_variables = llm_chain.input_keys+['label'],
        output_variables = llm_chain.output_keys+["label"],
        verbose=False
    )
    y_trues = []
    y_preds = []

    for i in tqdm(range(len(dataset['validation']))):
        data = dataset['validation'][i]
        question = data['question']
        choices = '0.%s,1.%s,2.%s,3.%s,4.%s'%(data['choice0'],data['choice1'],data['choice2'],data['choice3'],data['choice4'])
        y_true = data['label']
        y_pred = overall_chain({'question':question, 'choices':choices, 'label':y_true},callbacks=[WandbTracer()])['output'].strip()
        
        y_trues.append(y_true)
        y_preds.append(y_pred)
        torch.cuda.empty_cache()
        #time.sleep(1)
        #if i==100:# test with head20 for saving time. Please execute with all data for the formal evaluation.
        #    break
    y_trues=np.array(y_trues)
    y_preds=np.array(y_preds)
    vec_extract_first_number = np.vectorize(extract_first_number)
    y_preds = vec_extract_first_number(y_preds)
    y_preds = np.nan_to_num(y_preds, nan=0)
    JCommonsenseQA = accuracy_score(y_trues, y_preds)
    return JCommonsenseQA

def extract_first_number(s):
    match = re.match(r"(\d+)", s)
    return int(match.group(1)) if match else 0

def eval_JCoLA(dataset,llm_chain):
    overall_chain = SequentialChain(
        chains=[llm_chain], 
        input_variables = llm_chain.input_keys+['label'],
        output_variables = llm_chain.output_keys+["label"],
        verbose=False
    )
    y_trues = []
    y_preds = []
    
    for i in tqdm(range(len(dataset['validation']))):
        sentence = dataset['validation'][i]['sentence']
        y_true = dataset['validation'][i]['label']
        label = str(y_true)  + ["acceptable", "unacceptable"][y_true]
        y_pred = overall_chain({'sentence':sentence, 'label':label},callbacks=[WandbTracer()])['output']
        y_pred = y_pred.strip().lower()
        
        y_trues.append(y_true)
        y_preds.append(y_pred)
        torch.cuda.empty_cache()


    y_trues=np.array(y_trues)
    y_preds=np.array(y_preds)
    y_preds = np.nan_to_num(y_preds, nan=2)
    y_trues = [str(label) for label in y_trues]
    y_preds = [str(label) for label in y_preds]
    jcola_score = accuracy_score(y_trues, y_preds)
    jcola_balanced_score = balanced_accuracy_score(y_trues, y_preds)

    return jcola_score, jcola_balanced_score