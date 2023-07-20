import numpy as np
import torch
import wandb
import sentencepiece
from datasets import load_dataset
from wandb.integration.langchain import WandbTracer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, TrainerCallback, pipeline
from langchain import PromptTemplate, HuggingFaceHub, HuggingFacePipeline, LLMChain, OpenAI
from langchain.chains import SequentialChain
from huggingface_hub import HfApi, list_models
from huggingface_hub.inference_api import InferenceApi
from prompt_template import get_template
from utils import eval_MARC_ja, eval_JSTS, eval_JNLI, eval_JSQuAD, eval_JCommonsenseQA

"can be changed in W&B Launch"
config = dict(
    wandb_project="LLM_evaluation_Japan_public",
    wandb_entity="wandb",
    model_name="cyberagent/open-calm-small",
    prompt_type="other",
    )

if __name__ == "__main__":
    table_contents = []
    table_contents.append(config.model_name)
    eval_category = ['MARC-ja', 'JSTS', 'JNLI', 'JSQuAD', 'JCommonsenseQA']
    with wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=config, name=config.model_name,job_type="eval") as run:
        config = wandb.config

        if "rinna" in config.model_name:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name,use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            
        model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True)
        template_type = config.prompt_type

        #MRAC-ja --------------------------------------------------------
        dataset = load_dataset("shunk031/JGLUE", name=eval_category[0])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=5, device=0, torch_dtype=torch.float16,
            temperature=0.8, repetition_penalty=1.2,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[0], template_type), output_key="output")
        marc_ja_score = eval_MARC_ja(dataset,llm_chain)
        table_contents.append(marc_ja_score)
        #JSTS--------------------------------------------------------
        dataset = load_dataset("shunk031/JGLUE", name=eval_category[1])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=3, device=0, torch_dtype=torch.float16,
            temperature=0.8, repetition_penalty=1.2,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[1], template_type), output_key="output")
        jsts_peason, jsts_spearman= eval_JSTS(dataset,llm_chain)
        table_contents.append(jsts_peason)
        table_contents.append(jsts_spearman)
        #JNLI--------------------------------------------------------
        dataset = load_dataset("shunk031/JGLUE", name=eval_category[2])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer,
            max_new_tokens=3, device=0, torch_dtype=torch.float16,
            temperature=0.8, repetition_penalty=1.2,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[2], template_type), output_key="output")
        jnli_score = eval_JNLI(dataset,llm_chain)
        table_contents.append(jnli_score)

        #JSQuAD--------------------------------------------------------

        dataset = load_dataset("shunk031/JGLUE", name=eval_category[3])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, eos_token_id=0, pad_token_id=0,
            max_new_tokens=25, device=0, torch_dtype=torch.float16, top_p=1, top_k=0,
            temperature=0.1, repetition_penalty=1.1,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[3], template_type), output_key="output")
        JSQuAD_EM, JSQuAD_F1= eval_JSQuAD(dataset,llm_chain)
        
        table_contents.append(JSQuAD_EM)
        table_contents.append(JSQuAD_F1)
 
        #JCommonsenseQA--------------------------------------------------------
        dataset = load_dataset("shunk031/JGLUE", name=eval_category[4])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, eos_token_id=0, pad_token_id=0,
            max_new_tokens=5, device=0, torch_dtype=torch.float16, top_p=1, top_k=0,
            temperature=0.1, repetition_penalty=1.1,
            )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[4], template_type), output_key="output")

        JCommonsenseQA = eval_JCommonsenseQA(dataset,llm_chain)
        table_contents.append(JCommonsenseQA)

        #End--------------------------------------------------------
        table = wandb.Table(columns=['model_name ','MARC-ja', 'JSTS-pearson', 'JSTS-spearman', 'JNLI', 'JSQuAD-EM', 'JSQuAD-F1', 'JCommonsenseQA'] ,
                            data=[table_contents])
        table = wandb.Table(columns=['model_name ','MARC-ja', 'JSTS-pearson', 'JSTS-spearman', 'JNLI', 'JSQuAD-EM', 'JSQuAD-F1', 'JCommonsenseQA'] ,
                            data=table.data)
        run.log({'result_table':table}) 
        run.log_code()

