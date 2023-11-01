import os
import numpy as np
import torch
import wandb
import sentencepiece
from datasets import load_dataset, load_from_disk
#from wandb.integration.langchain import WandbTracer
from langchain.callbacks.tracers import WandbTracer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer, TrainerCallback, pipeline
from langchain import PromptTemplate, HuggingFaceHub, HuggingFacePipeline, LLMChain, OpenAI
from langchain.chains import SequentialChain
from huggingface_hub import HfApi, list_models
from huggingface_hub.inference_api import InferenceApi
from huggingface_hub import login
from prompt_template import get_template
from utils import eval_MARC_ja, eval_JSTS, eval_JNLI, eval_JSQuAD, eval_JCommonsenseQA, eval_JCoLA
from peft import PeftModel, PeftConfig


"can be changed in W&B Launch's setting"
config = dict(
    wandb_project="LLM_evaluation_Japan",
    wandb_entity="wandb",
    model_name="cyberagent/open-calm-small",
    prompt_type="other",
    use_artifact = True,
    use_peft=False
    )

login(os.environ['HUGGINGFACE_TOKEN'])

if __name__ == "__main__":
    eval_category = ['MARC-ja', 'JSTS', 'JNLI', 'JSQuAD', 'JCommonsenseQA','JCoLA']
    with wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config, job_type="eval") as run:
        config = wandb.config
        table_contents = []
        table_contents.append(config["model_name"])
        
        if "rinna" in config.model_name:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name,use_fast=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if "Llama-2" in config.model_name:
            temperature = 1e-9
        else:
            temperature = 0

        if config.use_peft:
            peft_config = PeftConfig.from_pretrained(config.model_name)
            model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path,  torch_dtype=torch.float16)
            model = PeftModel.from_pretrained(model, config.model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(config.model_name, trust_remote_code=True,torch_dtype=torch.float16)
        template_type = config.prompt_type

        #MRAC-ja --------------------------------------------------------
        if config.use_artifact:
            artifact = run.use_artifact('wandb/LLM_evaluation_Japan/JGLUE-MRAC-ja:v0', type='dataset')
            artifact_dir = artifact.download()
            dataset = load_from_disk(artifact_dir)
        else:
            dataset = load_dataset("shunk031/JGLUE", name=eval_category[0])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=5, device=0, torch_dtype=torch.float16, temperature=temperature, 
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[0], template_type), output_key="output")
        marc_ja_score, marc_ja_balanced_score = eval_MARC_ja(dataset,llm_chain)
        table_contents.append(marc_ja_score)
        table_contents.append(marc_ja_balanced_score)
        #JSTS--------------------------------------------------------
        if config.use_artifact:
            artifact = run.use_artifact('wandb/LLM_evaluation_Japan/JGLUE-JSTS:v0', type='dataset')
            artifact_dir = artifact.download()
            dataset = load_from_disk(artifact_dir)
        else:
            dataset = load_dataset("shunk031/JGLUE", name=eval_category[1])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=3, device=0, torch_dtype=torch.float16, temperature=temperature,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[1], template_type), output_key="output")
        jsts_peason, jsts_spearman= eval_JSTS(dataset,llm_chain)
        table_contents.append(jsts_peason)
        table_contents.append(jsts_spearman)
        #JNLI--------------------------------------------------------
        if config.use_artifact:
            artifact = run.use_artifact('wandb/LLM_evaluation_Japan/JGLUE-JNLI:v0', type='dataset')
            artifact_dir = artifact.download()
            dataset = load_from_disk(artifact_dir)
        else:
            dataset = load_dataset("shunk031/JGLUE", name=eval_category[2])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=3, device=0, torch_dtype=torch.float16, temperature=temperature,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[2], template_type), output_key="output")
        jnli_score, jnli_balanced_score = eval_JNLI(dataset,llm_chain)
        table_contents.append(jnli_score)
        table_contents.append(jnli_balanced_score)

        #JSQuAD--------------------------------------------------------
        if config.use_artifact:
            artifact = run.use_artifact('wandb/LLM_evaluation_Japan/JGLUE-JSQuAD:v0', type='dataset')
            artifact_dir = artifact.download()
            dataset = load_from_disk(artifact_dir)
        else:
            dataset = load_dataset("shunk031/JGLUE", name=eval_category[3])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=25, device=0, torch_dtype=torch.float16, temperature=temperature,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[3], template_type), output_key="output")
        JSQuAD_EM, JSQuAD_F1= eval_JSQuAD(dataset,llm_chain)
        
        table_contents.append(JSQuAD_EM)
        table_contents.append(JSQuAD_F1)
 
        #JCommonsenseQA--------------------------------------------------------
        if config.use_artifact:
            artifact = run.use_artifact('wandb/LLM_evaluation_Japan/JGLUE-JCommonsenseQA:v0', type='dataset')
            artifact_dir = artifact.download()
            dataset = load_from_disk(artifact_dir)
        else:
            dataset = load_dataset("shunk031/JGLUE", name=eval_category[4])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=5, device=0, torch_dtype=torch.float16, temperature=temperature,
            )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[4], template_type), output_key="output")

        JCommonsenseQA = eval_JCommonsenseQA(dataset,llm_chain)
        table_contents.append(JCommonsenseQA)

        #JCoLA--------------------------------------------------------
        if config.use_artifact:
            artifact = run.use_artifact('wandb/LLM_evaluation_Japan/JGLUE-JCoLA:v0', type='dataset')
            artifact_dir = artifact.download()
            dataset = load_from_disk(artifact_dir)
        else:
            dataset = load_dataset("shunk031/JGLUE", name=eval_category[5])
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=5, device=0, torch_dtype=torch.float16, temperature=temperature,
            )
        llm = HuggingFacePipeline(pipeline=pipe)
        llm_chain = LLMChain(llm=llm, prompt=get_template(eval_category[5], template_type), output_key="output")

        JCoLA_score, JCoLA_balanced_score = eval_JCoLA(dataset,llm_chain)
        table_contents.append(JCoLA_score)
        table_contents.append(JCoLA_balanced_score)

        #End--------------------------------------------------------
        table = wandb.Table(columns=['model_name ','MARC-ja','MARC-ja-balanced', 'JSTS-pearson', 'JSTS-spearman', 'JNLI', 'JNLI-balanced','JSQuAD-EM', 'JSQuAD-F1', 'JCommonsenseQA','JCoLA','JCoLA-balanced'] ,
                            data=[table_contents])
        table = wandb.Table(columns=['model_name ','MARC-ja','MARC-ja-balanced', 'JSTS-pearson', 'JSTS-spearman', 'JNLI', 'JNLI-balanced','JSQuAD-EM', 'JSQuAD-F1', 'JCommonsenseQA','JCoLA','JCoLA-balanced'] ,
                            data=table.data)
        run.log({'result_table_balanced':table}) 
        run.log_code()

