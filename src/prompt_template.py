from langchain import PromptTemplate

marc_ja_inst = '製品レビューをnegativeかpositiveのいずれかのセンチメントに分類してください。出力はnegativeかpositiveのいずれかのみで小文字化してください。それ以外には何も含めないことを厳守してください。\n製品レビュー:{sentence}'
jsts_insct = '日本語の文ペアの意味がどのくらい近いかを判定し、類似度を0〜5までの間の値が付与してください。0に近いほど文ペアの意味が異なり、5に近いほど文ペアの意味が似ていることを表しています。整数値のみを返し、それ以外には何も含めないことを厳守してください。 \n\nsentence1:{sentence1}\nsentence2:{sentence2}'
jnli_inst = "前提と仮説の関係をentailment、contradiction、neutralの中から回答してください。 それ以外には何も含めないことを厳守してください。\n\n制約：\n- 前提から仮説が、論理的知識や常識的知識を用いて導出可能である場合はentailmentと出力\n- 前提と仮説が両立しえない場合はcontradictionと出力\n- そのいずれでもない場合はneutralと出力 \n前提:{premise}\n仮説:{hypothesis}"
jsquad_inst = "質問に対する回答を文章から一言で抽出してください。回答は名詞で答えてください。 それ以外には何も含めないことを厳守してください。\n\n文章:{context}\n質問:{question}"
jcqa_inst = "質問と回答の選択肢を入力として受け取り、選択肢から回答を選択してください。なお、回答は選択肢の番号（例：0）でするものとします。 回答となる数値をint型で返し、他には何も含めないことを厳守してください。\n\n質問:{question}\n選択肢:{choices}"
jcola_inst = "与えられた日本語の文章の文法が正しければ1と答え、間違っている場合は0と答えてください。それ以外には何も含めないことを厳守してください。\n文章:{sentence}"

def alpaca(instruction):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction:
            {instruction}
            ### Response:
            """

def rinna(instruction):
    return f"ユーザー: {instruction}<NL>システム: ".replace("\n", "<NL>")

def pythia(instruction):
    return f"<|prompter|>{instruction}<|endoftext|><|assistant|> "

def llama2(instruction):
    system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{instruction} [/INST]"
    
def elyza(instruction):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。"
    prompt = "{b_inst} {system}{prompt}{e_inst} ".format(
        b_inst=B_INST,
        system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
        prompt=instruction,
        e_inst=E_INST,
    )
    return prompt


def other(instruction):
    return instruction


temp_dict = {'alpaca':alpaca, 'rinna':rinna, 'pythia':pythia,'llama2':llama2, 'elyza':elyza,'other': other}
prompt_dict = {}
instructions = [marc_ja_inst, jsts_insct, jnli_inst, jsquad_inst, jcqa_inst,jcola_inst]
eval_dict = {}
for e, i in zip(['MARC-ja', 'JSTS', 'JNLI', 'JSQuAD', 'JCommonsenseQA','JCoLA'], instructions):
    eval_dict[e] = i

def get_template(eval_category, template_type):
    inst_sentence = eval_dict[eval_category]
    inst_template = temp_dict[template_type]
    
    if eval_category=='MARC-ja':
        prompt_template = PromptTemplate(
            input_variables=["sentence"],
            template=inst_template(inst_sentence)
        )
        
    if eval_category=='JSTS':
        prompt_template = PromptTemplate(
            input_variables=["sentence1", "sentence2"],
            template=inst_template(inst_sentence)
        )
        
    if eval_category=='JNLI':
        prompt_template = PromptTemplate(
            input_variables=["premise", "hypothesis"],
            template=inst_template(inst_sentence)
        )
        
    if eval_category=='JSQuAD':
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=inst_template(inst_sentence)
        )
        
    if eval_category=='JCommonsenseQA':
        prompt_template = PromptTemplate(
            input_variables=["question", "choices"],
            template=inst_template(inst_sentence)
        )
    if eval_category=='JCoLA':
        prompt_template = PromptTemplate(
            input_variables=["sentence"],
            template=inst_template(inst_sentence)
        )
    return prompt_template

