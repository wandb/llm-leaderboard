import os
from typing import List
import concurrent.futures

import openai
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List
from tqdm.asyncio import tqdm

###########################################################
# Preprocess
###########################################################
def get_generated_result_wo_header_footer(gr_list: List[str], task: str) -> List[str]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    gr_wo_hf_list = []

    def process_item(gr):
        completion = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {
                    "role": "system",
                    "content": "あなたは指示に忠実に従うAIアシスタントです。ユーザーの指示に従って下さい。"
                },
                {
                    "role": "user",
                    "content": get_header_footer_remover_prompt(task, gr)
                }
            ]
        )
        return completion.choices[0].message.content

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        gr_wo_hf_list = list(tqdm(executor.map(process_item, gr_list), total=len(gr_list)))

    return gr_wo_hf_list


def _delete_blank(s: str) -> str:
    s = s.replace(" ", "")
    s = s.replace("　", "")
    s = s.replace("\n", "")
    s = s.replace("\t", "")
    return s


def _is_valid_format(s: str, s_wo_hf) -> bool:
    if s_wo_hf == "Invalid Error":
        return False

    s = _delete_blank(s)
    s_wo_hf = _delete_blank(s_wo_hf)
    if len(s_wo_hf) > 10:
        if (s[:10] == s_wo_hf[:10]) and (s[-10:] == s_wo_hf[-10:]):
            return True
        else:
            return False


def _is_valid_keyword(s: str, keyword: str) -> bool:
    if s == "Invalid Error":
        return False
    return keyword in s


def _is_valid_prohibited_word(s: str, keyword: str) -> bool:
    if s == "Invalid Error":
        return False
    return keyword not in s


def _is_valid_char_count(s: str, char_count: List[int]) -> bool:
    if s == "Invalid Error":
        return False
    return (char_count[0] <= len(s)) and (len(s) <= char_count[1])


def get_format_is_valid(row: pd.Series) -> bool:
    return _is_valid_format(row["format_result"], row["format_result_wo_hf"])

def get_keyword_is_valid(row: pd.Series) -> bool:
    return _is_valid_keyword(row["keyword_result_wo_hf"], row["keyword_answer"])


def get_prohibited_word_is_valid(row: pd.Series) -> bool:
    return _is_valid_prohibited_word(row["prohibited_word_result_wo_hf"], row["prohibited_word_answer"])

def get_char_count_is_valid(row: pd.Series) -> bool:
    return _is_valid_char_count(row["char_count_result_wo_hf"], row["char_count_answer"])


###########################################################
# Quality score check
###########################################################

def _delete_blank(s: str) -> str:
    s = s.replace(" ", "")
    s = s.replace("　", "")
    s = s.replace("\n", "")
    s = s.replace("\t", "")
    return s

def quality_process_results(df, task):
    result_dict = {"generated_text_id": df["generated_text_id"].tolist(), "prompt_id": df["prompt_id"].tolist()}
    ctg_cols = [col for col in df.columns if "wo_hf" in col]

    def process_row(base_text, generated_text, task):
        cr = check_generated_result(task, generated_text, base_text)
        if (cr == "適切") or (cr == "不適切"):
            label_map = {"適切": 1, "不適切": 0}
            return label_map[cr]
        else:
            cr = check_generated_result(task, generated_text, base_text, is_bool=True)
            if cr == "True" or cr == "False":
                label_map = {"True": 1, "False": 0}
                return label_map[cr]
            else:
                return cr

    for ctg_col in ctg_cols:
        ctg = ctg_col.replace("_result", "")
        base_texts = df["base_text"].tolist()
        generated_texts = df[ctg_col].tolist()

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            check_result_list = list(tqdm(executor.map(process_row, base_texts, generated_texts, [task]*len(base_texts)), total=len(base_texts)))

        result_dict[f"{ctg}"] = check_result_list

    return pd.DataFrame(result_dict)

def check_generated_result(task: str, generated_result: str, base_text: str = None, is_bool: bool = False):

    if is_bool:
        user_content = get_quality_check_prompt_bool(task, generated_result, base_text)
    else:
        user_content = get_quality_check_prompt(task, generated_result, base_text)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    completion = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=[
            {
                "role": "system",
                "content": "あなたは指示に忠実に従うAIアシスタントです。ユーザーの指示に従って下さい。"
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    )
    result = completion.choices[0].message.content
    
    return _delete_blank(result)

def check_label(cr):
    label_map = {"適切": True, "不適切": False, "True": True, "False": False}
    return label_map.get(cr, cr)

def calculate_true_proportions(df):
    return [df['format_wo_hf'].sum()/len(df),df['char_count_wo_hf'].sum()/len(df),df['keyword_wo_hf'].sum()/len(df),df['prohibited_word_wo_hf'].sum()/len(df)]

###########################################################
# Get scores
###########################################################

def transform_data(output_df, df, task):

    # Mapping data to the new structure
    categories = ['format', 'char_count', 'keyword', 'prohibited_word']
    outputs = ['format_result', 'char_count_result', 'keyword_result', 'prohibited_word_result']
    preprocessed_outputs = [
        'format_result_wo_hf', 'char_count_result_wo_hf', 'keyword_result_wo_hf', 'prohibited_word_result_wo_hf'
    ]
    ctgs = ['is_valid_format', 'is_valid_char_count', 'is_valid_keyword', 'is_valid_prohibited_word']
    #quals = ['format_wo_hf', 'char_count_wo_hf', 'keyword_wo_hf', 'prohibited_word_wo_hf']
    #reasons = ['format_wo_hf_reason', 'char_count_wo_hf_reason', 'keyword_wo_hf_reason', 'prohibited_word_wo_hf_reason']
    new_rows = []
    for _, row in df.iterrows():
        for i, cat in enumerate(categories):
            # Prepare data for the current category based on the current row
            row_data = {
                'question_id': row['prompt_id'],
                'task': task,
                'category': cat,
                'request': row[categories[i]],  # Assuming request is the same as category; adjust if different
                'base_text': row['base_text'],
                'prompt': "",
                'output': row[outputs[i]],
                'preprocessed_output': row[preprocessed_outputs[i]],
                'ctg': row[ctgs[i]]*1,
                #'qual': row[quals[i]],
                #'reason': row[reasons[i]]
            }
            # Append row data for the current category
            new_rows.append(row_data)
    # Create a DataFrame from the list of row data dictionaries
    new_rows_df = pd.DataFrame(new_rows)
    # Concatenate this new DataFrame with the existing output_df
    output_df = pd.concat([output_df, new_rows_df], ignore_index=True)

    return output_df


def get_quality_scores(file_path: str) -> Tuple[float, float, float, float]:
    output_list = list()
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "All-Acc" in line:
                output_list.append(float(line.split(": ")[1]))
        if len(output_list) != 4:
            raise RuntimeError("Invalid result file")
    return tuple(output_list)

###########################################################
# Prompt
###########################################################

def get_quality_check_prompt(task: str, generated_result: str, base_text: str = "") -> str:
    quality_check_prompt = ""
    if task == "summary":
        quality_check_prompt = f"""以下に要約した文章とその要約元の文章が提示されています。
要約した文章は要約元の文章を適切に要約できているかを判断してください。
適切に要約できている場合は「適切」、適切に要約できていない場合は「不適切」と回答してください。
ただし、要約元の文章から断定できない情報が要約した文章に含まれている場合も「不適切」と回答してください。
「適切」「不適切」のいずれかのみを出力し、説明文などは付与しないでください。
【要約元の文章】
{base_text}

【要約した文章】
{generated_result}
"""
    elif task == "ad_text":
        quality_check_prompt = f"""以下に、ランディングページの説明文とその説明文をもとに作成した1つの広告文のタイトルがあります。
説明文の内容に基づいているタイトルを作成できているかを判断してください。
適切に作成できている場合は「適切」、適切に作成できていない場合は「不適切」と回答してください。
ただし、説明文とタイトルが完全に一致している事例とタイトルとして長すぎる事例も「不適切」と回答してください。
「適切」「不適切」のいずれかのみを出力し、説明文などは付与しないでください。

【説明文】
{base_text}

【広告文のタイトル】
{generated_result}
"""
    elif task == "pros_and_cons":
        quality_check_prompt = f"""以下に提示している文章は、ある事象・事物についてのメリットとデメリットを生成AIに回答してもらった出力結果です。
出力結果が、メリット・デメリットの双方について言及できているか否かを回答してください。
言及できている場合は「適切」、言及できていない場合は「不適切」と回答してください。
「適切」「不適切」のいずれかのみを出力し、説明文などは付与しないでください。

【文章】
{generated_result}
"""
    return quality_check_prompt


def get_quality_check_prompt_bool(task: str, generated_result: str, base_text: str = "") -> str:
    if task == "summary":
        quality_check_prompt = f"""以下に要約した文章とその要約元の文章が提示されています。
要約した文章は要約元の文章を適切に要約できているかを判断してください。
適切に要約できている場合はTrue、適切に要約できていない場合はFalseと回答してください。
ただし、要約元の文章から断定できない情報が要約した文章に含まれている場合もFalseと回答してください。
必ずTrue, Falseのいずれかの単語を1つだけ出力してください。
【要約元の文章】
{base_text}

【要約した文章】
{generated_result}
"""
    elif task == "ad_text":
        quality_check_prompt = f"""以下に、ランディングページの説明文とその説明文をもとに作成した1つの広告文のタイトルがあります。
説明文の内容に基づいているタイトルを作成できているかを判断してください。
適切に作成できている場合はTrue、適切に作成できていない場合はFalseと回答してください。
ただし、説明文とタイトルが完全に一致している事例とタイトルとして長すぎる事例もFalseと回答してください。
必ずTrue, Falseのいずれかの単語を1つだけ出力してください。

【説明文】
{base_text}

【広告文のタイトル】
{generated_result}
"""
    elif task == "pros_and_cons":
        quality_check_prompt = f"""以下に提示している文章は、ある事象・事物についてのメリットとデメリットを生成AIに回答してもらった出力結果です。
出力結果が、メリット・デメリットの双方について言及できているか否かを回答してください。
言及できている場合はTrue、言及できていない場合はFalseと回答してください。
必ずTrue, Falseのいずれかの単語を1つだけ出力してください。

【文章】
{generated_result}
"""
    return quality_check_prompt



def get_header_footer_remover_prompt(task: str, generated_result: str) -> str:
    if task == "summary":
        remove_prompt = f"""以下に提示している文章は、ある文章を生成AIを用いて要約した出力結果です。
出力には「要約」あるいはそれに類する単語を含むような文として、「以下の文章を要約します。」「【要約】」などの冒頭の説明文や「以上が要約結果になります。」などの文末の説明文が入っていることがあります。また、英語でこれらの説明文が与えられることもあります。
提示した文章に上記で述べた説明文が含まれていない場合には提示した文章をそのまま出力し、上記で述べた説明文が含まれている場合は提示した文章から説明文を除去したものを抜き出してください。文章の中間部分を編集する必要は一切ありません。文が入っていることがあります。また、英語でこれらの説明文が与えられることもあります。
[文章]
{generated_result}
"""
    elif task == "ad_text":
        remove_prompt = f"""以下に提示している文章は、ある文章を元に作成した広告文のタイトルです。
出力には「広告文：」や「広告文を作成します」などの冒頭の接頭辞や説明文、「作成しました。」「このタイトルは、、」などの接尾辞やタイトルの後ろの説明文が含まれていることがあります。
提示した文章に上記で述べた説明文や接頭辞、接尾辞が含まれていない場合には、提示した文章をそのまま出力してください。「」や**などの記号で囲われている事例の場合、記号は全て残したまま出力してください。
上記で述べた説明文が含まれている場合は提示した文章から説明文や接頭辞、接尾辞を除去したものを抜き出してください。冒頭、末尾以外の中間部分を編集する必要は一切ありません。新しく文字を追加などをしないでください。

[文章]
{generated_result}
"""
    elif task == "pros_and_cons":
        remove_prompt = f"""以下に提示している文章は、ある事象・事物についてのメリットとデメリットを生成AIに回答してもらった出力結果です。
文章の冒頭や末尾に「そこで、メリットとデメリットをご紹介いたします。」「あなたのご質問にお答えいたします。」「以上が〇〇に関するメリット・デメリットです。」など内容と関係のない説明文が付与されている場合は、その説明文を除去して出力してください。ただし、文の一部は変更せずに、該当の文全体を除去してください。
上記のような説明文が付与されていない場合は、提示している文章をそのまま出力してください。

[文章]
{generated_result}
"""
    return remove_prompt