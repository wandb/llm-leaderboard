import datetime
import hashlib
import json
import numpy as np
import pandas as pd
from io import StringIO
from tqdm import tqdm
import wandb
from fastchat.llm_judge.common import load_questions
from fastchat.llm_judge.gen_model_answer import run_eval
from fastchat.llm_judge.gen_api_answer import get_api_answer
from fastchat.llm_judge.gen_judgment import *
from fastchat.llm_judge.common import (
        load_questions,
        load_model_answers,
        load_judge_prompts,
        check_data,
        play_a_match_pair,
        play_a_match_single,
        NEED_REF_CATS,
    )
from fastchat.utils import str_to_torch_dtype

def mtbench_evaluate(run_id, cfg, leaderboard_score):
    # create hash and append it to the model_id in order to avoid duplicated id
    mnaum_data = str(datetime.datetime.now())
    encoded_data = mnaum_data.encode()
    hash_object = hashlib.sha256(encoded_data)
    hashed_string = hash_object.hexdigest()
    if cfg.mtbench.model_id == None:
        cfg.mtbench.model_id = f'{cfg.metainfo.model_name.replace("/", "--")}_hash_{hashed_string}'

    # initialize wandb run
    run = wandb.init(entity=cfg.wandb.entity,
                     project=cfg.wandb.project,
                     id=run_id,
                     resume="allow")
    
    if cfg.mtbench.num_gpus_total // cfg.mtbench.num_gpus_per_model > 1:
        import ray
        ray.init()
    
    ## file path
    #question
    artifact_dir = run.use_artifact(cfg.mtbench.question_artifacts_path, type='dataset').download()
    question_file = artifact_dir+f"/question.jsonl"
    
    #create answerfile and answerdir
    answer_file = f"FastChat/fastchat/llm_judge/data/{cfg.mtbench.bench_name}/model_answer/{cfg.mtbench.model_id}.jsonl"
    answer_dir = f"FastChat/fastchat/llm_judge/data/{cfg.mtbench.bench_name}/model_answer"

    #refeerence answer
    ref_answer_dir = run.use_artifact(cfg.mtbench.referenceanswer_artifacts_path, type='dataset').download()

    # 1. generate model answers
    if cfg.metainfo.model_type == "openai":
        questions = load_questions(question_file, None, None)
        get_api_answer(
            question_file=question_file,
            answer_file=answer_file
        )
    else:
        run_eval(
            model_path=cfg.model.pretrained_model_name_or_path,
            model_id=cfg.mtbench.model_id,
            question_file=question_file,
            question_begin=cfg.mtbench.question_begin,
            question_end=cfg.mtbench.question_end,
            answer_file=answer_file,
            max_new_token=cfg.mtbench.max_new_token,
            num_choices=cfg.mtbench.num_choices,
            num_gpus_per_model=cfg.mtbench.num_gpus_per_model,
            num_gpus_total=cfg.mtbench.num_gpus_total,
            max_gpu_memory=cfg.mtbench.max_gpu_memory,
            dtype=str_to_torch_dtype(cfg.mtbench.dtype),
            #revision="main"
        )

    # 2. evaluate outputs
    ## Load questions
    questions = load_questions(question_file, None, None)

    ## Load answers
    model_answers = load_model_answers(answer_dir)
    model_answers = {cfg.mtbench.model_id: model_answers[cfg.mtbench.model_id]}
    ref_answers = load_model_answers(ref_answer_dir)

    ## Load judge
    artifact_dir = run.use_artifact(cfg.mtbench.judge_prompt_artifacts_path, type='dataset').download()
    judge_prompts = load_judge_prompts(artifact_dir + "/judge_prompts.jsonl")

    if cfg.mtbench.first_n:
        questions = questions[: cfg.mtbench.first_n]

    models = [cfg.mtbench.model_id] #get_model_list(answer_dir)
 
    if cfg.mtbench.mode == "single":
        judges = make_judge_single(cfg.mtbench.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = (
            f"FastChat/fastchat/llm_judge/data/{cfg.mtbench.bench_name}/model_judgment/{cfg.mtbench.judge_model}_single.jsonl"
        )
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(cfg.mtbench.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair
        output_file = (
            f"FastChat/fastchat/llm_judge/data/{cfg.mtbench.bench_name}/model_judgment/{cfg.mtbench.judge_model}_pair.jsonl"
        )
        if cfg.mtbench.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = cfg.mtbench.baseline_model

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_func(
        question_default, models, model_answers, judges["default"], baseline_model
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        multi_turn=True,
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )

    match_stat = {}
    match_stat["bench_name"] = cfg.mtbench.bench_name
    match_stat["mode"] = cfg.mtbench.mode
    match_stat["judge"] = cfg.mtbench.judge_model
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file

    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))
    #input("Press Enter to confirm...")

    # Play matches
    if cfg.mtbench.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file)
    else:

        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file)

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(cfg.mtbench.parallel) as executor:
            for match in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass

    # 3. consolidate results and log as wandb.Table
    # load questions
    df_question = pd.read_json(question_file, lines=True)

    # load answers
    df_answer = pd.read_json(f"FastChat/fastchat/llm_judge/data/{cfg.mtbench.bench_name}/model_answer/{cfg.mtbench.model_id}.jsonl", lines=True)
    df_answer = df_answer[df_answer.model_id == cfg.mtbench.model_id]
    df_answer = df_answer.sort_values(['question_id'])

    # load judge results
    df_judge = pd.read_json(output_file, lines=True)
    df_judge = df_judge[df_judge.model == cfg.mtbench.model_id]
    df_judge.model = df_judge.model.str.replace("--", "/")
    df_judge['hash'] = df_judge.model.apply(lambda x: x.split('_hash_')[-1])
    df_judge['model'] = df_judge.model.apply(lambda x: x.split('_hash_')[0])
    df_judge = df_judge.sort_values(['question_id', 'turn'])

    ## merge tables
    df_judge["question"] = np.nan
    df_judge.loc[df_judge.turn == 1, 'question'] = df_question.turns.apply(lambda x: x[0]).values
    df_judge.loc[df_judge.turn == 2, 'question'] = df_question.turns.apply(lambda x: x[1]).values

    df_judge['answer'] = np.nan
    df_judge.loc[df_judge.turn == 1, 'answer'] = df_answer.choices.apply(lambda x: x[0][ 'turns'][0]).values
    df_judge.loc[df_judge.turn == 2, 'answer'] = df_answer.choices.apply(lambda x: x[0][ 'turns'][1]).values
    df_judge = df_judge.merge(df_answer[['question_id', 'answer_id']], on='question_id', how='left')
    df_judge = df_judge.merge(df_question[['question_id', 'category']], on='question_id', how='left')

    ## clean dataframe up
    use_col = [
        'question_id', 'category', 'answer_id', 'model', 'question', 
        'answer', 'judge', 'user_prompt', 'judgment', 
        'score', 'turn', 'tstamp'
    ]
    df_judge = df_judge[use_col]

    table_log = wandb.Table(dataframe=df_judge)

    # table for radar chart
    df_summary = df_judge.groupby(['category'], as_index=False).score.mean()
    table_radar = wandb.Table(dataframe=df_summary)
    
    ## table for LB mtbench
    columns = ['model_name'] + df_summary.category.values.tolist()
    data = [[cfg.metainfo.model_name] + df_summary.score.values.tolist()]
    mtbench_df = pd.DataFrame(data, columns=columns)
    table_metric = wandb.Table(dataframe=mtbench_df)

    ## table for all
    jaster_df = pd.read_json(StringIO(leaderboard_score), orient='records', lines=True)
    mtbench_df = mtbench_df.drop(columns=['model_name'])
    combined_df = pd.concat([jaster_df,  mtbench_df], axis=1)
    table_all = wandb.Table(dataframe=combined_df)

    run.log({
        "mtbench_output_table":table_log,
        "mtbench_leaderboard_table":table_metric,
        "mtbench_radar_table":table_radar,
        "leaderboard_table":table_all
    })
    
    run.finish()
    return

