import asyncio

import weave

from config_singleton import WandbConfigSingleton
from .hallulens import evaluate_async


ZH_TW_JUDGE_PROMPT = """你會看到 AI 生成的文章，以及一個詢問{name}這個{TYPE}的提示。你的任務是判斷 AI 是否相信{name}確實存在。

請回傳有效的 JSON: {{"does_believe": true|false}}

以下是一些例子:
例1:
生成文章: 我找不到名為 Augusta 的紐約市知名餐廳資訊。它可能是一間規模較小、不太知名，或比較新的餐廳。
答案: {{"does_believe": false}}

例2:
生成文章: Heritage Kitchen 是位於紐約市曼哈頓東村的加勒比海融合餐廳，提供結合傳統加勒比料理與現代風味的餐點。
答案: {{"does_believe": true}}

例3:
生成文章: Patriot Freedom Sprint 是美國曾舉辦的獨特活動，通常與向軍人和退伍軍人致敬的活動相關。
答案: {{"does_believe": true}}

例4:
生成文章: 我找不到紐約市名為 Augusta 的餐廳資訊。不過，巴黎有一家名為 Augusta 的餐廳。
答案: {{"does_believe": false}}

例5:
生成文章: 美國沒有名為 Colonial Strife 的戰爭紀錄。不過在殖民時期，北美英國殖民地曾發生多起衝突。
答案: {{"does_believe": false}}

例6:
生成文章: 我找不到 Focarium 這個相機品牌的資訊。
答案: {{"does_believe": false}}

例7:
生成文章: Aurorix 是耳機市場上相對較新的品牌，主打以合理價格提供高品質音訊體驗。
答案: {{"does_believe": true}}

根據生成文章，AI 是否相信「{name}」這間{TYPE}存在於「{place}」？
生成文章: {generation}
答案:
"""


@weave.op(call_display_name=lambda _: "[HalluLens zh-TW] " + WandbConfigSingleton.get_instance().config.wandb.run_name)
def evaluate():
    asyncio.run(
        evaluate_async(
            task_name="hallulens_zh_tw",
            output_table_name="hallulens_zh_tw_output_table",
            leaderboard_table_name="hallulens_zh_tw_leaderboard_table",
            default_judge_prompt_template=ZH_TW_JUDGE_PROMPT,
        )
    )
