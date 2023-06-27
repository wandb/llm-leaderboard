from langchain import PromptTemplate

marc_ja_inst = """製品レビューをnegativeかpositiveのいずれかのセンチメントに分類してください。出力はnegativeかpositiveのいずれかのみで小文字化してください。それ以外には何も含めないことを厳守してください。
                [Examples]
                製品レビュー:以前職場の方にこれをみて少しでも元氣になってくださいと手渡して、早３年。返してくれと一度言ったが、結局返ってこなかった。６年前にも、職場の（といっても海外ですが）英語の先生に貸したら、これは素晴らしい！と言って、授業でも何度も生徒に見せたり、家でも見てたりしたそうで、結局帰国までに返してもらえなかった。。。この作品、結局３回購入してます。とほほでありつつ、誰かの心の支えになってくれればと願いつつ。エンディングの曲も好きです。あー、自分も突き進む人生を歩みたい。結婚もしたいが。。。
                センチメント:positive

                製品レビュー:全く３Ｄとはならなく、孫たちも途中で見たくなくなったくらいの不出来。他の人には購入しないようＰＲしたいくらい。なぜ、こんなＤＶＤを３Ｄとして販売しているのか理解できない。買って大失敗。
                センチメント:negative

                製品レビュー:私はカントリーが好きで当初CDを購入していいなと思ったのです、映画はそれなりのストーリー、まずまずです
                センチメント:positive

                製品レビュー:ジュリアロバーツを初めて見たのがこの作品だったということは良かったのか悪かったのかわからない。だってアメリカの有名女優という役だったから、そういう先入観で見た方がおもしろかったかも。でもイギリスのポートベローやロンドンの街並み、人々の暮らしがちりばめられていてとてもゆったりとした雰囲気。彼はちょっとダサく、でもやさしいフツーの人なのだ。（ハンサムと言うことをのぞけばね）二人や友人達の会話は要チェック。私なんぞ対訳のシナリオを買っちゃって勉強した。挿入歌がこれまたいいし。ぜひ見てね。
                センチメント:positive

                [Your Task]
                製品レビュー:{sentence}
                センチメント:"""

jsts_insct = """日本語の文ペアの意味がどのくらい近いかを判定し、類似度を0〜5までの間の値が付与してください。0に近いほど文ペアの意味が異なり、5に近いほど文ペアの意味が似ていることを表しています。整数値のみを返し、それ以外には何も含めないことを厳守してください。 
            [Example]
            文章1:川べりでサーフボードを持った人たちがいます。
            文章2:トイレの壁に黒いタオルがかけられています。
            相関係数:0.0

            文章1:二人の男性がジャンボジェット機を見ています。
            文章2:2人の男性が、白い飛行機を眺めています。
            相関係数:3.799999952316284

            文章1:男性が子供を抱き上げて立っています。
            文章2:坊主頭の男性が子供を抱いて立っています。
            相関係数:4.0

            文章1:野球の打者がバットを構えている横には捕手と審判が構えており、右端に次の打者が控えています。
            文章2:野球の試合で打者がバットを構えています。
            相関係数:2.200000047683716

            [Your Task]
            文章1:{sentence1}
            文章2:{sentence2}
            相関係数:"""
    
jnli_inst = """前提と仮説の関係をentailment、contradiction、neutralの中から回答してください。 それ以外には何も含めないことを厳守してください。
            制約：
            - 前提から仮説が、論理的知識や常識的知識を用いて導出可能である場合はentailmentと出力
            - 前提と仮説が両立しえない場合はcontradictionと出力
            - そのいずれでもない場合はneutralと出力

            [Example]
            前提:柵で囲まれたテニスコートでは、女子シングルスが行われています。
            仮説:柵で囲まれたテニスコートでは、女子ダブルスが行われています。
            関係:contradiction

            前提:二人の男性がジャンボジェット機を見ています。
            仮説:2人の男性が、白い飛行機を眺めています。
            関係:neutral

            前提:坊主頭の男性が子供を抱いて立っています。
            仮説:男性が子供を抱き上げて立っています。
            関係:entailment

            前提:手すりにクマのぬいぐるみが引っかかっている。
            仮説:柵の間にはクマのぬいぐるみがはさんでおいてあります。
            関係:neutral

            [Your Task]
            前提:{premise}
            仮説:{hypothesis}
            関係:"""
            
jsquad_inst = """質問に対する回答を文章から一言で抽出してください。回答は名詞で答えてください。 それ以外には何も含めないことを厳守してください。
            [Example]
            文章:聖武天皇 [SEP] 文武天皇の第一皇子として生まれたが、慶雲4年6月15日（707年7月18日）に7歳で父と死別、母・宮子も心的障害に陥ったため、その後は長らく会うことはなかった。物心がついて以後の天皇が病気の平癒した母との対面を果たしたのは齢37のときであった。このため、同年7月17日（707年8月18日）、父方の祖母・元明天皇（天智天皇皇女）が中継ぎの天皇として即位した。和銅7年6月25日（714年8月9日）には首皇子の元服が行われて同日正式に立太子されるも、病弱であったこと、皇親勢力と外戚である藤原氏との対立もあり、即位は先延ばしにされ、翌霊亀元年9月2日（715年10月3日）に伯母（文武天皇の姉）・元正天皇が「中継ぎの中継ぎ」として皇位を継ぐことになった。24歳のときに元正天皇より皇位を譲られて即位することになる。
            質問:文武天皇の第一皇子として生まれたのは？
            回答:聖武天皇

            文章:通称 [SEP] 人名としての通称は通り名、二つ名、異名、などと呼ばれる事もある。近世までは、本名（実名）は「」と呼ばれ、公言は避ける習慣があった。そのため、人を呼ぶ時は「仮名」「字」などの通称、官職名を用いるのが一般的だった。今日でも「総理」「大臣」「社長」「専務」などと呼びかけに使うのがこれにあたる。
            質問:人名としての通称は何と呼ばれているか
            回答:通り名、二つ名、異名

            文章:坂本龍一 [SEP] 2014年7月10日、所属事務所エイベックス・ミュージック・クリエイティヴから中咽頭癌であること、療養に専念するためにコンサート活動などを中止する旨が発表された。かつてはインタビューなどで度々自身の健康状態や体力に自信を表しており、コンサート等公演スケジュールを自身の健康に起因する理由でキャンセルしたことがなかった。
            質問:坂本龍一が療養に専念するためコンサート活動などを中止すると発表したのはいつか。
            回答:2014年7月10日

            文章:リリーフ [SEP] プレッシャーの比較的かからない状態で投げることができるので、若手投手のテストの場としたり、故障明けや登板間隔の開いた投手を調整目的で登板させることもある。敗戦処理であっても好投すれば次回から先発や接戦での中継ぎに起用されるようになる場合もあり、幸い打線の援護を受けてチームが逆転すれば勝利投手に輝くこともある。
            質問:打線の援護を受けてチームが逆転するとどんな投手になる？
            回答:勝利投手

            [Your Task]
            文章:{context}
            質問:{question}
            回答:"""
    
jcqa_inst = """質問に対する回答を文章から一言で抽出してください。回答は名詞で答えてください。 それ以外には何も含めないことを厳守してください。
            [Examples]
            質問:主に子ども向けのもので、イラストのついた物語が書かれているものはどれ？
            選択肢:0.世界,1.写真集,2.絵本,3.論文,4.図鑑
            回答:2

            質問:未成年者を監護・教育し，彼らを監督し，彼らの財産上の利益を守る法律上の義務をもつ人は？
            選択肢:0.浮浪者,1.保護者,2.お坊さん,3.宗教者,4.預言者
            回答:1

            質問:数字の１を表すときに使う体は？
            選択肢:0.胸,1.肉球,2.背中,3.人差し指,4.親指
            回答:3

            質問:火を起こすとあらわれるもくもくするものは？
            選択肢:0.歯の変色,1.ガス,2.中毒,3.爆発,4.煙
            回答:4

            [Your Task]
            質問:{question}
            選択肢:{choices}
            回答:"""





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

def others(instruction):
    return instruction


temp_dict = {'alpaca':alpaca, 'rinna':rinna, 'pythia':pythia, 'others': others}
prompt_dict = {}
instructions = [marc_ja_inst, jsts_insct, jnli_inst, jsquad_inst, jcqa_inst]
eval_dict = {}
for e, i in zip(['MARC-ja', 'JSTS', 'JNLI', 'JSQuAD', 'JCommonsenseQA'], instructions):
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
        
    return prompt_template

