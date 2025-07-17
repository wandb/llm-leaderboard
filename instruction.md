

# 背景
- オリジナルのBFCLでは、vllm起動時にchat templateを利用せず、推論実行時にモデルごとのclassでtemplateの対応を行なっていた。Nejumi leaderboardでは、vllm起動時にchat templateを利用するので、モデルごとのclass内で手動で_format_promptで実装していたchat templateを削除し、llm-leaderboard/scripts/evaluator/evaluate_utils/bfcl_pkg/bfcl/model_handler/local_inference/base_oss_handler.py内のOSSHandler内でChat Completion形式に対応できるようにしたい。

- chat completionだからといって、chat completionのtool useはしなくて良い。chat completionを利用しつつ、function callingは従来通り自然言語（promptで行って）



例: For Llama 4 series, they use a different set of tokens than Llama 3
        if "Llama-4" in self.model_name:
            formatted_prompt = "<|begin_of_text|>"

            for message in messages:
                formatted_prompt += f"<|header_start|>{message['role']}<|header_end|>\n\n{message['content'].strip()}<|eot|>"

            formatted_prompt += f"<|header_start|>assistant<|header_end|>\n\n"
        # For Llama 3 series
        else:
            formatted_prompt = "<|begin_of_text|>"

            for message in messages:
                formatted_prompt += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content'].strip()}<|eot_id|>"

            formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"

        return formatted_prompt



# 補足1: llamaやdeepseekのかくモデルのクラスの中で必要となるメソッドは下記。すでにすべてのファイルで_format_promptはコメントアウトしているから安心して
- 不要になるメソッド
  - **`_format_prompt`**: Chat Completions APIが入力フォーマットを統一するため不要。チャットテンプレートの二重適用問題も解決される
- 依然として必要なメソッド
  - **`decode_ast`/`decode_execute`**: 出力パースは模型固有のため必要
  - **`_pre_query_processing_prompting`**: 前処理は模型固有のため必要。詳細は以下で解説します。

# 補足2: 一回失敗して、推論が開始されなくなった。慎重に考えてほしい。まず現状をgitで保存しながら慎重に進めて

# 補足3: 変更を行う際は、元のコードをコメントアウトして進め