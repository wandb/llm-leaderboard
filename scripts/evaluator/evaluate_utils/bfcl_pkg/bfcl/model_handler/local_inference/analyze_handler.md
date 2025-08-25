# Find best handler
- これから利用するモデルのhandlerが`bfcl/model_handler/local_inference/unified_oss_handler.py`で十分かどうかを調査したい
- `bfcl/model_handler/local_inference/`の中のファイルも照らし合わせ、どんな実装にするべきか判断して。chat templateや、継続事前学習に利用したモデルやデータセットをもとに推察して
- `bfcl/model_handler/local_inference/unified_oss_handler.py`で対応が不十分であれば、`bfcl/model_handler/local_inference/`の中のファイルも照らし合わせ、どのhandlerを起点に作るべきかを教えて


- モデル名: tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5
- chat_tempalte: {{- bos_token }}{#- This block extracts the system message, so we can slot it into the right place. #}{%- if messages[0]['role'] == 'system' %}{%- set system_message = messages[0]['content']|trim %}{%- set messages = messages[1:] %}{%- else %}{%- set system_message = 'あなたは誠実で優秀な日本人のアシスタントです。'|trim %}{%- endif %}{#- System message #}{{- '<|start_header_id|>system<|end_header_id|>\n\n' }}{{- system_message }}{{- '<|eot_id|>' }}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}
- HuggingFaceモデルカードのURL: https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5
