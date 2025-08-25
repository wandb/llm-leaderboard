"""
モデルのchat_templateを取得するスクリプト
"""
import os
from argparse import ArgumentParser
from transformers import AutoTokenizer

def main(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    org, model_name = model_name_or_path.split('/')
    template_dir = os.path.join('chat_templates', org)
    os.makedirs(template_dir, exist_ok=True)
    template_path = os.path.join(template_dir, model_name + '.jinja')
    with open(template_path, 'w') as f:
        f.write(tokenizer.chat_template)
    print(template_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model_name_or_path', type=str)
    args = parser.parse_args()
    main(args.model_name_or_path)
