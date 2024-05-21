FROM nvcr.io/nvidia/pytorch:24.04-py3

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && apt-get install -y sudo 

# 必要なパッケージをインストール
RUN pip install git+https://github.com/matsuolab/FastChat@main git+https://github.com/llm-jp/llm-jp-eval.git@wandb-nejumi2
RUN pip install google-generativeai langchain-community langchain-google-genai langchain-mistralai langchain-anthropic cohere sentencepiece
