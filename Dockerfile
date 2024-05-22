FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -y && apt-get install -y sudo python3 python3-pip git

# requirements.txtをコンテナにコピー
RUN pip3 install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
# pip install で必要なパッケージをインストール
RUN pip3 install -r /tmp/requirements.txt