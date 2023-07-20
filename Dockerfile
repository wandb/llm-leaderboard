FROM rapidsai/rapidsai:cuda11.8-runtime-ubuntu22.04-py3.10
WORKDIR /app
COPY requirements.txt .
COPY japanese-task-evaluation.py ./

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt
RUN huggingface-cli login --token $HUGGINGFACE_TOKEN

CMD ["python", "japanese-task-evaluation.py"]

