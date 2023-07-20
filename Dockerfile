FROM rapidsai/rapidsai:cuda11.8-base-ubuntu22.04-py3.10
WORKDIR /app
COPY requirements.txt .
COPY src/japanese-task-evaluation.py ./

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "japanese-task-evaluation.py"]
