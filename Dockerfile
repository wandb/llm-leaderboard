FROM jupyter/minimal-notebook:lab-3.6.3
WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install --no-cache-dir -r requirements.txt

# ポートの公開
EXPOSE 8888

# SSHおよびJupyterLabの実行コマンド
CMD jupyter-lab --ip 0.0.0.0 --allow-root -b localhost
