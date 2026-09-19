# Ruri v3 embeddingサーバー用イメージ
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU版PyTorchを明示的にインストール（GPU不要・イメージサイズも抑える）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV EMBEDDING_MODEL_NAME=cl-nagoya/ruri-v3-310m

# ビルド時点でモデルをダウンロードしてイメージに焼き込んでおく。
# これにより、初回コンテナ起動時の待ち時間がなくなる（ビルド自体には時間がかかる）。
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL_NAME}')"

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]