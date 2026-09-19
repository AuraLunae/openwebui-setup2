"""
Ruri v3 (cl-nagoya/ruri-v3-310m) を提供する小さな埋め込みAPIサーバー。

Open WebUI標準のRAG機能はHugging Faceから直接ロードできるが、
カスタムツール（searxng_tool.py）からはOllama経由でしか埋め込みを
呼べないため、それを補うために独立したHTTPサーバーとして用意する。

Ruri v3は「1+3 prefixスキーム」を採用しており、用途に応じて
テキストの先頭に付けるプレフィックスが異なる:
  - クエリ側: "検索クエリ: "
  - 文書側:   "検索文書: "
  - 意味比較のみ: プレフィックスなし
"""

import os
from typing import List, Literal

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "cl-nagoya/ruri-v3-310m")

app = FastAPI(title="Ruri v3 Embedding Server")

print(f"[ruri-embedder] モデルをロード中: {MODEL_NAME}", flush=True)
model = SentenceTransformer(MODEL_NAME)
print("[ruri-embedder] モデルのロード完了", flush=True)

_PREFIXES = {
    "query": "検索クエリ: ",
    "document": "検索文書: ",
    "none": "",
}


class EmbedRequest(BaseModel):
    texts: List[str]
    type: Literal["query", "document", "none"] = "none"


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    prefix = _PREFIXES.get(req.type, "")
    prefixed_texts = [prefix + t for t in req.texts]
    vectors = model.encode(prefixed_texts, normalize_embeddings=True)
    return EmbedResponse(embeddings=vectors.tolist(), model=MODEL_NAME)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}