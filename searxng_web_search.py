"""
title: SearXNG Search & Browserless Scraper (Ruri Embedding Ranked)
author: user
description: SearXNGで検索し、Browserlessで本文取得後、Recursive Character Splittingで分割、Ruri embeddingで質問との関連度が高いチャンクだけをLLMに渡すツール。日英対応。
version: 2.0.0
"""

import concurrent.futures
import math
import re
import sys
import time
import traceback
from datetime import datetime
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _log(msg: str):
    line = f"[SEARXNG_TOOL] {datetime.now().isoformat()} | {msg}"
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)


_SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".mp4", ".mp3", ".zip", ".rar", ".exe", ".dmg",
)

_MAIN_CONTENT_PATTERNS = [
    re.compile(r"<article[^>]*>(.*?)</article>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<main[^>]*>(.*?)</main>", re.DOTALL | re.IGNORECASE),
]

_STRIP_TAGS_PATTERN = re.compile(
    r"<(script|style|nav|footer|header|aside|form|iframe|noscript).*?>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _extract_main_text(html: str) -> str:
    """本文らしき領域を優先しつつ、HTMLからテキストを抽出する（切り詰めはしない。切り詰めはembeddingランキング後に行う）。"""
    main_html = html
    for pattern in _MAIN_CONTENT_PATTERNS:
        match = pattern.search(html)
        if match and len(match.group(1)) > 200:
            main_html = match.group(1)
            break

    cleaned = _STRIP_TAGS_PATTERN.sub("", main_html)
    text = _TAG_PATTERN.sub(" ", cleaned)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def _should_skip_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(_SKIP_EXTENSIONS)


def _recursive_split(text: str, chunk_size: int, overlap: int, separators: list[str]) -> list[str]:
    """
    LangChain不使用の自前実装。階層的セパレータ（段落→改行→日本語句読点→
    英語文末記号→空白→1文字）を順に試し、文脈境界を尊重しつつchunk_sizeに収める。
    2026年2月のVectaベンチマークでRecursive Character Splitting(512トークン)が
    全手法中1位（69%）だった結果を踏まえ、Semantic Chunkingより優先して採用する。
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # どのセパレータでも収まらない場合は文字数で強制分割
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest_separators = separators[0], separators[1:]
    parts = text.split(sep) if sep else list(text)

    chunks: list[str] = []
    current = ""

    for part in parts:
        piece = part + sep if sep else part
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                chunks.append(current)
            if len(piece) > chunk_size:
                # このセパレータでも大きすぎる場合はさらに細かいセパレータで再帰分割
                chunks.extend(_recursive_split(piece, chunk_size, overlap, rest_separators))
                current = ""
            else:
                current = piece

    if current.strip():
        chunks.append(current)

    # オーバーラップの付与（前チャンクの末尾をわずかに次チャンクへ含める）
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prefix = chunks[i - 1][-overlap:]
            overlapped.append(prefix + chunks[i])
        chunks = overlapped

    return [c for c in chunks if c.strip()]


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """日本語・英語どちらの文章にも対応した階層的セパレータで分割する。"""
    if not text.strip():
        return []

    separators = [
        "\n\n", "\n",
        "。", "！", "？",  # 日本語の文末記号
        ". ", "! ", "? ",  # 英語の文末記号
        "、", ", ",
        " ", "",
    ]
    return _recursive_split(text, chunk_size, overlap, separators)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class _PageCache:
    def __init__(self, ttl_seconds: int = 600):
        self._store: dict[str, tuple[float, str]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str):
        entry = self._store.get(key)
        if not entry:
            return None
        ts, value = entry
        if time.time() - ts > self._ttl:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: str):
        self._store[key] = (time.time(), value)


_page_cache = _PageCache()


class Tools:
    class Valves(BaseModel):
        SEARXNG_URL: str = Field(
            default="http://searxng:8080/search",
            description="SearXNGのベースURL",
        )
        BROWSERLESS_URL: str = Field(
            default="http://browserless:3000",
            description="Browserless(Chromium)のベースURL",
        )
        OLLAMA_URL: str = Field(
            default="http://host.docker.internal:11434",
            description="OllamaのベースURL（embedding計算に使用）",
        )
        EMBEDDING_MODEL: str = Field(
            default="kun432/cl-nagoya-ruri-base:latest",
            description="embeddingに使うOllamaモデル名（docker-compose.ymlのRAG_EMBEDDING_MODELと合わせる）",
        )
        RESULT_COUNT: int = Field(
            default=5,
            description="SearXNGから取得する検索結果の件数",
        )
        SCRAPE_TIMEOUT: int = Field(
            default=8,
            description="1ページあたりのレンダリングタイムアウト（秒）",
        )
        MAX_WORKERS: int = Field(
            default=6,
            description="並行取得の上限数",
        )
        CACHE_TTL_SECONDS: int = Field(
            default=600,
            description="同一URLの本文取得結果をキャッシュしておく秒数",
        )
        CHUNK_SIZE: int = Field(
            default=500,
            description="1チャンクあたりの目安文字数（ベンチマークで最高精度だったRecursive 512トークン相当）",
        )
        CHUNK_OVERLAP: int = Field(
            default=75,
            description="チャンク間のオーバーラップ文字数（目安15%程度、Chroma Researchの推奨レンジ）",
        )
        TOP_K_CHUNKS: int = Field(
            default=8,
            description="全ページのチャンクの中から、質問と関連度の高い上位いくつをLLMに渡すか",
        )
        EMBEDDING_TIMEOUT: int = Field(
            default=15,
            description="embedding計算1回あたりのタイムアウト秒数",
        )

    def __init__(self):
        _log("Tools が初期化されました")
        try:
            self.valves = self.Valves()
            _page_cache._ttl = self.valves.CACHE_TTL_SECONDS

            self.session = requests.Session()
            retry = Retry(
                total=1,
                backoff_factor=0.3,
                status_forcelist=[502, 503, 504],
                allowed_methods=["GET", "POST"],
            )
            adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retry)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

        except Exception:
            _log("初期化エラー:\n" + traceback.format_exc())
            raise

    # =========================================================
    # Embedding（Ruri）関連
    # =========================================================
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Ollama経由でRuri embeddingを計算する。1件失敗したら空ベクトルを返す。"""
        vectors = []
        for text in texts:
            try:
                res = self.session.post(
                    f"{self.valves.OLLAMA_URL}/api/embeddings",
                    json={"model": self.valves.EMBEDDING_MODEL, "prompt": text},
                    timeout=self.valves.EMBEDDING_TIMEOUT,
                )
                res.raise_for_status()
                vectors.append(res.json().get("embedding", []))
            except Exception as e:
                _log(f"[Embedding] 失敗 ({type(e).__name__}): {text[:30]!r}...")
                vectors.append([])
        return vectors

    def _rank_chunks_by_relevance(self, query: str, chunks_with_meta: list[dict]) -> list[dict]:
        """
        chunks_with_meta: [{"text": ..., "title": ..., "url": ...}, ...]
        質問のembeddingとの類似度でソートし、上位TOP_K_CHUNKSだけ返す。
        """
        if not chunks_with_meta:
            return []

        _log(f"[Embedding] クエリと{len(chunks_with_meta)}チャンクをembedding計算中")

        query_vec_list = self._embed([query])
        query_vec = query_vec_list[0] if query_vec_list else []

        if not query_vec:
            _log("[Embedding] クエリのembeddingに失敗。関連度ランキングをスキップし先頭から採用します")
            return chunks_with_meta[: self.valves.TOP_K_CHUNKS]

        chunk_texts = [c["text"] for c in chunks_with_meta]
        chunk_vecs = self._embed(chunk_texts)

        scored = []
        for meta, vec in zip(chunks_with_meta, chunk_vecs):
            score = _cosine_similarity(query_vec, vec) if vec else -1.0
            scored.append((score, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [meta for score, meta in scored[: self.valves.TOP_K_CHUNKS] if score > -1.0]

        _log(
            f"[Embedding] 上位{len(top)}チャンクを採用（スコア範囲: "
            f"{scored[0][0]:.3f} 〜 {scored[min(len(scored), self.valves.TOP_K_CHUNKS) - 1][0]:.3f}）"
            if top else "[Embedding] 有効なチャンクがありませんでした"
        )
        return top

    # =========================================================
    # 検索・本文取得
    # =========================================================
    def _search_searxng(self, query: str) -> list[dict]:
        _log(f"[Phase 1: Web検索] SearXNGにリクエスト送信: query='{query}'")
        params = {"q": query, "format": "json", "lang": "ja"}
        response = self.session.get(self.valves.SEARXNG_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        seen = set()
        results = []
        for r in data.get("results", []):
            u = r.get("url", "")
            if u and u not in seen:
                seen.add(u)
                results.append(r)
            if len(results) >= self.valves.RESULT_COUNT:
                break

        _log(f"[Phase 1: Web検索] {len(results)} 件のURLを取得しました（重複除去後）")
        return results

    def _read_page_raw(self, item: dict) -> dict:
        """本文を取得し、切り詰めずに返す（切り詰めはembeddingランキング後に行うため）。"""
        title = item.get("title", "(タイトルなし)")
        url = item.get("url", "")
        snippet = item.get("content", "(概要なし)")

        if not url or _should_skip_url(url):
            return {"title": title, "url": url, "text": snippet, "source": "snippet"}

        cached = _page_cache.get(url)
        if cached is not None:
            _log(f"[Phase 2: 本文読み取り] キャッシュヒット: {url}")
            return {"title": title, "url": url, "text": cached, "source": "cache"}

        payload = {
            "url": url,
            "gotoOptions": {
                "waitUntil": "domcontentloaded",
                "timeout": (self.valves.SCRAPE_TIMEOUT - 1) * 1000,
            },
        }

        try:
            _log(f"[Phase 2: 本文読み取り] ブラウザアクセス開始: {url}")
            res = self.session.post(
                f"{self.valves.BROWSERLESS_URL}/content",
                json=payload,
                timeout=self.valves.SCRAPE_TIMEOUT,
            )
            res.raise_for_status()
            text = _extract_main_text(res.text)
            _page_cache.set(url, text)
            _log(f"[Phase 2: 本文読み取り] 成功 ({len(text)}文字): {url}")
            return {"title": title, "url": url, "text": text, "source": "scraped"}

        except Exception as e:
            _log(f"[Phase 2: 本文読み取り] スキップ/エラー ({type(e).__name__}): {url}")
            return {"title": title, "url": url, "text": snippet, "source": "fallback_snippet"}

    def _fetch_all_pages_parallel(self, search_results: list[dict]) -> list[dict]:
        worker_count = min(len(search_results), self.valves.MAX_WORKERS) or 1
        _log(f"[Phase 2: 本文読み取り] {len(search_results)} 件を並行度{worker_count}で取得開始")

        pages = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(self._read_page_raw, item) for item in search_results]
            for future in concurrent.futures.as_completed(futures):
                pages.append(future.result())
        return pages

    def search_web(self, query: str) -> str:
        """
        Web検索を実行し、Browserlessで本文を取得したうえで、Ruri embeddingにより
        質問との関連度が高い部分だけを抽出して返します。
        最新情報、具体的な出来事、事実確認などが必要な際に使用してください。

        :param query: 検索キーワード
        :return: 関連度の高い抜粋を中心にまとめたテキスト
        """
        start = time.time()
        _log(f"=== search_web 開始: query='{query}' ===")
        try:
            search_results = self._search_searxng(query)
            if not search_results:
                return f"「{query}」に関する検索結果が見つかりませんでした。"

            pages = self._fetch_all_pages_parallel(search_results)

            # 全ページの本文をチャンク分割
            chunks_with_meta = []
            for page in pages:
                for chunk in _chunk_text(page["text"], self.valves.CHUNK_SIZE, self.valves.CHUNK_OVERLAP):
                    chunks_with_meta.append({"text": chunk, "title": page["title"], "url": page["url"]})

            _log(f"[Phase 3: Embeddingランキング] 総チャンク数: {len(chunks_with_meta)}")

            top_chunks = self._rank_chunks_by_relevance(query, chunks_with_meta)

            if not top_chunks:
                # embeddingが全滅した場合のフォールバック：各ページ先頭を少しずつ
                _log("[Phase 3] embeddingランキング結果が空のため、各ページ先頭にフォールバック")
                output_parts = [
                    f"■ {p['title']}\nURL: {p['url']}\n【内容】\n{p['text'][:800]}\n" for p in pages
                ]
            else:
                # URL単位でグルーピングして読みやすく整形
                grouped: dict[str, dict] = {}
                for c in top_chunks:
                    key = c["url"] or c["title"]
                    grouped.setdefault(key, {"title": c["title"], "url": c["url"], "texts": []})
                    grouped[key]["texts"].append(c["text"])

                output_parts = []
                for g in grouped.values():
                    merged_text = "\n…\n".join(g["texts"])
                    output_parts.append(f"■ {g['title']}\nURL: {g['url']}\n【関連抜粋】\n{merged_text}\n")

            output = f"「{query}」のWeb検索結果（関連度の高い部分を抜粋）:\n\n" + "\n".join(output_parts)
            elapsed = time.time() - start
            _log(f"=== search_web 正常終了（{elapsed:.1f}秒） ===")
            return output

        except Exception:
            tb = traceback.format_exc()
            _log("search_web 全体で例外発生:\n" + tb)
            return "検索処理中にエラーが発生しました。"
