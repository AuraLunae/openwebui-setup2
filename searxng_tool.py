"""
title: SearXNG Search & Crawl4AI Scraper
author: user
description: SearXNGでキーワード検索を行い、上位ページの本文を取得・抽出して返すツール。
version: 4.5.0
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
    line = f"[SEARXNG_SEARCH] {datetime.now().isoformat()} | {msg}"
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)


_SKIP_EXTENSIONS = (
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".mp4",
    ".mp3",
    ".zip",
    ".rar",
    ".exe",
    ".dmg",
)


def _should_skip_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(_SKIP_EXTENSIONS)


def _strip_links(markdown: str) -> str:
    if not markdown:
        return markdown
    text = re.sub(r"[ \t]+", " ", markdown)
    text = re.sub(r"\n[ \t]+\n", "\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


_NUMBERED_LINE_PATTERN = re.compile(
    r"^\s*(?:\d+[.\)、]|手順\s*\d+[:：]|Step\s*\d+[:：])\s*.+$",
    re.MULTILINE,
)


def _find_procedure_blocks(text: str) -> list[tuple[int, int]]:
    lines = text.split("\n")
    line_starts = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line) + 1

    blocks = []
    i = 0
    while i < len(lines):
        if _NUMBERED_LINE_PATTERN.match(lines[i]):
            j = i
            while j < len(lines) and (
                _NUMBERED_LINE_PATTERN.match(lines[j]) or lines[j].strip() == ""
            ):
                j += 1
            while j > i and lines[j - 1].strip() == "":
                j -= 1
            if j - i >= 2:
                start = line_starts[i]
                end = line_starts[j - 1] + len(lines[j - 1])
                blocks.append((start, end))
            i = j
        else:
            i += 1
    return blocks


def _recursive_split(
    text: str, chunk_size: int, overlap: int, separators: list[str]
) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
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
                chunks.extend(
                    _recursive_split(piece, chunk_size, overlap, rest_separators)
                )
                current = ""
            else:
                current = piece

    if current.strip():
        chunks.append(current)

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prefix = chunks[i - 1][-overlap:]
            overlapped.append(prefix + chunks[i])
        chunks = overlapped

    return [c for c in chunks if c.strip()]


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text.strip():
        return []
    separators = [
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        ". ",
        "! ",
        "? ",
        "、",
        ", ",
        " ",
        "",
    ]
    return _recursive_split(text, chunk_size, overlap, separators)


def _chunk_text_safe(
    text: str, chunk_size: int, overlap: int, max_procedure_chars: int
) -> list[str]:
    if not text.strip():
        return []

    blocks = _find_procedure_blocks(text)
    if not blocks:
        return _chunk_text(text, chunk_size, overlap)

    chunks: list[str] = []
    cursor = 0
    for start, end in blocks:
        before = text[cursor:start]
        if before.strip():
            chunks.extend(_chunk_text(before, chunk_size, overlap))

        procedure_text = text[start:end]
        chunks.append(procedure_text)
        cursor = end

    remainder = text[cursor:]
    if remainder.strip():
        chunks.extend(_chunk_text(remainder, chunk_size, overlap))

    return [c for c in chunks if c.strip()]


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
        CRAWL4AI_URL: str = Field(
            default="http://crawl4ai:11235",
            description="Crawl4AIのベースURL",
        )
        CRAWL4AI_API_TOKEN: str = Field(
            default="your_secret_crawl4ai_token",
            description="CRAWL4AI_API_TOKEN",
        )
        EMBEDDING_SERVER_URL: str = Field(
            default="http://ruri-embedder:8001",
            description="Ruri v3 embeddingサーバーのベースURL",
        )
        SEARCH_RESULT_COUNT: int = Field(
            default=8,
            description="SearXNGから取得する検索結果件数",
        )
        PAGES_TO_FETCH: int = Field(
            default=4,
            description="本文取得する上位ページ数",
        )
        MAX_CHARS_PER_PAGE: int = Field(
            default=20000,
            description="1ページあたりの文字数上限",
        )
        CHUNK_SIZE: int = Field(default=500, description="1チャンクの目安文字数")
        CHUNK_OVERLAP: int = Field(default=75, description="オーバーラップ文字数")
        MAX_PROCEDURE_CHARS: int = Field(
            default=3000, description="手順ブロックの文字数上限目安"
        )
        TOP_K_CHUNKS: int = Field(
            default=8, description="上位いくつかチャンクをLLMに渡すか"
        )
        USE_EMBEDDING_RANKING: bool = Field(
            default=False,
            description="TrueならEmbeddingを使用（低速・高精度）、Falseなら全文モード（高速）",
        )
        MAX_WORKERS: int = Field(default=6, description="並行取得上限")
        CACHE_TTL_SECONDS: int = Field(
            default=600, description="キャッシュ保持時間（秒）"
        )
        SCRAPE_TIMEOUT: int = Field(
            default=15, description="1ページの取得タイムアウト（秒）"
        )
        FETCH_TIME_BUDGET_SECONDS: float = Field(
            default=8.0, description="全ページ取得の全体時間予算（秒）"
        )
        MIN_PAGES_REQUIRED: int = Field(
            default=1, description="最低保証取得ページ数"
        )
        EMBEDDING_TIMEOUT: int = Field(
            default=15, description="embeddingのタイムアウト（秒）"
        )

    def __init__(self):
        self.valves = self.Valves()
        _page_cache._ttl = self.valves.CACHE_TTL_SECONDS

        self.session = requests.Session()
        retry = Retry(
            total=1,
            backoff_factor=0.3,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(
            pool_connections=10, pool_maxsize=10, max_retries=retry
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _embed(self, texts: list[str], kind: str = "none") -> list[list[float]]:
        if not texts:
            return []
        try:
            res = self.session.post(
                f"{self.valves.EMBEDDING_SERVER_URL}/embed",
                json={"texts": texts, "type": kind},
                timeout=self.valves.EMBEDDING_TIMEOUT,
            )
            res.raise_for_status()
            data = res.json()
            return data.get("embeddings", [])
        except Exception:
            return [[] for _ in texts]

    def _select_relevant_pages(
        self, query: str, search_results: list[dict]
    ) -> list[dict]:
        if not search_results:
            return []
        snippets = [r.get("content", "") or r.get("title", "") for r in search_results]
        query_vecs = self._embed([query], kind="query")
        snippet_vecs = self._embed(snippets, kind="document")

        if not query_vecs or not query_vecs[0]:
            return search_results[: self.valves.PAGES_TO_FETCH]

        query_vec = query_vecs[0]
        scored = [
            (_cosine_similarity(query_vec, vec), item)
            for item, vec in zip(search_results, snippet_vecs)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored[: self.valves.PAGES_TO_FETCH]]

    def _rank_chunks_by_relevance(
        self, query: str, chunks_with_meta: list[dict]
    ) -> list[dict]:
        if not chunks_with_meta:
            return []
        query_vecs = self._embed([query], kind="query")
        if not query_vecs or not query_vecs[0]:
            return chunks_with_meta[: self.valves.TOP_K_CHUNKS]

        query_vec = query_vecs[0]
        chunk_texts = [c["text"] for c in chunks_with_meta]
        chunk_vecs = self._embed(chunk_texts, kind="document")

        scored = [
            (_cosine_similarity(query_vec, vec), meta)
            for meta, vec in zip(chunks_with_meta, chunk_vecs)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            meta for score, meta in scored[: self.valves.TOP_K_CHUNKS] if score > -1.0
        ]

    def _search_searxng_once(self, query: str, attempt: int) -> list[dict]:
        try:
            params = {"q": query, "format": "json", "lang": "ja"}
            response = self.session.get(
                self.valves.SEARXNG_URL, params=params, timeout=10
            )
            response.raise_for_status()
            return response.json().get("results", [])
        except Exception:
            return []

    def _search_searxng(self, query: str) -> list[dict]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(self._search_searxng_once, query, i + 1)
                for i in range(2)
            ]
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())

        seen = set()
        results = []
        for r in all_results:
            u = r.get("url", "")
            if u and u not in seen:
                seen.add(u)
                results.append(r)
            if len(results) >= self.valves.SEARCH_RESULT_COUNT:
                break
        return results

    def _fetch_page_via_crawl4ai(self, item: dict) -> dict:
        title = item.get("title") or item.get("url") or "(タイトルなし)"
        url = item.get("url", "")
        snippet = item.get("content", "(概要なし)")

        if not url or _should_skip_url(url):
            return {"title": title, "url": url, "text": snippet}

        cached = _page_cache.get(url)
        if cached is not None:
            return {"title": title, "url": url, "text": cached}

        headers = {}
        if self.valves.CRAWL4AI_API_TOKEN:
            headers["Authorization"] = f"Bearer {self.valves.CRAWL4AI_API_TOKEN}"

        try:
            res = self.session.post(
                f"{self.valves.CRAWL4AI_URL}/crawl",
                json={"urls": [url]},
                headers=headers,
                timeout=self.valves.SCRAPE_TIMEOUT,
            )
            res.raise_for_status()
            data = res.json()
            results = data.get("results", [data])
            page_result = results[0] if results else {}

            meta_title = (
                page_result.get("metadata", {}).get("title")
                or page_result.get("title")
            )
            if meta_title and title in (url, "(タイトルなし)"):
                title = meta_title

            markdown = page_result.get("markdown", "")
            if isinstance(markdown, dict):
                markdown = markdown.get("fit_markdown") or markdown.get(
                    "raw_markdown", ""
                )

            if not markdown:
                return {"title": title, "url": url, "text": snippet}

            markdown = _strip_links(markdown)
            if len(markdown) > self.valves.MAX_CHARS_PER_PAGE:
                markdown = (
                    markdown[: self.valves.MAX_CHARS_PER_PAGE] + "\n...(以下省略)"
                )

            _page_cache.set(url, markdown)
            return {"title": title, "url": url, "text": markdown}
        except Exception:
            return {"title": title, "url": url, "text": snippet}

    def _fetch_pages_parallel(self, items: list[dict]) -> list[dict]:
        worker_count = min(len(items), self.valves.MAX_WORKERS) or 1
        pages: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:
            future_to_item = {
                executor.submit(self._fetch_page_via_crawl4ai, item): item
                for item in items
            }
            done, not_done = concurrent.futures.wait(
                future_to_item.keys(),
                timeout=self.valves.FETCH_TIME_BUDGET_SECONDS,
                return_when=concurrent.futures.ALL_COMPLETED,
            )
            for future in done:
                pages.append(future.result())

            if len(pages) < self.valves.MIN_PAGES_REQUIRED and not_done:
                extra_done, not_done = concurrent.futures.wait(
                    not_done, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in extra_done:
                    pages.append(future.result())

            if not_done:
                for future in not_done:
                    item = future_to_item[future]
                    future.cancel()
                    pages.append(
                        {
                            "title": item.get("title", "(タイトルなし)"),
                            "url": item.get("url", ""),
                            "text": item.get("content", "(概要なし)"),
                        }
                    )
        return pages

    def search_web(self, query: str) -> str:
        """
        指定したキーワードでWeb検索を実行し、最新情報や具体的な情報を収集します。

        :param query: 検索キーワード
        :return: 検索結果のテキスト
        """
        _log(f"=== search_web 開始: query='{query}' ===")
        try:
            search_results = self._search_searxng(query)
            if not search_results:
                return f"「{query}」に関する検索結果が見つかりませんでした。"

            if self.valves.USE_EMBEDDING_RANKING:
                selected = self._select_relevant_pages(query, search_results)
                pages = self._fetch_pages_parallel(selected)
                chunks_with_meta = []
                for page in pages:
                    for chunk in _chunk_text_safe(
                        page["text"],
                        self.valves.CHUNK_SIZE,
                        self.valves.CHUNK_OVERLAP,
                        self.valves.MAX_PROCEDURE_CHARS,
                    ):
                        chunks_with_meta.append(
                            {
                                "text": chunk,
                                "title": page["title"],
                                "url": page["url"],
                            }
                        )

                top_chunks = self._rank_chunks_by_relevance(query, chunks_with_meta)
                if not top_chunks:
                    output_parts = [
                        f"■ {p['title']}\nURL: {p['url']}\n【内容】\n{p['text'][:800]}\n"
                        for p in pages
                    ]
                else:
                    grouped: dict[str, dict] = {}
                    for c in top_chunks:
                        key = c["url"] or c["title"]
                        grouped.setdefault(
                            key,
                            {"title": c["title"], "url": c["url"], "texts": []},
                        )
                        grouped[key]["texts"].append(c["text"])

                    output_parts = []
                    for g in grouped.values():
                        merged_text = "\n…\n".join(g["texts"])
                        output_parts.append(
                            f"■ {g['title']}\nURL: {g['url']}\n【関連抜粋】\n{merged_text}\n"
                        )
                return f"「{query}」のWeb検索結果:\n\n" + "\n".join(output_parts)
            else:
                top_results = search_results[: self.valves.PAGES_TO_FETCH]
                pages = self._fetch_pages_parallel(top_results)
                output_parts = [
                    f"■ {p['title']}\nURL: {p['url']}\n【本文】\n{p['text']}\n"
                    for p in pages
                ]
                return f"「{query}」のWeb検索結果:\n\n" + "\n".join(output_parts)

        except Exception:
            _log("search_web 例外発生:\n" + traceback.format_exc())
            return "検索処理中にエラーが発生しました。"