"""
title: SearXNG Web Search
author: user
description: SearXNG、Crawl4AI(DDG JSON抽出)、Wikipedia APIを並行実行し、AIがread_full_webpageで気になったリンクを追加取得できるWeb検索ツール。
version: 8.2.0
"""

import concurrent.futures
import json
import math
import re
import sys
import time
import traceback
from datetime import datetime
from urllib.parse import parse_qs, quote, urlparse

import requests
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _log(msg: str):
    line = f"[SEARXNG_SEARCH] {datetime.now().isoformat()} | {msg}"
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)


_SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
    ".mp4", ".mp3", ".zip", ".rar", ".exe", ".dmg",
)


def _should_skip_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(_SKIP_EXTENSIONS)


# 画像リンク ![alt](URL) はalt込みで丸ごと削除
_MD_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# 通常のMarkdownリンク [表示テキスト](URL) は表示テキストだけを残す
_MD_LINK_PATTERN = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# Markdownリンク記法に含まれない裸のURLも削除する
_BARE_URL_PATTERN = re.compile(r"https?://\S+")


def _extract_links(markdown: str, max_links: int = 15) -> list[tuple[str, str]]:
    """
    Markdown中の [表示テキスト](URL) からリンク一覧を抽出する。
    本文中のURLは_strip_linksで除去して読みやすくする一方、AIが
    read_full_webpageツールで「気になったリンクを開く」判断ができるよう、
    リンクテキストとURLの対応関係だけは別途一覧として保持しておく。
    """
    if not markdown:
        return []

    seen_urls = set()
    links: list[tuple[str, str]] = []

    for match in _MD_LINK_PATTERN.finditer(markdown):
        text = match.group(1).strip()
        full = match.group(0)
        url_match = re.search(r"\(([^)]*)\)$", full)
        url = url_match.group(1).strip() if url_match else ""

        if not text or not url or not url.startswith(("http://", "https://")):
            continue
        if _should_skip_url(url):
            continue
        if url in seen_urls:
            continue

        seen_urls.add(url)
        links.append((text, url))
        if len(links) >= max_links:
            break

    return links


def _strip_links(markdown: str) -> str:
    """
    Markdown中のリンクURLを取り除き、リンクテキストだけを残す。
    ニュースサイト等は見出しリンクの羅列でURLが大半を占めがちで、
    文字数上限のほとんどをURL文字列に食われてしまうため、
    切り詰め処理の前にこれを適用して実際の文章を優先的に残す。
    """
    if not markdown:
        return markdown

    text = _MD_IMAGE_PATTERN.sub("", markdown)
    text = _MD_LINK_PATTERN.sub(lambda m: m.group(1).strip(), text)
    text = _BARE_URL_PATTERN.sub("", text)

    text = re.sub(r"^\s*[-*]\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
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
    """連番の手順（1. 2. 3... / 手順1: / Step 1:）を検出し、割らずに保護する範囲を返す。"""
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


def _recursive_split(text: str, chunk_size: int, overlap: int, separators: list[str]) -> list[str]:
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
                chunks.extend(_recursive_split(piece, chunk_size, overlap, rest_separators))
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
        "\n\n", "\n",
        "。", "！", "？",
        ". ", "! ", "? ",
        "、", ", ",
        " ", "",
    ]
    return _recursive_split(text, chunk_size, overlap, separators)


def _chunk_text_safe(text: str, chunk_size: int, overlap: int, max_procedure_chars: int) -> list[str]:
    """手順ブロックは割らずに保持しつつ、それ以外は通常のチャンク分割にかける。"""
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
        if len(procedure_text) > max_procedure_chars:
            _log(
                f"[Chunking] 手順ブロックが上限({max_procedure_chars}文字)を超過"
                f"（実際{len(procedure_text)}文字）— 安全のため分割せずそのまま1チャンクとして保持します"
            )
        chunks.append(procedure_text)
        cursor = end

    remainder = text[cursor:]
    if remainder.strip():
        chunks.extend(_chunk_text(remainder, chunk_size, overlap))

    return [c for c in chunks if c.strip()]


class _PageCache:
    """キャッシュ値は {"text": str, "links": list[tuple[str, str]], "title": str} の辞書。"""

    def __init__(self, ttl_seconds: int = 600):
        self._store: dict[str, tuple[float, dict]] = {}
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

    def set(self, key: str, value: dict):
        self._store[key] = (time.time(), value)


_page_cache = _PageCache()


class Tools:
    class Valves(BaseModel):
        SEARXNG_URL: str = Field(
            default="http://searxng:8080/search",
            description="SearXNGのベースURL",
        )
        USE_DUCKDUCKGO_BACKUP: bool = Field(
            default=True,
            description="SearXNGが0件だった場合に、Crawl4AI経由でDuckDuckGoを代替として使うか",
        )
        ALWAYS_FETCH_WIKIPEDIA: bool = Field(
            default=True,
            description="Wikipedia APIから直接検索・本文取得を常時行うか",
        )
        WIKIPEDIA_MAX_RESULTS: int = Field(
            default=5,
            description="Wikipediaから取得する記事の上限数",
        )
        WIKIPEDIA_MAX_CHARS: int = Field(
            default=50000,
            description="Wikipedia1ページあたりの最大文字数（緩め設定）",
        )
        CRAWL4AI_URL: str = Field(
            default="http://crawl4ai:11235",
            description="Crawl4AIのベースURL",
        )
        CRAWL4AI_API_TOKEN: str = Field(
            default="your_secret_crawl4ai_token",
            description="docker-compose.ymlのCRAWL4AI_API_TOKENと同じ値",
        )
        EMBEDDING_SERVER_URL: str = Field(
            default="http://ruri-embedder:8001",
            description="Ruri v3 embeddingサーバーのベースURL",
        )
        SEARCH_RESULT_COUNT: int = Field(
            default=15,
            description="Web検索結果から取得する件数",
        )
        PAGES_TO_FETCH: int = Field(
            default=10,
            description="本文取得する上位Webページ数",
        )
        MAX_CHARS_PER_PAGE: int = Field(
            default=20000,
            description="1Webページあたりの文字数上限（緩め）",
        )
        MAX_LINKS_PER_PAGE: int = Field(
            default=15,
            description="1ページから提示するリンクの最大件数（read_full_webpageで開く候補）",
        )
        CHUNK_SIZE: int = Field(default=500, description="1チャンクの目安文字数")
        CHUNK_OVERLAP: int = Field(default=75, description="チャンク間オーバーラップ文字数")
        MAX_PROCEDURE_CHARS: int = Field(default=3000, description="手順ブロックの文字数上限目安")
        TOP_K_CHUNKS: int = Field(default=10, description="上位いくつのチャンクをLLMに渡すか")
        USE_EMBEDDING_RANKING: bool = Field(
            default=False,
            description="TrueならEmbeddingでページ・チャンク選定（低速・高精度）、Falseなら全文モード（高速）",
        )
        MAX_WORKERS: int = Field(default=10, description="並行取得の上限数")
        CACHE_TTL_SECONDS: int = Field(default=600, description="キャッシュ保持時間（秒）")
        SCRAPE_TIMEOUT: int = Field(default=15, description="1ページの取得タイムアウト（秒）")
        FETCH_TIME_BUDGET_SECONDS: float = Field(
            default=8.0,
            description="全ページ並行取得にかける全体の時間予算（秒）。超過分は打ち切ってスニペットで代用",
        )
        MIN_PAGES_REQUIRED: int = Field(default=1, description="時間予算超過時でも最低限待つページ数")
        EMBEDDING_TIMEOUT: int = Field(default=15, description="embeddingのタイムアウト（秒）")

    def __init__(self):
        _log("Tools が初期化されました")
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

    # =========================================================
    # Embedding（Ruri v3 embeddingサーバー経由・USE_EMBEDDING_RANKING=True時のみ使用）
    # =========================================================
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
            embeddings = data.get("embeddings", [])
            if len(embeddings) != len(texts):
                _log(f"[Embedding] 件数不一致: 入力{len(texts)}件 / 出力{len(embeddings)}件")
            return embeddings
        except Exception:
            _log("[Embedding] 失敗:\n" + traceback.format_exc())
            return [[] for _ in texts]

    def _select_relevant_pages(self, query: str, search_results: list[dict]) -> list[dict]:
        if not search_results:
            return []
        snippets = [r.get("content", "") or r.get("title", "") for r in search_results]
        query_vecs = self._embed([query], kind="query")
        snippet_vecs = self._embed(snippets, kind="document")

        if not query_vecs or not query_vecs[0]:
            _log("[ページ選定] embedding失敗のため、検索順位をそのまま使用")
            return search_results[: self.valves.PAGES_TO_FETCH]

        query_vec = query_vecs[0]
        scored = [
            (_cosine_similarity(query_vec, vec), item)
            for item, vec in zip(search_results, snippet_vecs)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored[: self.valves.PAGES_TO_FETCH]]

    def _rank_chunks_by_relevance(self, query: str, chunks_with_meta: list[dict]) -> list[dict]:
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
        return [meta for score, meta in scored[: self.valves.TOP_K_CHUNKS] if score > -1.0]

    # =========================================================
    # Wikipedia直接取得
    # =========================================================
    def _search_wikipedia_api(self, query: str) -> list[dict]:
        """Wikipedia MediaWiki APIで検索し、プレーンテキスト本文を取得する。"""
        try:
            _log(f"[Wikipedia] 直接取得開始: query='{query}'")
            headers = {"User-Agent": "OpenWebUI-SearchTool/1.0 (self-hosted)"}
            api_url = "https://ja.wikipedia.org/w/api.php"

            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": self.valves.WIKIPEDIA_MAX_RESULTS,
            }
            res = self.session.get(api_url, params=search_params, headers=headers, timeout=12)
            res.raise_for_status()
            search_items = res.json().get("query", {}).get("search", [])

            if not search_items:
                _log("[Wikipedia] 該当する記事が見つかりませんでした")
                return []

            titles = [item["title"] for item in search_items]
            titles_str = "|".join(titles)

            fetch_params = {
                "action": "query",
                "prop": "extracts",
                "explaintext": "true",
                "titles": titles_str,
                "format": "json",
                "redirects": 1,
            }
            res_pages = self.session.get(api_url, params=fetch_params, headers=headers, timeout=15)
            res_pages.raise_for_status()
            pages_data = res_pages.json().get("query", {}).get("pages", {})

            pages = []
            for page_id, page_info in pages_data.items():
                if page_id == "-1":
                    continue
                title = page_info.get("title", "")
                url = f"https://ja.wikipedia.org/wiki/{quote(title)}"
                body_text = page_info.get("extract", "").strip()

                if body_text:
                    if len(body_text) > self.valves.WIKIPEDIA_MAX_CHARS:
                        body_text = body_text[: self.valves.WIKIPEDIA_MAX_CHARS] + "\n...(以下省略)"

                    pages.append(
                        {
                            "title": f"[Wikipedia] {title}",
                            "url": url,
                            "text": body_text,
                            "links": [],  # プレーンテキストのためリンクなし
                            "is_wikipedia": True,
                        }
                    )

            _log(f"[Wikipedia] 取得成功: {len(pages)}件")
            return pages

        except Exception as e:
            _log(f"[Wikipedia] 取得エラー: {e}")
            return []

    # =========================================================
    # 検索（SearXNG / DuckDuckGoフォールバック）
    # =========================================================
    def _search_searxng_once(self, query: str, attempt: int) -> list[dict]:
        try:
            params = {"q": query, "format": "json", "lang": "ja"}
            response = self.session.get(self.valves.SEARXNG_URL, params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get("results", [])
            _log(f"[SearXNG] 試行{attempt}: {len(results)}件取得")
            return results
        except Exception as e:
            _log(f"[SearXNG] 試行{attempt}が失敗: {type(e).__name__}: {e}")
            return []

    def _search_searxng(self, query: str) -> list[dict]:
        """SearXNGを2回並行で呼び、結果をマージして重複除去する（一時的なブロック対策）。"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self._search_searxng_once, query, i + 1) for i in range(2)]
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

        _log(f"[SearXNG] マージ後: {len(results)}件")
        return results

    def _search_duckduckgo_via_crawl4ai(self, query: str) -> list[dict]:
        """SearXNGが0件だった場合のフォールバック。Crawl4AIのCSS抽出でDuckDuckGoから直接取得する。"""
        try:
            ddg_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            _log(f"[DuckDuckGo] Crawl4AI経由で取得開始: {ddg_url}")

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            }
            if self.valves.CRAWL4AI_API_TOKEN:
                headers["Authorization"] = f"Bearer {self.valves.CRAWL4AI_API_TOKEN}"

            extraction_strategy = {
                "type": "json_css",
                "params": {
                    "schema": {
                        "name": "DuckDuckGo Search Results",
                        "baseSelector": ".result",
                        "fields": [
                            {"name": "title", "selector": ".result__title", "type": "text"},
                            {"name": "url", "selector": ".result__a", "type": "attribute", "attribute": "href"},
                            {"name": "snippet", "selector": ".result__snippet", "type": "text"},
                        ],
                    }
                },
            }

            payload = {"urls": [ddg_url], "extraction_strategy": extraction_strategy}

            res = self.session.post(
                f"{self.valves.CRAWL4AI_URL}/crawl",
                json=payload,
                headers=headers,
                timeout=self.valves.SCRAPE_TIMEOUT,
            )
            res.raise_for_status()
            data = res.json()

            results_data = data.get("results", [data])
            page_result = results_data[0] if results_data else {}

            extracted_raw = page_result.get("extracted_content")
            if not extracted_raw:
                _log("[DuckDuckGo] extracted_contentが空でした")
                return []

            items = json.loads(extracted_raw) if isinstance(extracted_raw, str) else extracted_raw

            results = []
            seen_urls = set()

            for item in items:
                title = (item.get("title") or "").strip()
                raw_url = item.get("url") or ""
                snippet = (item.get("snippet") or "").strip()

                actual_url = raw_url
                if "duckduckgo.com/l/?" in raw_url or "uddg=" in raw_url:
                    parsed = urlparse(raw_url)
                    qs = parse_qs(parsed.query)
                    if "uddg" in qs:
                        actual_url = qs["uddg"][0]

                if (
                    not actual_url.startswith(("http://", "https://"))
                    or "duckduckgo.com" in actual_url
                    or actual_url in seen_urls
                    or _should_skip_url(actual_url)
                ):
                    continue

                title_clean = title.replace("\n", " ").strip()
                snippet_clean = snippet.replace("\n", " ").strip()

                if not title_clean or len(title_clean) < 2:
                    continue

                seen_urls.add(actual_url)
                results.append({"title": title_clean, "url": actual_url, "content": snippet_clean or title_clean})

                if len(results) >= self.valves.SEARCH_RESULT_COUNT:
                    break

            _log(f"[DuckDuckGo] 取得完了: {len(results)}件")
            return results

        except Exception as e:
            _log(f"[DuckDuckGo] 取得エラー: {e}")
            return []

    # =========================================================
    # 本文取得
    # =========================================================
    def _fetch_page_via_crawl4ai(self, item: dict) -> dict:
        """本文とページ内リンク一覧の両方を取得する。"""
        title = item.get("title") or item.get("url") or "(タイトルなし)"
        url = item.get("url", "")
        snippet = item.get("content", "(概要なし)")

        if not url or _should_skip_url(url):
            return {"title": title, "url": url, "text": snippet, "links": [], "is_wikipedia": False}

        cached = _page_cache.get(url)
        if cached is not None:
            _log(f"[本文取得] キャッシュヒット: {url}")
            return {
                "title": cached.get("title", title),
                "url": url,
                "text": cached["text"],
                "links": cached["links"],
                "is_wikipedia": False,
            }

        headers = {}
        if self.valves.CRAWL4AI_API_TOKEN:
            headers["Authorization"] = f"Bearer {self.valves.CRAWL4AI_API_TOKEN}"

        try:
            _log(f"[本文取得] Crawl4AIへリクエスト: {url}")
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

            meta_title = page_result.get("metadata", {}).get("title") or page_result.get("title")
            if meta_title and title in (url, "(タイトルなし)"):
                title = meta_title

            markdown = page_result.get("markdown", "")
            if isinstance(markdown, dict):
                markdown = markdown.get("fit_markdown") or markdown.get("raw_markdown", "")

            if not markdown:
                _log(f"[本文取得] Markdownが空でした: {url}")
                return {"title": title, "url": url, "text": snippet, "links": [], "is_wikipedia": False}

            links = _extract_links(markdown, max_links=self.valves.MAX_LINKS_PER_PAGE)

            before_len = len(markdown)
            cleaned = _strip_links(markdown)
            _log(f"[本文取得] リンク除去: {before_len}文字 → {len(cleaned)}文字, リンク{len(links)}件抽出")

            if len(cleaned) > self.valves.MAX_CHARS_PER_PAGE:
                cleaned = cleaned[: self.valves.MAX_CHARS_PER_PAGE] + "\n...(以下省略)"

            _page_cache.set(url, {"text": cleaned, "links": links, "title": title})
            _log(f"[本文取得] 成功 ({len(cleaned)}文字): {url}")
            return {"title": title, "url": url, "text": cleaned, "links": links, "is_wikipedia": False}

        except Exception as e:
            _log(f"[本文取得] スキップ/エラー ({type(e).__name__}): {url}")
            return {"title": title, "url": url, "text": snippet, "links": [], "is_wikipedia": False}

    def _fetch_pages_parallel(self, items: list[dict]) -> list[dict]:
        """
        全ページを並行取得するが、FETCH_TIME_BUDGET_SECONDSを過ぎたら
        遅いリクエストは待たずに切り上げる（レース方式）。
        """
        worker_count = min(len(items), self.valves.MAX_WORKERS) or 1
        _log(f"[本文取得] {len(items)}件を並行度{worker_count}で取得開始（時間予算{self.valves.FETCH_TIME_BUDGET_SECONDS}秒）")

        pages: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_item = {executor.submit(self._fetch_page_via_crawl4ai, item): item for item in items}

            done, not_done = concurrent.futures.wait(
                future_to_item.keys(),
                timeout=self.valves.FETCH_TIME_BUDGET_SECONDS,
                return_when=concurrent.futures.ALL_COMPLETED,
            )
            for future in done:
                pages.append(future.result())

            if len(pages) < self.valves.MIN_PAGES_REQUIRED and not_done:
                _log(f"[本文取得] 完了が{len(pages)}件のみのため、MIN_PAGES_REQUIREDまで追加待機")
                extra_done, not_done = concurrent.futures.wait(
                    not_done, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in extra_done:
                    pages.append(future.result())

            if not_done:
                skipped = [future_to_item[f].get("url", "") for f in not_done]
                _log(f"[本文取得] 時間予算超過のため{len(not_done)}件を打ち切り: {skipped}")
                for future in not_done:
                    item = future_to_item[future]
                    future.cancel()
                    pages.append(
                        {
                            "title": item.get("title", "(タイトルなし)"),
                            "url": item.get("url", ""),
                            "text": item.get("content", "(概要なし)"),
                            "links": [],
                            "is_wikipedia": False,
                        }
                    )

        _log(f"[本文取得] 完了: 通常{len(done)}件 / 打ち切り{len(not_done)}件")
        return pages

    # =========================================================
    # AIが気になったリンクを開くためのツール
    # =========================================================
    def read_full_webpage(self, urls: str) -> str:
        """
        指定したURLのページを実際に開き、本文全文を取得します。
        search_webの結果や、ページ内の「主なリンク」一覧に出てきたURLの中から、
        より詳しく知りたい・裏取りしたいページがあるときに使用してください。
        複数のURLを一度に開きたい場合は、カンマまたは改行区切りで指定できます。

        :param urls: 開きたいページのURL。1件でも、カンマや改行区切りで複数指定してもよい
        :return: 各ページの本文と、そのページ内で見つかったリンク一覧
        """
        url_list = [u.strip() for u in re.split(r"[,\n]", urls) if u.strip()]
        _log(f"read_full_webpage が呼び出されました urls={url_list}")

        if not url_list:
            return "有効なURLが指定されませんでした。"

        items = [{"title": u, "url": u, "content": "(概要なし)"} for u in url_list]
        pages = self._fetch_pages_parallel(items)

        return "\n".join(self._format_page_with_links(p) for p in pages)

    @staticmethod
    def _format_page_with_links(p: dict, label: str = "本文") -> str:
        """ページ本文に加え、ページ内で見つかったリンク一覧を付記する。"""
        block = f"■ {p['title']}\nURL: {p['url']}\n【{label}】\n{p['text']}\n"
        links = p.get("links") or []
        if links:
            link_lines = "\n".join(f"  - {text} : {href}" for text, href in links)
            block += f"【このページ内の主なリンク（read_full_webpageで開けます）】\n{link_lines}\n"
        return block

    # =========================================================
    # メインエントリーポイント
    # =========================================================
    def search_web(self, query: str) -> str:
        """
        指定したキーワードでWeb検索およびWikipedia本文取得を実行します。
        SearXNG・DuckDuckGo(フォールバック)・Wikipediaを並行して調べ、
        Wikipediaの結果は末尾にまとめて配置します。
        本文中のURLは読みやすさのため省略されていますが、各ページ末尾の
        リンク一覧やこの結果自体のURLは read_full_webpage ツールで開けます。

        :param query: 検索キーワード
        :return: 検索結果のテキスト
        """
        start = time.time()
        _log(f"=== search_web 開始: query='{query}' ===")

        footer = (
            "\n\n----------------------------------------\n"
            "【次のアクション】\n"
            "上記は検索結果の概要・抜粋です。より詳しく知りたい、あるいは裏取りしたい"
            "ページがあれば、そのURLを read_full_webpage ツールに渡して本文全体を取得してください。"
        )

        try:
            searxng_results: list[dict] = []
            ddg_results: list[dict] = []
            wiki_pages: list[dict] = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_searxng = executor.submit(self._search_searxng, query)
                future_ddg = (
                    executor.submit(self._search_duckduckgo_via_crawl4ai, query)
                    if self.valves.USE_DUCKDUCKGO_BACKUP
                    else None
                )
                future_wiki = (
                    executor.submit(self._search_wikipedia_api, query)
                    if self.valves.ALWAYS_FETCH_WIKIPEDIA
                    else None
                )

                searxng_results = future_searxng.result()
                if future_ddg:
                    ddg_results = future_ddg.result()
                if future_wiki:
                    wiki_pages = future_wiki.result()

            if searxng_results:
                search_results = searxng_results
            elif ddg_results:
                _log(f"SearXNGが0件のため、DuckDuckGo(Crawl4AI)の結果を採用（{len(ddg_results)}件）")
                search_results = ddg_results
            else:
                search_results = []

            web_pages: list[dict] = []
            if search_results:
                if self.valves.USE_EMBEDDING_RANKING:
                    selected = self._select_relevant_pages(query, search_results)
                    web_pages = self._fetch_pages_parallel(selected)
                else:
                    top_results = search_results[: self.valves.PAGES_TO_FETCH]
                    web_pages = self._fetch_pages_parallel(top_results)

            if not web_pages and not wiki_pages:
                return f"「{query}」に関する検索結果が見つかりませんでした。"

            all_pages = web_pages + wiki_pages

            if self.valves.USE_EMBEDDING_RANKING:
                chunks_with_meta = []
                for page in all_pages:
                    for chunk in _chunk_text_safe(
                        page["text"], self.valves.CHUNK_SIZE, self.valves.CHUNK_OVERLAP, self.valves.MAX_PROCEDURE_CHARS
                    ):
                        chunks_with_meta.append(
                            {
                                "text": chunk,
                                "title": page["title"],
                                "url": page["url"],
                                "is_wikipedia": page.get("is_wikipedia", False),
                            }
                        )

                top_chunks = self._rank_chunks_by_relevance(query, chunks_with_meta)
                links_by_url = {p["url"]: p.get("links") or [] for p in web_pages}

                if not top_chunks:
                    web_output = [self._format_page_with_links(p, label="内容") for p in web_pages]
                    wiki_output = [f"■ {p['title']}\nURL: {p['url']}\n【本文】\n{p['text']}\n" for p in wiki_pages]
                    output_parts = web_output + wiki_output
                else:
                    grouped: dict[str, dict] = {}
                    for c in top_chunks:
                        key = c["url"] or c["title"]
                        grouped.setdefault(
                            key,
                            {"title": c["title"], "url": c["url"], "texts": [], "is_wikipedia": c.get("is_wikipedia", False)},
                        )
                        grouped[key]["texts"].append(c["text"])

                    web_output, wiki_output = [], []
                    for g in grouped.values():
                        merged_text = "\n…\n".join(g["texts"])
                        block = f"■ {g['title']}\nURL: {g['url']}\n【関連抜粋】\n{merged_text}\n"
                        links = links_by_url.get(g["url"], [])
                        if links:
                            link_lines = "\n".join(f"  - {text} : {href}" for text, href in links)
                            block += f"【このページ内の主なリンク（read_full_webpageで開けます）】\n{link_lines}\n"
                        (wiki_output if g["is_wikipedia"] else web_output).append(block)

                    output_parts = web_output + wiki_output
            else:
                web_output = [self._format_page_with_links(p) for p in web_pages]
                wiki_output = [f"■ {p['title']}\nURL: {p['url']}\n【Wikipedia全文】\n{p['text']}\n" for p in wiki_pages]
                output_parts = web_output + wiki_output

            output = f"「{query}」のWeb検索・Wikipedia取得結果:\n\n" + "\n".join(output_parts) + footer
            elapsed = time.time() - start
            _log(f"=== search_web 正常終了（{elapsed:.1f}秒） ===")
            return output

        except Exception:
            _log("search_web 例外発生:\n" + traceback.format_exc())
            return "検索処理中にエラーが発生しました。"

