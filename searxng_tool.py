"""
title: SearXNG Search & Crawl4AI Scraper (Ruri v3, Procedure-Safe Chunking)
author: user
description: SearXNGでページを選定 → Crawl4AIで本文取得 → 手順ブロックを保護しつつチャンク分割 → Ruri v3 embeddingで関連度の高いチャンクを抽出するツール。
version: 4.0.0
"""

import concurrent.futures
import html as html_lib
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


def _should_skip_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(_SKIP_EXTENSIONS)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


# =========================================================
# チャンク分割（手順ブロックを保護する安全版）
# =========================================================

# 「1.」「2)」「手順1:」「Step 1:」のような連番行を検出する
_NUMBERED_LINE_PATTERN = re.compile(
    r"^\s*(?:\d+[.\)、]|手順\s*\d+[:：]|Step\s*\d+[:：])\s*.+$",
    re.MULTILINE,
)


def _find_procedure_blocks(text: str) -> list[tuple[int, int]]:
    """
    連番の手順（1. 2. 3... / 手順1: 手順2: / Step 1: Step 2: など）が
    2行以上連続している範囲を検出し、(開始位置, 終了位置) のリストを返す。
    この範囲はチャンク分割時に割らず、常にひとまとまりのまま扱う
    ——手順の途中で切れて一部のステップが欠落するのを防ぐため。
    """
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
    """LangChain不使用の自前実装。階層的セパレータで文脈境界を尊重しつつ分割する。"""
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
    """日本語・英語どちらの文章にも対応した階層的セパレータで分割する（手順ブロックは含まない前提）。"""
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
    """
    手順ブロック（連番の手順・ステップ）を検出し、それ以外の部分だけを通常のチャンク分割にかける。
    手順ブロックは分割せず、常に1チャンクとして丸ごと保持する
    ——途中のステップが欠落すると安全性に関わる情報（作業手順・操作手順等）が
    壊れる可能性があるため、サイズ上限を多少超えても割らない。
    """
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
            description="docker-compose.ymlのCRAWL4AI_API_TOKENと同じ値",
        )
        EMBEDDING_SERVER_URL: str = Field(
            default="http://ruri-embedder:8001",
            description="Ruri v3 embeddingサーバーのベースURL",
        )
        SEARCH_RESULT_COUNT: int = Field(
            default=8,
            description="SearXNGから取得する検索結果（候補）の件数",
        )
        PAGES_TO_FETCH: int = Field(
            default=4,
            description="候補の中から、スニペットの関連度が高い順に本文取得する上位ページ数",
        )
        MAX_CHARS_PER_PAGE: int = Field(
            default=20000,
            description="チャンク分割前の1ページあたりの文字数上限（緩め・安全弁）",
        )
        CHUNK_SIZE: int = Field(
            default=500,
            description="1チャンクあたりの目安文字数（手順ブロック以外に適用）",
        )
        CHUNK_OVERLAP: int = Field(
            default=75,
            description="チャンク間のオーバーラップ文字数",
        )
        MAX_PROCEDURE_CHARS: int = Field(
            default=3000,
            description="手順ブロックの文字数上限の目安。超えても分割はせず警告ログのみ出す",
        )
        TOP_K_CHUNKS: int = Field(
            default=8,
            description="全チャンクの中から、質問と関連度の高い上位いくつをLLMに渡すか",
        )
        MAX_WORKERS: int = Field(
            default=6,
            description="並行取得の上限数",
        )
        CACHE_TTL_SECONDS: int = Field(
            default=600,
            description="同一URLの本文取得結果をキャッシュしておく秒数",
        )
        SCRAPE_TIMEOUT: int = Field(
            default=15,
            description="1ページあたりのCrawl4AI取得タイムアウト秒数",
        )
        EMBEDDING_TIMEOUT: int = Field(
            default=15,
            description="embedding計算のタイムアウト秒数",
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

    def debug_ping(self) -> str:
        """デバッグ用の疎通確認関数。"""
        _log("debug_ping が呼び出されました")
        return "pong: SearXNG Web Search ツールは正しくロードされ、呼び出し可能です。"

    # =========================================================
    # Embedding（Ruri v3 embeddingサーバー経由）
    # =========================================================
    def _embed(self, texts: list[str], kind: str = "none") -> list[list[float]]:
        """
        自前のRuri v3 embeddingサーバー（/embed）を呼び出す。
        kindは "query"（検索クエリ側）または "document"（本文・スニペット側）を指定し、
        Ruri v3の1+3 prefixスキームに従ったプレフィックスを付与させる。
        """
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
        """本文取得前に、質問とスニペットの類似度でページを評価し、上位だけを本文取得対象にする。"""
        if not search_results:
            return []

        snippets = [r.get("content", "") or r.get("title", "") for r in search_results]
        _log(f"[Phase 2: ページ選定] {len(snippets)}件のスニペットをembedding計算中")

        query_vecs = self._embed([query], kind="query")
        snippet_vecs = self._embed(snippets, kind="document")

        if not query_vecs or not query_vecs[0]:
            _log("[Phase 2: ページ選定] embedding失敗のため、SearXNGの順位をそのまま使用")
            return search_results[: self.valves.PAGES_TO_FETCH]

        query_vec = query_vecs[0]
        scored = [
            (_cosine_similarity(query_vec, vec), item)
            for item, vec in zip(search_results, snippet_vecs)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [item for score, item in scored[: self.valves.PAGES_TO_FETCH]]

        _log(f"[Phase 2: ページ選定] 上位{len(top)}件を本文取得対象として採用")
        return top

    def _rank_chunks_by_relevance(self, query: str, chunks_with_meta: list[dict]) -> list[dict]:
        """全ページのチャンクの中から、質問との関連度が高い上位TOP_K_CHUNKSだけを返す。"""
        if not chunks_with_meta:
            return []

        _log(f"[Phase 5: Embeddingランキング] {len(chunks_with_meta)}チャンクを評価中")

        query_vecs = self._embed([query], kind="query")
        if not query_vecs or not query_vecs[0]:
            _log("[Phase 5] embedding失敗のため、先頭から採用します")
            return chunks_with_meta[: self.valves.TOP_K_CHUNKS]

        query_vec = query_vecs[0]
        chunk_texts = [c["text"] for c in chunks_with_meta]
        chunk_vecs = self._embed(chunk_texts, kind="document")

        scored = [
            (_cosine_similarity(query_vec, vec), meta)
            for meta, vec in zip(chunks_with_meta, chunk_vecs)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [meta for score, meta in scored[: self.valves.TOP_K_CHUNKS] if score > -1.0]

        _log(f"[Phase 5] 上位{len(top)}チャンクを採用")
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
            if len(results) >= self.valves.SEARCH_RESULT_COUNT:
                break

        _log(f"[Phase 1: Web検索] {len(results)} 件のURLを取得しました（重複除去後）")
        return results

    def _extract_markdown_from_crawl4ai_payload(self, payload: dict) -> str:
        """Crawl4AI の /crawl, /crawl_sync, /task/{id} で返る複数フォーマットを吸収する。"""
        if not isinstance(payload, dict):
            return ""

        candidates = []
        for key in ("results", "result"):
            value = payload.get(key)
            if value is not None:
                candidates.append(value)
        if not candidates:
            candidates.append(payload)

        for item in candidates:
            if isinstance(item, list):
                for entry in item:
                    if isinstance(entry, dict):
                        markdown = entry.get("markdown")
                        if isinstance(markdown, dict):
                            markdown = markdown.get("fit_markdown") or markdown.get("raw_markdown", "")
                        if markdown:
                            return markdown
                        if entry.get("text"):
                            return str(entry["text"])
            elif isinstance(item, dict):
                markdown = item.get("markdown")
                if isinstance(markdown, dict):
                    markdown = markdown.get("fit_markdown") or markdown.get("raw_markdown", "")
                if markdown:
                    return markdown
                if item.get("text"):
                    return str(item["text"])
                if item.get("content"):
                    return str(item["content"])

        return ""

    def _http_fallback_text(self, url: str) -> str:
        """Crawl4AI が使えない場合に、通常の HTTP 取得で本文テキストを拾う。"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
            "Accept-Language": "ja-JP,ja;q=0.9,en;q=0.8",
        }
        try:
            res = self.session.get(url, headers=headers, timeout=15)
            if res.status_code >= 400:
                return ""
            content_type = res.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type and "text/plain" not in content_type:
                return ""
            text = res.text
            text = re.sub(r"(?is)<script.*?</script>", " ", text)
            text = re.sub(r"(?is)<style.*?</style>", " ", text)
            text = re.sub(r"(?is)<[^>]+>", "\n", text)
            text = html_lib.unescape(text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n +", "\n", text)
            text = text.strip()
            return text
        except Exception:
            _log(f"[Phase 3: 本文取得] HTTPフォールバック取得失敗: {url}")
            return ""

    def _build_crawl4ai_crawler_params(self) -> dict:
        """Docker/CI 環境で Playwright が閉じられる問題を避けるため、安全な Chromium 引数を明示的に渡す。"""
        return {
            "browser_type": "chromium",
            "headless": True,
            "extra_args": [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-setuid-sandbox",
                "--disable-software-rasterizer",
            ],
        }

    def _fetch_page_via_crawl4ai(self, item: dict) -> dict:
        title = item.get("title", "(タイトルなし)")
        url = item.get("url", "")
        snippet = item.get("content", "(概要なし)")

        if not url or _should_skip_url(url):
            return {"title": title, "url": url, "text": snippet, "source": "snippet"}

        cached = _page_cache.get(url)
        if cached is not None:
            _log(f"[Phase 3: 本文取得] キャッシュヒット: {url}")
            return {"title": title, "url": url, "text": cached, "source": "cache"}

        headers = {}
        if self.valves.CRAWL4AI_API_TOKEN:
            headers["Authorization"] = f"Bearer {self.valves.CRAWL4AI_API_TOKEN}"

        try:
            _log(f"[Phase 3: 本文取得] Crawl4AIへリクエスト: {url}")
            payload = {
                "urls": [url],
                "js_code": [],
                "wait_for": "networkidle",
                "priority": 5,
                "ttl": 3600,
                "crawler_params": self._build_crawl4ai_crawler_params(),
            }

            markdown = ""
            for endpoint in ("/crawl_sync", "/crawl"):
                try:
                    res = self.session.post(
                        f"{self.valves.CRAWL4AI_URL}{endpoint}",
                        json=payload,
                        headers=headers,
                        timeout=self.valves.SCRAPE_TIMEOUT,
                    )
                    res.raise_for_status()
                    data = res.json()
                    if endpoint == "/crawl_sync":
                        markdown = self._extract_markdown_from_crawl4ai_payload(data)
                        if markdown:
                            break
                    else:
                        task_id = data.get("task_id")
                        if task_id:
                            deadline = time.time() + self.valves.SCRAPE_TIMEOUT
                            while time.time() < deadline:
                                status_res = self.session.get(
                                    f"{self.valves.CRAWL4AI_URL}/task/{task_id}",
                                    headers=headers,
                                    timeout=10,
                                )
                                status_res.raise_for_status()
                                status_data = status_res.json()
                                status = status_data.get("status")
                                if status == "completed":
                                    markdown = self._extract_markdown_from_crawl4ai_payload(status_data)
                                    if markdown:
                                        break
                                elif status == "failed":
                                    raise RuntimeError(status_data.get("error", "Crawl4AI task failed"))
                                time.sleep(1)
                            if markdown:
                                break
                        markdown = ""
                except Exception:
                    markdown = ""

            if not markdown:
                _log(f"[Phase 3: 本文取得] Crawl4AIが空/失敗のためHTTPフォールバック: {url}")
                markdown = self._http_fallback_text(url)

            if not markdown:
                _log(f"[Phase 3: 本文取得] Markdownが空でした: {url}")
                return {"title": title, "url": url, "text": snippet, "source": "fallback_snippet"}

            if len(markdown) > self.valves.MAX_CHARS_PER_PAGE:
                markdown = markdown[: self.valves.MAX_CHARS_PER_PAGE] + "\n...(以下省略)"

            _page_cache.set(url, markdown)
            _log(f"[Phase 3: 本文取得] 成功 ({len(markdown)}文字): {url}")
            return {"title": title, "url": url, "text": markdown, "source": "scraped"}

        except Exception as e:
            _log(f"[Phase 3: 本文取得] スキップ/エラー ({type(e).__name__}): {url}")
            return {"title": title, "url": url, "text": snippet, "source": "fallback_snippet"}

    def _fetch_pages_parallel(self, items: list[dict]) -> list[dict]:
        worker_count = min(len(items), self.valves.MAX_WORKERS) or 1
        _log(f"[Phase 3: 本文取得] {len(items)} 件を並行度{worker_count}で取得開始")

        pages = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(self._fetch_page_via_crawl4ai, item) for item in items]
            for future in concurrent.futures.as_completed(futures):
                pages.append(future.result())
        return pages

    def search_web(self, query: str) -> str:
        """
        Web検索を実行します。
        1. SearXNGで候補を取得
        2. スニペットをRuri v3 embeddingで評価し、関連度の高いページだけ本文取得
        3. 本文を手順ブロック保護つきでチャンク分割
        4. チャンクをRuri v3 embeddingで再評価し、上位だけをLLMに渡す
        最新情報、具体的な出来事、事実確認、手順・操作方法の確認などに使用してください。

        :param query: 検索キーワード
        :return: 関連度の高い抜粋を中心にまとめたテキスト
        """
        start = time.time()
        _log(f"=== search_web 開始: query='{query}' ===")
        try:
            search_results = self._search_searxng(query)
            if not search_results:
                return f"「{query}」に関する検索結果が見つかりませんでした。"

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
                    chunks_with_meta.append({"text": chunk, "title": page["title"], "url": page["url"]})

            _log(f"[Phase 4: チャンク分割] 総チャンク数: {len(chunks_with_meta)}")

            top_chunks = self._rank_chunks_by_relevance(query, chunks_with_meta)

            if not top_chunks:
                _log("[Phase 5] ランキング結果が空のため、各ページ先頭にフォールバック")
                output_parts = [
                    f"■ {p['title']}\nURL: {p['url']}\n【内容】\n{p['text'][:800]}\n" for p in pages
                ]
            else:
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