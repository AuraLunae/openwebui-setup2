"""
title: SearXNG Search & Crawl4AI Scraper (Ruri v3, Procedure-Safe Chunking)
author: user
description: SearXNGでページを選定 → Crawl4AIで本文取得 → 手順ブロックを保護しつつチャンク分割 → Ruri v3 embeddingで関連度の高いチャンクを抽出するツール。
version: 4.4.0
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
    """
    URLの削除を行わず、余分な空白や改行の整理のみを行います。
    """
    if not markdown:
        return markdown

    text = markdown
    # 空行・余分な空白を整理
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


def _recursive_split(
    text: str, chunk_size: int, overlap: int, separators: list[str]
) -> list[str]:
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
    """日本語・英語どちらの文章にも対応した階層的セパレータで分割する（手順ブロックは含まない前提）。"""
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
        USE_EMBEDDING_RANKING: bool = Field(
            default=False,
            description=(
                "Trueならページ選定・チャンク選別にRuri v3 embeddingを使う（精度重視・遅い）。"
                "Falseなら埋め込みを一切呼ばず、SearXNGの順位そのまま・チャンク分割なしの"
                "全文取得モードで動く（速度重視）。ruri-embedderが重い間はFalse推奨。"
            ),
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
            description="1ページあたりのCrawl4AI取得タイムアウト秒数（個々のリクエストの上限）",
        )
        FETCH_TIME_BUDGET_SECONDS: float = Field(
            default=8.0,
            description=(
                "全ページ並行取得にかける全体の時間予算（秒）。これを過ぎたら、"
                "まだ終わっていない遅いページは待たずに切り上げ、スニペットで代用する"
                "（＝速く返ってきたものだけを採用するレース方式）"
            ),
        )
        MIN_PAGES_REQUIRED: int = Field(
            default=1,
            description="時間予算を過ぎても、最低これだけのページ取得が完了するまでは待つ（0件で返るのを防ぐ安全弁）",
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
            adapter = HTTPAdapter(
                pool_connections=10, pool_maxsize=10, max_retries=retry
            )
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
                _log(
                    f"[Embedding] 件数不一致: 入力{len(texts)}件 / 出力{len(embeddings)}件"
                )
            return embeddings
        except Exception:
            _log("[Embedding] 失敗:\n" + traceback.format_exc())
            return [[] for _ in texts]

    def _select_relevant_pages(
        self, query: str, search_results: list[dict]
    ) -> list[dict]:
        """本文取得前に、質問とスニペットの類似度でページを評価し、上位だけを本文取得対象にする。"""
        if not search_results:
            return []

        snippets = [r.get("content", "") or r.get("title", "") for r in search_results]
        _log(f"[Phase 2: ページ選定] {len(snippets)}件のスニペットをembedding計算中")

        query_vecs = self._embed([query], kind="query")
        snippet_vecs = self._embed(snippets, kind="document")

        if not query_vecs or not query_vecs[0]:
            _log(
                "[Phase 2: ページ選定] embedding失敗のため、SearXNGの順位をそのまま使用"
            )
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

    def _rank_chunks_by_relevance(
        self, query: str, chunks_with_meta: list[dict]
    ) -> list[dict]:
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
        top = [
            meta for score, meta in scored[: self.valves.TOP_K_CHUNKS] if score > -1.0
        ]

        _log(f"[Phase 5] 上位{len(top)}チャンクを採用")
        return top

    # =========================================================
    # 検索・本文取得
    # =========================================================
    def _search_searxng_once(self, query: str, attempt: int) -> list[dict]:
        """SearXNGへの単発リクエスト。エンジンのブロック等で結果が偏ることがあるため、
        呼び出し側で複数回呼んでマージする前提の内部関数。"""
        try:
            _log(
                f"[Phase 1: Web検索] SearXNGへリクエスト送信 (試行{attempt}): query='{query}'"
            )
            params = {"q": query, "format": "json", "lang": "ja"}
            response = self.session.get(
                self.valves.SEARXNG_URL, params=params, timeout=10
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            _log(f"[Phase 1: Web検索] 試行{attempt}: {len(results)}件取得")
            return results
        except Exception as e:
            _log(f"[Phase 1: Web検索] 試行{attempt}が失敗: {type(e).__name__}: {e}")
            return []

    def _search_searxng(self, query: str) -> list[dict]:
        """
        SearXNGを2回（並行）呼び、結果をマージして重複除去する。
        エンジンが一時的にCAPTCHA/ブロックで結果を返せないことがあるため、
        2回呼んで補い合うことで取りこぼしを減らす。
        """
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

        _log(
            f"[Phase 1: Web検索] 2回の呼び出しをマージ: {len(results)}件（重複除去後）"
        )
        return results

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

            markdown = page_result.get("markdown", "")
            if isinstance(markdown, dict):
                markdown = markdown.get("fit_markdown") or markdown.get(
                    "raw_markdown", ""
                )

            if not markdown:
                _log(f"[Phase 3: 本文取得] Markdownが空でした: {url}")
                return {
                    "title": title,
                    "url": url,
                    "text": snippet,
                    "source": "fallback_snippet",
                }

            before_len = len(markdown)
            markdown = _strip_links(markdown)
            _log(
                f"[Phase 3: 本文取得] フォーマット調整: {before_len}文字 → {len(markdown)}文字"
            )

            if len(markdown) > self.valves.MAX_CHARS_PER_PAGE:
                markdown = (
                    markdown[: self.valves.MAX_CHARS_PER_PAGE] + "\n...(以下省略)"
                )

            _page_cache.set(url, markdown)
            _log(f"[Phase 3: 本文取得] 成功 ({len(markdown)}文字): {url}")
            return {"title": title, "url": url, "text": markdown, "source": "scraped"}

        except Exception as e:
            _log(f"[Phase 3: 本文取得] スキップ/エラー ({type(e).__name__}): {url}")
            return {
                "title": title,
                "url": url,
                "text": snippet,
                "source": "fallback_snippet",
            }

    def _fetch_pages_parallel(self, items: list[dict]) -> list[dict]:
        """
        全ページを並行取得するが、FETCH_TIME_BUDGET_SECONDSを過ぎたら
        まだ完了していない遅いリクエストは待たずに切り上げる（レース方式）。
        取れなかった分はスニペットにフォールバックするので、結果が0件になることはない。
        ただしMIN_PAGES_REQUIRED件に達していない場合は、それを満たすまで
        時間予算を超えても待つ（安全弁）。
        """
        worker_count = min(len(items), self.valves.MAX_WORKERS) or 1
        _log(
            f"[Phase 3: 本文取得] {len(items)} 件を並行度{worker_count}で取得開始 "
            f"（時間予算 {self.valves.FETCH_TIME_BUDGET_SECONDS}秒）"
        )

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
                _log(
                    f"[Phase 3: 本文取得] 時間予算内の完了が{len(pages)}件のみ。"
                    f"MIN_PAGES_REQUIRED({self.valves.MIN_PAGES_REQUIRED})に達するまで追加で待機します"
                )
                extra_done, not_done = concurrent.futures.wait(
                    not_done, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in extra_done:
                    pages.append(future.result())

            if not_done:
                skipped_urls = [future_to_item[f].get("url", "") for f in not_done]
                _log(
                    f"[Phase 3: 本文取得] 時間予算超過のため{len(not_done)}件を打ち切り、"
                    f"スニペットにフォールバックします: {skipped_urls}"
                )
                for future in not_done:
                    item = future_to_item[future]
                    future.cancel()
                    pages.append(
                        {
                            "title": item.get("title", "(タイトルなし)"),
                            "url": item.get("url", ""),
                            "text": item.get("content", "(概要なし)"),
                            "source": "fallback_time_budget",
                        }
                    )

        _log(
            f"[Phase 3: 本文取得] 完了: 通常取得{len(done)}件 / 打ち切り{len(not_done)}件"
        )
        return pages

    def search_web(self, query: str) -> str:
        """
        Web検索を実行します。
        USE_EMBEDDING_RANKING=True の場合:
          SearXNGで候補取得 → Ruri v3 embeddingでページ選定 → 本文取得 →
          手順ブロック保護つきチャンク分割 → Ruri v3 embeddingでチャンク再評価
        USE_EMBEDDING_RANKING=False の場合（現在のデフォルト・高速モード）:
          SearXNGで候補取得 → 上位から順にそのまま本文取得（embedding未使用、チャンク分割なし）
        最新情報、具体的な出来事、事実確認、手順・操作方法の確認などに使用してください。

        :param query: 検索キーワード
        :return: 検索結果をまとめたテキスト
        """
        start = time.time()
        mode = "embedding" if self.valves.USE_EMBEDDING_RANKING else "全文(高速)"
        _log(f"=== search_web 開始: query='{query}' mode={mode} ===")
        try:
            search_results = self._search_searxng(query)
            if not search_results:
                return f"「{query}」に関する検索結果が見つかりませんでした。"

            if self.valves.USE_EMBEDDING_RANKING:
                output = self._search_web_embedding_mode(query, search_results)
            else:
                output = self._search_web_fulltext_mode(query, search_results)

            elapsed = time.time() - start
            _log(f"=== search_web 正常終了（{elapsed:.1f}秒、mode={mode}） ===")
            return output

        except Exception:
            tb = traceback.format_exc()
            _log("search_web 全体で例外発生:\n" + tb)
            return "検索処理中にエラーが発生しました。"

    def _search_web_fulltext_mode(self, query: str, search_results: list[dict]) -> str:
        """
        embeddingを一切呼ばない高速モード。SearXNGの順位のまま上位PAGES_TO_FETCH件を
        本文取得し、チャンク分割もせずMAX_CHARS_PER_PAGEで緩く切り詰めるだけ。
        ruri-embedderが重い/遅い間の代替として使う。
        """
        top_results = search_results[: self.valves.PAGES_TO_FETCH]
        pages = self._fetch_pages_parallel(top_results)

        output_parts = [
            f"■ {p['title']}\nURL: {p['url']}\n【本文】\n{p['text']}\n" for p in pages
        ]
        return (
            f"「{query}」のWeb検索結果（全文モード・{len(pages)}ページ）:\n\n"
            + "\n".join(output_parts)
        )

    def _search_web_embedding_mode(self, query: str, search_results: list[dict]) -> str:
        """Ruri v3 embeddingによるページ選定＋チャンク再評価を行う精度重視モード。"""
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
                    {"text": chunk, "title": page["title"], "url": page["url"]}
                )

        _log(f"[Phase 4: チャンク分割] 総チャンク数: {len(chunks_with_meta)}")

        top_chunks = self._rank_chunks_by_relevance(query, chunks_with_meta)

        if not top_chunks:
            _log("[Phase 5] ランキング結果が空のため、各ページ先頭にフォールバック")
            output_parts = [
                f"■ {p['title']}\nURL: {p['url']}\n【内容】\n{p['text'][:800]}\n"
                for p in pages
            ]
        else:
            grouped: dict[str, dict] = {}
            for c in top_chunks:
                key = c["url"] or c["title"]
                grouped.setdefault(
                    key, {"title": c["title"], "url": c["url"], "texts": []}
                )
                grouped[key]["texts"].append(c["text"])

            output_parts = []
            for g in grouped.values():
                merged_text = "\n…\n".join(g["texts"])
                output_parts.append(
                    f"■ {g['title']}\nURL: {g['url']}\n【関連抜粋】\n{merged_text}\n"
                )

        return f"「{query}」のWeb検索結果（関連度の高い部分を抜粋）:\n\n" + "\n".join(
            output_parts
        )