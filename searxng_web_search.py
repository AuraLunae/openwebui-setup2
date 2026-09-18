"""
title: SearXNG Web Search (Debug)
author: user
description: SearXNGを直接呼び出してWeb検索を行うツール。デバッグ用に詳細ログを出力する。
version: 1.1.0
"""

import sys
import traceback
from datetime import datetime

import requests
from pydantic import BaseModel, Field


def _log(msg: str):
    """
    Docker logs に確実に出すためのログ関数。
    - print(flush=True) で標準出力バッファリングを回避
    - 先頭に [SEARXNG_TOOL] という目印を付け、grepしやすくする
    - stderr にも出しておく（uvicornのログ設定によってstdoutが拾われない場合の保険）
    """
    line = f"[SEARXNG_TOOL] {datetime.now().isoformat()} | {msg}"
    print(line, flush=True)
    print(line, file=sys.stderr, flush=True)


class Tools:
    class Valves(BaseModel):
        SEARXNG_URL: str = Field(
            default="http://searxng:8080/search",
            description="SearXNGのベースURL",
        )
        RESULT_COUNT: int = Field(
            default=5,
            description="取得する検索結果の件数",
        )

    def __init__(self):
        _log("Tools.__init__ が呼ばれました（= ツールがロードされた証拠）")
        try:
            self.valves = self.Valves()
            _log(
                f"Valves初期化OK: SEARXNG_URL={self.valves.SEARXNG_URL}, RESULT_COUNT={self.valves.RESULT_COUNT}"
            )
        except Exception:
            _log("Valves初期化で例外発生:\n" + traceback.format_exc())
            raise

    def debug_ping(self) -> str:
        """
        デバッグ用の疎通確認関数。引数なしで呼び出すと固定文字列を返すだけ。
        この関数が呼ばれてログに出れば、少なくとも「モデルがこのツールを認識して
        呼び出す能力がある」ことが証明できる。

        :return: 固定の確認メッセージ
        """
        _log("debug_ping が呼び出されました")
        return "pong: SearXNG Web Search ツールは正しくロードされ、呼び出し可能です。"

    def search_web(self, query: str) -> str:
        """
        Web検索を実行し、上位の検索結果（タイトル・URL・概要）を返します。
        最新の情報、ニュース、天気、現在の状況など、モデルが知らない可能性のある
        質問に答える際に使用してください。

        :param query: 検索したいキーワード（日本語で可）
        :return: 検索結果のテキスト（タイトル・URL・概要のリスト）
        """
        _log(f"search_web が呼び出されました query={query!r}")

        try:
            url = self.valves.SEARXNG_URL
            params = {
                "q": query,
                "format": "json",
                "lang": "ja",
            }
            _log(f"SearXNGへリクエスト送信: url={url} params={params}")

            response = requests.get(url, params=params, timeout=10)
            _log(f"SearXNGからのレスポンス: status_code={response.status_code}")

            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            _log(f"取得件数: {len(results)}")

            results = results[: self.valves.RESULT_COUNT]

            if not results:
                _log("検索結果が0件でした")
                return f"「{query}」に関する検索結果が見つかりませんでした。"

            formatted = [f"「{query}」の検索結果:\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "(タイトルなし)")
                url_ = r.get("url", "")
                content = r.get("content", "(概要なし)")
                formatted.append(f"{i}. {title}\n   URL: {url_}\n   概要: {content}\n")

            result_text = "\n".join(formatted)
            _log(f"返却するテキストの先頭200文字: {result_text[:200]!r}")
            return result_text

        except requests.exceptions.Timeout:
            _log("タイムアウトが発生しました")
            return (
                "検索がタイムアウトしました。SearXNGが応答していない可能性があります。"
            )
        except requests.exceptions.ConnectionError as e:
            _log(f"接続エラー: {e}")
            return f"SearXNG ({self.valves.SEARXNG_URL}) に接続できませんでした。コンテナが起動しているか確認してください。"
        except Exception:
            tb = traceback.format_exc()
            _log("予期しない例外が発生しました:\n" + tb)
            return f"検索中にエラーが発生しました。詳細はサーバーログ（[SEARXNG_TOOL]タグ）を確認してください。"
