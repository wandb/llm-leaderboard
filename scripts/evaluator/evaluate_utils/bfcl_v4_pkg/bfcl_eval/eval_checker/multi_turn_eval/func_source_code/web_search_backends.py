"""Configurable search backends for BFCL v4 agentic web tasks."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_DUCKDUCKGO_ENDPOINT = "https://html.duckduckgo.com/html/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
SUPPORTED_BACKENDS = {"ddgs", "duckduckgo_html", "serpapi"}

_RATE_LOCK = threading.Lock()
_NEXT_REQUEST_AT = 0.0
_STATS_LOCK = threading.Lock()
_STATS: dict[str, int] = {
    "queries": 0,
    "cache_hits": 0,
    "network_requests": 0,
    "retries": 0,
    "errors": 0,
    "fetches": 0,
    "fetch_truncations": 0,
    "fetch_errors": 0,
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def configured_backend() -> str:
    backend = os.getenv("BFCL_WEB_SEARCH_BACKEND", "ddgs")
    backend = backend.strip().lower().replace("-", "_")
    aliases = {
        "direct": "ddgs",
        "direct_search": "ddgs",
        "multi_engine": "ddgs",
        "duckduckgo": "duckduckgo_html",
        "ddg": "duckduckgo_html",
        "ddg_html": "duckduckgo_html",
        "serp_api": "serpapi",
    }
    backend = aliases.get(backend, backend)
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported BFCL web search backend {backend!r}; "
            f"expected one of {sorted(SUPPORTED_BACKENDS)}"
        )
    return backend


def get_search_stats() -> dict[str, Any]:
    with _STATS_LOCK:
        return {
            "backend": configured_backend(),
            **_STATS,
        }


def reset_search_stats() -> None:
    global _NEXT_REQUEST_AT
    with _STATS_LOCK:
        for key in _STATS:
            _STATS[key] = 0
    with _RATE_LOCK:
        _NEXT_REQUEST_AT = 0.0


def _increment_stat(name: str) -> None:
    with _STATS_LOCK:
        _STATS[name] += 1


def record_fetch(*, truncated: bool = False, error: bool = False) -> None:
    _increment_stat("fetches")
    if truncated:
        _increment_stat("fetch_truncations")
    if error:
        _increment_stat("fetch_errors")


def _wait_for_request_slot() -> None:
    global _NEXT_REQUEST_AT
    min_interval = max(
        0.0,
        _env_float("BFCL_WEB_SEARCH_MIN_INTERVAL_SEC", 2.0),
    )
    jitter = max(
        0.0,
        _env_float("BFCL_WEB_SEARCH_JITTER_SEC", 0.25),
    )
    if min_interval <= 0 and jitter <= 0:
        return
    with _RATE_LOCK:
        now = time.monotonic()
        sleep_for = max(0.0, _NEXT_REQUEST_AT - now)
        random_delay = random.uniform(0.0, jitter) if jitter else 0.0
        _NEXT_REQUEST_AT = (
            max(now, _NEXT_REQUEST_AT) + min_interval + random_delay
        )
    if sleep_for > 0:
        time.sleep(sleep_for)


def _cache_path() -> Path | None:
    value = os.getenv("BFCL_WEB_SEARCH_CACHE_PATH", "").strip()
    return Path(value).expanduser().resolve() if value else None


def _cache_key(
    *,
    backend: str,
    keywords: str,
    max_results: int,
    region: str,
) -> str:
    payload = {
        "schema_version": 1,
        "backend": backend,
        "endpoint": (
            os.getenv(
                "BFCL_WEB_SEARCH_ENDPOINT",
                DEFAULT_DUCKDUCKGO_ENDPOINT,
            )
            if backend == "duckduckgo_html"
            else (
                "ddgs:"
                + os.getenv("BFCL_WEB_SEARCH_DDGS_BACKEND", "auto")
                if backend == "ddgs"
                else "serpapi:duckduckgo"
            )
        ),
        "keywords": keywords,
        "max_results": max_results,
        "region": region,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS search_cache (
            cache_key TEXT PRIMARY KEY,
            backend TEXT NOT NULL,
            keywords TEXT NOT NULL,
            region TEXT NOT NULL,
            max_results INTEGER NOT NULL,
            response_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """
    )
    return connection


def _read_cache(cache_key: str) -> list[dict[str, str]] | None:
    path = _cache_path()
    if path is None:
        return None
    ttl = max(0.0, _env_float("BFCL_WEB_SEARCH_CACHE_TTL_SEC", 0.0))
    with _open_cache(path) as connection:
        row = connection.execute(
            "SELECT response_json, created_at FROM search_cache "
            "WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
    if row is None:
        return None
    response_json, created_at = row
    if ttl and time.time() - float(created_at) > ttl:
        return None
    try:
        value = json.loads(response_json)
    except (TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, list) else None


def _write_cache(
    *,
    cache_key: str,
    backend: str,
    keywords: str,
    region: str,
    max_results: int,
    results: list[dict[str, str]],
) -> None:
    path = _cache_path()
    if path is None or not results:
        return
    with _open_cache(path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO search_cache (
                cache_key, backend, keywords, region, max_results,
                response_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                backend,
                keywords,
                region,
                max_results,
                json.dumps(results, ensure_ascii=False),
                time.time(),
            ),
        )


def _decode_duckduckgo_url(href: str) -> str:
    absolute = urljoin("https://duckduckgo.com", href)
    parsed = urlparse(absolute)
    redirect_target = parse_qs(parsed.query).get("uddg")
    return redirect_target[0] if redirect_target else absolute


def parse_duckduckgo_html(
    html: str,
    *,
    max_results: int,
) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for node in soup.select(".result"):
        link = node.select_one("a.result__a")
        if link is None:
            continue
        title = link.get_text(" ", strip=True)
        href = _decode_duckduckgo_url(str(link.get("href") or ""))
        if not title or not href.startswith(("http://", "https://")):
            continue
        if href in seen_urls:
            continue
        snippet_node = node.select_one(".result__snippet")
        snippet = (
            snippet_node.get_text(" ", strip=True)
            if snippet_node is not None
            else ""
        )
        results.append({"title": title, "href": href, "body": snippet})
        seen_urls.add(href)
        if len(results) >= max_results:
            break
    return results


def _duckduckgo_query(
    keywords: str,
    *,
    max_results: int,
    region: str,
) -> list[dict[str, str]]:
    endpoint = os.getenv(
        "BFCL_WEB_SEARCH_ENDPOINT",
        DEFAULT_DUCKDUCKGO_ENDPOINT,
    )
    timeout = max(
        1.0,
        _env_float("BFCL_WEB_SEARCH_TIMEOUT_SEC", 20.0),
    )
    response = requests.post(
        endpoint,
        data={"q": keywords, "kl": region},
        headers={
            "User-Agent": os.getenv(
                "BFCL_WEB_SEARCH_USER_AGENT",
                DEFAULT_USER_AGENT,
            ),
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=timeout,
        allow_redirects=True,
    )
    response.raise_for_status()
    results = parse_duckduckgo_html(
        response.text,
        max_results=max_results,
    )
    if not results:
        raise RuntimeError(
            "DuckDuckGo returned no parseable organic results"
        )
    return results


def _ddgs_query(
    keywords: str,
    *,
    max_results: int,
    region: str,
) -> list[dict[str, str]]:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("The ddgs backend requires the ddgs package") from exc

    timeout = max(
        1.0,
        _env_float("BFCL_WEB_SEARCH_TIMEOUT_SEC", 20.0),
    )
    backend = (
        os.getenv("BFCL_WEB_SEARCH_DDGS_BACKEND", "auto").strip() or "auto"
    )
    raw_results = DDGS(timeout=timeout).text(
        keywords,
        region=region,
        safesearch="moderate",
        max_results=max_results,
        backend=backend,
    )
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in raw_results:
        title = str(item.get("title") or "").strip()
        href = str(item.get("href") or item.get("url") or "").strip()
        if (
            not title
            or not href.startswith(("http://", "https://"))
            or href in seen_urls
        ):
            continue
        results.append(
            {
                "title": title,
                "href": href,
                "body": str(
                    item.get("body") or item.get("description") or ""
                ).strip(),
            }
        )
        seen_urls.add(href)
        if len(results) >= max_results:
            break
    if not results:
        raise RuntimeError("DDGS returned no organic results")
    return results


def _serpapi_query(
    keywords: str,
    *,
    max_results: int,
    region: str,
) -> list[dict[str, str]]:
    try:
        from serpapi import GoogleSearch
    except ImportError as exc:
        raise RuntimeError(
            "The serpapi backend requires google-search-results"
        ) from exc

    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "SERPAPI_API_KEY is required for the serpapi backend"
        )
    payload = GoogleSearch(
        {
            "engine": "duckduckgo",
            "q": keywords,
            "kl": region,
            "api_key": api_key,
        }
    ).get_dict()
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))

    results: list[dict[str, str]] = []
    for item in payload.get("organic_results", [])[:max_results]:
        title = str(item.get("title") or "").strip()
        href = str(item.get("link") or "").strip()
        if not title or not href:
            continue
        results.append(
            {
                "title": title,
                "href": href,
                "body": str(item.get("snippet") or "").strip(),
            }
        )
    if not results:
        raise RuntimeError("SerpAPI returned no organic results")
    return results


def search_web(
    keywords: str,
    *,
    max_results: int = 10,
    region: str = "wt-wt",
) -> list[dict[str, str]] | dict[str, str]:
    backend = configured_backend()
    keywords = str(keywords).strip()
    region = str(region or "wt-wt").strip()
    max_results = max(1, min(int(max_results or 10), 20))
    _increment_stat("queries")
    if not keywords:
        _increment_stat("errors")
        return {"error": "Search keywords must not be empty"}

    cache_key = _cache_key(
        backend=backend,
        keywords=keywords,
        max_results=max_results,
        region=region,
    )
    cached = _read_cache(cache_key)
    if cached is not None:
        _increment_stat("cache_hits")
        return cached

    max_attempts = max(
        1,
        _env_int("BFCL_WEB_SEARCH_MAX_ATTEMPTS", 4),
    )
    backoff = max(
        0.1,
        _env_float("BFCL_WEB_SEARCH_RETRY_BASE_SEC", 2.0),
    )
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        _wait_for_request_slot()
        _increment_stat("network_requests")
        try:
            if backend == "ddgs":
                results = _ddgs_query(
                    keywords,
                    max_results=max_results,
                    region=region,
                )
            elif backend == "duckduckgo_html":
                results = _duckduckgo_query(
                    keywords,
                    max_results=max_results,
                    region=region,
                )
            else:
                results = _serpapi_query(
                    keywords,
                    max_results=max_results,
                    region=region,
                )
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            _increment_stat("retries")
            time.sleep(backoff + random.uniform(0.0, backoff))
            backoff = min(backoff * 2, 60.0)
            continue

        _write_cache(
            cache_key=cache_key,
            backend=backend,
            keywords=keywords,
            region=region,
            max_results=max_results,
            results=results,
        )
        return results

    _increment_stat("errors")
    return {
        "error": (
            f"{backend} search failed after {max_attempts} attempts: "
            f"{last_error}"
        )
    }
