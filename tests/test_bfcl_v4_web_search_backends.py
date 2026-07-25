import importlib
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = (
    ROOT
    / "scripts"
    / "evaluator"
    / "evaluate_utils"
    / "bfcl_v4_pkg"
)
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

MODULE_NAME = (
    "bfcl_eval.eval_checker.multi_turn_eval.func_source_code."
    "web_search_backends"
)
backends = importlib.import_module(MODULE_NAME)


DDG_HTML = """
<html><body>
  <div class="result">
    <a class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone">
      First result
    </a>
    <div class="result__snippet">Useful first snippet.</div>
  </div>
  <div class="result">
    <a class="result__a" href="https://example.org/two">Second result</a>
    <div class="result__snippet">Useful second snippet.</div>
  </div>
  <div class="result">
    <a class="result__a" href="https://example.org/two">Duplicate</a>
  </div>
</body></html>
"""


class FakeResponse:
    def __init__(self, text=DDG_HTML):
        self.text = text

    def raise_for_status(self):
        return None


@pytest.fixture(autouse=True)
def clean_search_environment(monkeypatch):
    values = {
        "BFCL_WEB_SEARCH_BACKEND": "duckduckgo_html",
        "BFCL_WEB_SEARCH_ENDPOINT": "https://html.duckduckgo.com/html/",
        "BFCL_WEB_SEARCH_TIMEOUT_SEC": "5",
        "BFCL_WEB_SEARCH_MAX_ATTEMPTS": "2",
        "BFCL_WEB_SEARCH_RETRY_BASE_SEC": "0.1",
        "BFCL_WEB_SEARCH_MIN_INTERVAL_SEC": "0",
        "BFCL_WEB_SEARCH_JITTER_SEC": "0",
        "BFCL_WEB_SEARCH_CACHE_PATH": "",
        "BFCL_WEB_SEARCH_CACHE_TTL_SEC": "0",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    backends.reset_search_stats()


def test_parse_duckduckgo_html_returns_serpapi_compatible_results():
    results = backends.parse_duckduckgo_html(DDG_HTML, max_results=10)

    assert results == [
        {
            "title": "First result",
            "href": "https://example.com/one",
            "body": "Useful first snippet.",
        },
        {
            "title": "Second result",
            "href": "https://example.org/two",
            "body": "Useful second snippet.",
        },
    ]


def test_direct_search_sends_query_region_and_honors_max_results(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr(backends.requests, "post", fake_post)

    results = backends.search_web(
        "Taiwan benchmark",
        max_results=1,
        region="tw-tzh",
    )

    assert results == [
        {
            "title": "First result",
            "href": "https://example.com/one",
            "body": "Useful first snippet.",
        }
    ]
    assert calls[0][0] == "https://html.duckduckgo.com/html/"
    assert calls[0][1]["data"] == {
        "q": "Taiwan benchmark",
        "kl": "tw-tzh",
    }
    assert calls[0][1]["timeout"] == 5.0
    assert backends.get_search_stats() == {
        "backend": "duckduckgo_html",
        "queries": 1,
        "cache_hits": 0,
        "network_requests": 1,
        "retries": 0,
        "errors": 0,
        "fetches": 0,
        "fetch_truncations": 0,
        "fetch_errors": 0,
    }


def test_direct_search_retries_then_succeeds(monkeypatch):
    attempts = 0

    def flaky_post(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise backends.requests.ConnectionError("temporary")
        return FakeResponse()

    monkeypatch.setattr(backends.requests, "post", flaky_post)
    monkeypatch.setattr(backends.time, "sleep", lambda _: None)
    monkeypatch.setattr(backends.random, "uniform", lambda _a, _b: 0.0)

    results = backends.search_web("retry me")

    assert isinstance(results, list)
    assert attempts == 2
    assert backends.get_search_stats()["retries"] == 1
    assert backends.get_search_stats()["network_requests"] == 2


def test_ddgs_backend_normalizes_multi_engine_results(monkeypatch):
    captured = {}

    class FakeDDGS:
        def __init__(self, *, timeout):
            captured["timeout"] = timeout

        def text(self, keywords, **kwargs):
            captured["keywords"] = keywords
            captured.update(kwargs)
            return [
                {
                    "title": "First DDGS result",
                    "href": "https://example.com/ddgs",
                    "body": "DDGS snippet",
                },
                {
                    "title": "Duplicate",
                    "href": "https://example.com/ddgs",
                    "body": "Ignored",
                },
            ]

    fake_ddgs = types.ModuleType("ddgs")
    fake_ddgs.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_ddgs)
    monkeypatch.setenv("BFCL_WEB_SEARCH_BACKEND", "ddgs")
    monkeypatch.setenv("BFCL_WEB_SEARCH_DDGS_BACKEND", "auto")

    results = backends.search_web(
        "multi-engine search",
        max_results=3,
        region="tw-tzh",
    )

    assert results == [
        {
            "title": "First DDGS result",
            "href": "https://example.com/ddgs",
            "body": "DDGS snippet",
        }
    ]
    assert captured == {
        "timeout": 5.0,
        "keywords": "multi-engine search",
        "region": "tw-tzh",
        "safesearch": "moderate",
        "max_results": 3,
        "backend": "auto",
    }


def test_successful_results_are_reused_from_sqlite_cache(
    tmp_path, monkeypatch
):
    cache_path = tmp_path / "search.sqlite3"
    monkeypatch.setenv("BFCL_WEB_SEARCH_CACHE_PATH", str(cache_path))
    calls = 0

    def fake_post(*args, **kwargs):
        nonlocal calls
        calls += 1
        return FakeResponse()

    monkeypatch.setattr(backends.requests, "post", fake_post)

    first = backends.search_web("cached query", max_results=2)
    second = backends.search_web("cached query", max_results=2)

    assert first == second
    assert calls == 1
    assert cache_path.is_file()
    stats = backends.get_search_stats()
    assert stats["queries"] == 2
    assert stats["cache_hits"] == 1
    assert stats["network_requests"] == 1


def test_empty_query_and_exhausted_retries_return_tool_errors(monkeypatch):
    assert backends.search_web(" ") == {
        "error": "Search keywords must not be empty"
    }

    monkeypatch.setattr(
        backends.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            backends.requests.Timeout("slow")
        ),
    )
    monkeypatch.setattr(backends.time, "sleep", lambda _: None)
    result = backends.search_web("always fails")

    assert result["error"].startswith(
        "duckduckgo_html search failed after 2 attempts:"
    )
    assert backends.get_search_stats()["errors"] == 2


@pytest.mark.parametrize(
    "configured, expected",
    [
        ("ddgs", "ddgs"),
        ("direct-search", "ddgs"),
        ("duckduckgo", "duckduckgo_html"),
        ("ddg-html", "duckduckgo_html"),
        ("serp-api", "serpapi"),
    ],
)
def test_backend_aliases(monkeypatch, configured, expected):
    monkeypatch.setenv("BFCL_WEB_SEARCH_BACKEND", configured)
    assert backends.configured_backend() == expected


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("BFCL_WEB_SEARCH_BACKEND", "mystery")
    with pytest.raises(ValueError, match="Unsupported BFCL web search"):
        backends.configured_backend()


def test_serpapi_backend_normalizes_sdk_results(monkeypatch):
    captured = {}

    class FakeGoogleSearch:
        def __init__(self, params):
            captured.update(params)

        def get_dict(self):
            return {
                "organic_results": [
                    {
                        "title": "Serp result",
                        "link": "https://example.com/serp",
                        "snippet": "Serp snippet",
                    }
                ]
            }

    fake_serpapi = types.ModuleType("serpapi")
    fake_serpapi.GoogleSearch = FakeGoogleSearch
    monkeypatch.setitem(sys.modules, "serpapi", fake_serpapi)
    monkeypatch.setenv("BFCL_WEB_SEARCH_BACKEND", "serpapi")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")

    results = backends.search_web(
        "provider-selectable search",
        max_results=1,
        region="tw-tzh",
    )

    assert results == [
        {
            "title": "Serp result",
            "href": "https://example.com/serp",
            "body": "Serp snippet",
        }
    ]
    assert captured == {
        "engine": "duckduckgo",
        "q": "provider-selectable search",
        "kl": "tw-tzh",
        "api_key": "test-key",
    }


def test_official_tool_removes_snippets_for_no_snippet_scenario(
    monkeypatch,
):
    web_search = importlib.import_module(
        "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search"
    )
    monkeypatch.setattr(
        web_search,
        "search_web",
        lambda *args, **kwargs: [
            {
                "title": "Title",
                "href": "https://example.com",
                "body": "Snippet",
            }
        ],
    )
    tool = web_search.WebSearchAPI()
    tool._load_scenario({"show_snippet": False})

    assert tool.search_engine_query("query") == [
        {
            "title": "Title",
            "href": "https://example.com",
        }
    ]


def test_fetch_url_content_caps_download_and_tool_output(monkeypatch):
    web_search = importlib.import_module(
        "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search"
    )

    class StreamingResponse:
        encoding = "utf-8"

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            assert chunk_size == 64 * 1024
            yield b"abcdefgh"
            yield b"ijklmnop"

    monkeypatch.setattr(
        web_search.requests,
        "get",
        lambda *args, **kwargs: StreamingResponse(),
    )
    monkeypatch.setenv("BFCL_WEB_FETCH_MAX_BYTES", "10")
    monkeypatch.setenv("BFCL_WEB_FETCH_MAX_CHARS", "5")
    backends.reset_search_stats()

    result = web_search.WebSearchAPI().fetch_url_content(
        "https://example.com/large",
        mode="raw",
    )

    assert result["content"].startswith("abcde")
    assert "Content truncated" in result["content"]
    stats = backends.get_search_stats()
    assert stats["fetches"] == 1
    assert stats["fetch_truncations"] == 1
    assert stats["fetch_errors"] == 0


def test_fetch_url_content_markdown_reports_missing_dependency(monkeypatch):
    web_search = importlib.import_module(
        "bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search"
    )

    class StreamingResponse:
        encoding = "utf-8"

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"<h1>Title</h1>"

    monkeypatch.setattr(
        web_search.requests,
        "get",
        lambda *args, **kwargs: StreamingResponse(),
    )
    monkeypatch.setitem(sys.modules, "html2text", None)

    result = web_search.WebSearchAPI().fetch_url_content(
        "https://example.com/markdown",
        mode="markdown",
    )

    assert "requires the html2text package" in result["error"]
