"""Regression guards for provider proxy and transport selection.

ChatGPT Codex uses the OpenAI SDK transport to avoid SSE stalls behind local
proxies. Other providers retain Hermes' keepalive client and NO_PROXY logic.
"""
from unittest.mock import patch

import httpx

from run_agent import AIAgent, _get_proxy_from_env, _get_proxy_for_base_url


def _make_agent():
    return AIAgent(
        api_key="test-key",
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openai-codex",
        model="gpt-5.4",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _extract_http_client(client_kwargs: dict):
    """_create_openai_client calls ``OpenAI(**client_kwargs)``; grab the injected client."""
    return client_kwargs.get("http_client")


def test_get_proxy_from_env_prefers_https_then_http_then_all(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    assert _get_proxy_from_env() is None

    monkeypatch.setenv("ALL_PROXY", "http://all:1")
    assert _get_proxy_from_env() == "http://all:1"

    monkeypatch.setenv("HTTP_PROXY", "http://http:2")
    assert _get_proxy_from_env() == "http://http:2"

    monkeypatch.setenv("HTTPS_PROXY", "http://https:3")
    assert _get_proxy_from_env() == "http://https:3"


def test_get_proxy_from_env_ignores_blank_values(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "   ")
    monkeypatch.setenv("HTTP_PROXY", "http://real-proxy:8080")
    assert _get_proxy_from_env() == "http://real-proxy:8080"


def test_get_proxy_from_env_normalizes_socks_alias(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:1080/")
    assert _get_proxy_from_env() == "socks5://127.0.0.1:1080/"


@patch("run_agent.OpenAI")
def test_create_openai_client_uses_sdk_transport_for_chatgpt_codex(
    mock_openai, monkeypatch
):
    """ChatGPT Codex must use the SDK's default proxy-aware transport.

    A custom httpx client can connect through a local HTTP proxy but then stall
    before the first Codex SSE event, while the official Codex CLI succeeds on
    the same host and proxy.
    """
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")

    agent = _make_agent()
    kwargs = {
        "api_key": "test-key",
        "base_url": "https://chatgpt.com/backend-api/codex",
    }
    agent._create_openai_client(kwargs, reason="test", shared=False)

    forwarded = mock_openai.call_args.kwargs
    assert "http_client" not in forwarded


@patch("run_agent.OpenAI")
def test_create_openai_client_uses_sdk_transport_for_chatgpt_codex_without_proxy(
    mock_openai, monkeypatch
):
    """ChatGPT Codex keeps the SDK transport even without proxy variables."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)

    agent = _make_agent()
    kwargs = {
        "api_key": "test-key",
        "base_url": "https://chatgpt.com/backend-api/codex",
    }
    agent._create_openai_client(kwargs, reason="test", shared=False)

    forwarded = mock_openai.call_args.kwargs
    assert "http_client" not in forwarded


@patch("run_agent.OpenAI")
def test_chatgpt_codex_sdk_transport_preserves_custom_tls(
    mock_openai, monkeypatch
):
    """Custom TLS verification must survive Codex transport selection."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)

    agent = _make_agent()
    kwargs = {
        "api_key": "test-key",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "ssl_verify": False,
    }
    agent._create_openai_client(kwargs, reason="test", shared=False)

    forwarded = mock_openai.call_args.kwargs
    http_client = _extract_http_client(forwarded)
    assert isinstance(http_client, httpx.Client)
    assert http_client._transport._pool._ssl_context.check_hostname is False
    http_client.close()


@patch("run_agent.OpenAI")
def test_create_openai_client_uses_plain_httpx_client_for_copilot(mock_openai, monkeypatch):
    """All providers now use a standard httpx.Client (no custom socket-options
    transport) so Copilot Claude chat-completions works without a host bypass."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)

    agent = _make_agent()
    kwargs = {
        "api_key": "test-key",
        "base_url": "https://api.githubcopilot.com",
    }
    agent._create_openai_client(kwargs, reason="test", shared=False)

    forwarded = mock_openai.call_args.kwargs
    http_client = _extract_http_client(forwarded)
    assert isinstance(http_client, httpx.Client)
    assert getattr(http_client._transport._pool, "_socket_options", None) is None
    http_client.close()


def test_get_proxy_for_base_url_returns_none_when_host_bypassed(monkeypatch):
    """NO_PROXY must suppress the proxy for matching base_urls.

    Regression for #14966: users running a local inference endpoint
    (Ollama, LM Studio, llama.cpp) with a global HTTPS_PROXY would see
    the keepalive client route loopback traffic through the proxy, which
    typically answers 502 for local hosts. NO_PROXY should opt those
    hosts out via stdlib ``urllib.request.proxy_bypass_environment``.
    """
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy",
                "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,192.168.0.0/16")

    # Local endpoint — must bypass the proxy.
    assert _get_proxy_for_base_url("http://127.0.0.1:11434/v1") is None
    assert _get_proxy_for_base_url("http://localhost:1234/v1") is None

    # Non-local endpoint — proxy still applies.
    assert _get_proxy_for_base_url("https://api.openai.com/v1") == "http://127.0.0.1:7897"


def test_get_proxy_for_base_url_returns_proxy_when_no_proxy_unset(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy",
                "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://corp:8080")
    assert _get_proxy_for_base_url("http://127.0.0.1:11434/v1") == "http://corp:8080"


def test_get_proxy_for_base_url_returns_none_when_proxy_unset(monkeypatch):
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy",
                "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")
    assert _get_proxy_for_base_url("http://127.0.0.1:11434/v1") is None
    assert _get_proxy_for_base_url("https://api.openai.com/v1") is None


@patch("run_agent.OpenAI")
def test_create_openai_client_bypasses_proxy_for_no_proxy_host(mock_openai, monkeypatch):
    """E2E: with HTTPS_PROXY + NO_PROXY=localhost, a local base_url gets a
    keepalive client with NO HTTPProxy mount."""
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
                "https_proxy", "http_proxy", "all_proxy",
                "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1")

    agent = _make_agent()
    kwargs = {
        "api_key": "***",
        "base_url": "http://127.0.0.1:11434/v1",
    }
    agent._create_openai_client(kwargs, reason="test", shared=False)

    forwarded = mock_openai.call_args.kwargs
    http_client = _extract_http_client(forwarded)
    assert isinstance(http_client, httpx.Client)
    pool_types = [
        type(mount._pool).__name__
        for mount in http_client._mounts.values()
        if mount is not None and hasattr(mount, "_pool")
    ]
    assert "HTTPProxy" not in pool_types, (
        "NO_PROXY host must not route through HTTPProxy; pools were %r" % (pool_types,)
    )
    http_client.close()
