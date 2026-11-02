import openai
import pytest

from emporio_agent.config import Settings
from emporio_agent.llm import OPENAI_DEFAULT_BASE_URL, LLMError, OpenAIClient


def settings(base_url):
    return Settings(api_key="test", base_url=base_url, model="gpt-4.1-mini", reference_now=None)


def test_empty_base_url_in_environment_falls_back_to_openai(monkeypatch):
    # load_dotenv turns "OPENAI_BASE_URL=" into an empty environment variable,
    # which the SDK would otherwise use as the URL.
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    client = OpenAIClient(settings(None))
    assert str(client._client.base_url).rstrip("/") == OPENAI_DEFAULT_BASE_URL


@pytest.mark.parametrize("url", ["https://api.anthropic.com/v1/", "http://localhost:11434/v1"])
def test_custom_base_url_is_used(url):
    assert str(OpenAIClient(settings(url))._client.base_url).rstrip("/") == url.rstrip("/")


@pytest.mark.parametrize(
    "body, expected",
    [({"type": "insufficient_quota", "code": "credit_balance_exhausted"}, "insufficient_quota"),
     ({"type": "requests", "code": "rate_limit_exceeded"}, "rate_limit")],
)
def test_429_distinguishes_no_credits_from_rate_limit(monkeypatch, body, expected):
    client = OpenAIClient(settings(None))

    def raise_429(**kwargs):
        # Built without an HTTP response: the client only reads .body.
        error = openai.RateLimitError.__new__(openai.RateLimitError)
        Exception.__init__(error, "429")
        error.body = body
        raise error

    monkeypatch.setattr(client._client.chat.completions, "create", raise_429)
    with pytest.raises(LLMError) as info:
        client.complete([{"role": "user", "content": "oi"}], [])
    assert info.value.error_type == expected
