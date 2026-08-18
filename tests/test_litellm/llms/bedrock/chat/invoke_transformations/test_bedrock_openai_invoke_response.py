"""
Bedrock serves its `openai.*` models (gpt-oss, gpt-5.6-*) over the plain Invoke
route, and they speak OpenAI Chat Completions in both directions.

`transform_request` has always delegated those to AmazonBedrockOpenAIConfig, but
the response side had no `openai` branch, so a perfectly good body came back
through Titan's `results[0].outputText` and surfaced as
`BedrockException - ... 'NoneType' object is not subscriptable`. Streaming had
the mirrored gap: no shape in the base chunk parser matches an OpenAI chunk, so
every delta decoded to empty text.

Upstream: BerriAI/litellm#37132 (closed without a fix; main is still affected).
"""

from typing import get_args
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.constants import BEDROCK_INVOKE_PROVIDERS_LITERAL
from litellm.llms.bedrock.chat import invoke_handler
from litellm.llms.bedrock.chat.invoke_handler import (
    AmazonOpenAIStreamDecoder,
    make_call,
    make_sync_call,
)
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.utils import ModelResponse

MODEL = "us.openai.gpt-5.6-luna"
MESSAGES = [{"role": "user", "content": "hi"}]

# Verbatim from the run that broke deploy-new-model:
# https://github.com/Noma-Security/github-actions/actions/runs/32112359383
BEDROCK_OPENAI_BODY = {
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "annotations": [],
                "content": "Hi! How can I help you today?",
                "refusal": None,
                "role": "assistant",
            },
        }
    ],
    "created": 1787038693,
    "id": "chatcmpl-rchxtsxqviffhron5vmsvpkirkecbcsw7xz527v35jsntjepcc2a",
    "model": MODEL,
    "object": "chat.completion",
    "service_tier": "default",
    "usage": {"completion_tokens": 13, "prompt_tokens": 7, "total_tokens": 20},
}


def _transform_response(model: str, body: dict) -> ModelResponse:
    raw_response = httpx.Response(
        200,
        json=body,
        request=httpx.Request("POST", "https://bedrock-runtime.us-east-1.amazonaws.com"),
    )
    return AmazonInvokeConfig().transform_response(
        model=model,
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=MagicMock(),
        request_data={},
        messages=MESSAGES,
        optional_params={},
        litellm_params={},
        encoding=litellm.encoding,
    )


def test_transform_response_reads_bedrock_openai_chat_completion():
    response = _transform_response(MODEL, BEDROCK_OPENAI_BODY)

    assert response.choices[0].message.content == "Hi! How can I help you today?"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.prompt_tokens == 7
    assert response.usage.completion_tokens == 13


def test_provider_detection_still_routes_bedrock_openai_models_to_invoke():
    assert AmazonInvokeConfig.get_bedrock_invoke_provider(MODEL) == "openai"


@pytest.mark.parametrize("provider", get_args(BEDROCK_INVOKE_PROVIDERS_LITERAL))
def test_only_amazon_titan_is_parsed_as_titan(provider):
    """`transform_response` dispatches over a closed provider set, so anything it
    does not name explicitly must fail loudly rather than inherit Titan's parse.

    Feeding every provider a Titan-only body makes a silent fallthrough visible:
    a non-Amazon provider that answers with Titan's text took the wrong branch.
    """
    model = f"{provider}.test-model"
    assert AmazonInvokeConfig.get_bedrock_invoke_provider(model) == provider

    titan_body = {"results": [{"outputText": "titan-only-shape"}]}

    if provider == "amazon":
        assert _transform_response(model, titan_body).choices[0].message.content == "titan-only-shape"
        return

    try:
        content = _transform_response(model, titan_body).choices[0].message.content
    except Exception:
        return  # rejecting a body it cannot parse is the correct outcome
    assert content != "titan-only-shape", f"provider={provider} silently fell through to the Titan parse"


def test_unknown_provider_is_reported_as_such():
    """The Titan fallthrough used to blame Bedrock ('Error processing=<body>')
    for what is really an unrouted provider."""
    with pytest.raises(BedrockError) as exc_info:
        _transform_response("qwen3.test-model", {"anything": "here"})

    assert "Unknown provider" in str(exc_info.value)
    assert exc_info.value.status_code == 404


def test_stream_decoder_reads_openai_delta():
    chunk = {
        "id": "chatcmpl-abc",
        "object": "chat.completion.chunk",
        "created": 1787038693,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}],
    }

    parsed = AmazonOpenAIStreamDecoder(model=MODEL, sync_stream=True)._chunk_parser(chunk_data=chunk)

    assert parsed.choices[0].delta.content == "Hello"


def test_stream_decoder_carries_finish_reason_and_usage():
    chunk = {
        "id": "chatcmpl-abc",
        "object": "chat.completion.chunk",
        "created": 1787038693,
        "model": MODEL,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 13, "prompt_tokens": 7, "total_tokens": 20},
    }

    parsed = AmazonOpenAIStreamDecoder(model=MODEL, sync_stream=True)._chunk_parser(chunk_data=chunk)

    assert parsed.choices[0].finish_reason == "stop"
    assert parsed.usage.total_tokens == 20


def _mock_streaming_client(is_async: bool):
    response = MagicMock()
    response.status_code = 200
    client = MagicMock()
    client.post = AsyncMock(return_value=response) if is_async else MagicMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_make_call_selects_the_openai_decoder():
    client = _mock_streaming_client(is_async=True)

    with patch.object(invoke_handler, "AmazonOpenAIStreamDecoder") as decoder:
        await make_call(
            client=client,
            api_base=f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{MODEL}/invoke-with-response-stream",
            headers={},
            data="{}",
            model=MODEL,
            messages=MESSAGES,
            logging_obj=MagicMock(),
            bedrock_invoke_provider="openai",
        )

    decoder.assert_called_once_with(model=MODEL, sync_stream=False, json_mode=False)


def test_make_sync_call_selects_the_openai_decoder():
    client = _mock_streaming_client(is_async=False)

    with patch.object(invoke_handler, "AmazonOpenAIStreamDecoder") as decoder:
        make_sync_call(
            client=client,
            api_base=f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{MODEL}/invoke-with-response-stream",
            headers={},
            data="{}",
            signed_json_body=None,
            model=MODEL,
            messages=MESSAGES,
            logging_obj=MagicMock(),
            bedrock_invoke_provider="openai",
        )

    decoder.assert_called_once_with(model=MODEL, sync_stream=True, json_mode=False)
