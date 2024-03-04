import os

from azure_gateway._chat import (
    OpenAIBody,
    openai_chat_request,
    parse_openai_chat_response,
)
from azure_gateway._embedding import (
    openai_embedding_request,
    parse_openai_embedding_response,
)
from azure_gateway._translate import (
    azure_translation_request,
    parse_azure_translation_response,
)


def test_chat_request():
    params = OpenAIBody(
        model_id="arti-cgpt-aim3516-chn",
        messages=[{"role": "user", "content": "Hello there"}],
    )
    res = openai_chat_request(os.environ["GATEWAY_PROJECT"], params, os.environ["GATEWAY_TOKEN"])
    assert res.status_code == 200

    parsed = parse_openai_chat_response(res)
    assert isinstance(parsed, str)


def test_embedding_request():
    res = openai_embedding_request(
        os.environ["GATEWAY_PROJECT"], "hello", os.environ["GATEWAY_TOKEN"]
    )
    assert res.status_code == 200

    parsed = parse_openai_embedding_response(res)
    assert isinstance(parsed, list)
    assert isinstance(parsed[0], float)


def test_translation_request():
    res = azure_translation_request(
        os.environ["GATEWAY_PROJECT"],
        "hello",
        target_languages=["es", "de"],
        original_language="en",
        token=os.environ["GATEWAY_TOKEN"],
    )
    assert res.status_code == 200

    parsed = parse_azure_translation_response(res)
    assert isinstance(parsed, list)
    assert isinstance(parsed[0], str)
