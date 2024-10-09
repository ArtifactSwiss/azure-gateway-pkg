import json
import warnings
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, field_validator

from ._constants import ENDPOINT_URL


class BytesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return obj.decode("utf-8")
        return json.JSONEncoder.default(self, obj)


class OpenAIBody(BaseModel):
    model_config = ConfigDict(protected_namespaces=(""))
    user_id: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    model_region: str | None = None
    messages: list[dict[str, Any]]
    openai_params: dict[str, Any] | None = None
    api_version: str | None = "2023-05-15"
    direct_http_call: bool | None = False

    @field_validator("messages")
    @classmethod
    def must_be_in_openai_format(cls, v: str) -> str:
        for i in v:
            if "role" not in i:
                raise ValueError('key "role" must be present')
            if i["role"] not in ["assistant", "user", "system"]:
                raise ValueError('key "role" must be one of assistant, user, system')
            if "content" not in i:
                raise ValueError('key "content" must be present')
        return v

    def to_json(self):
        return json.dumps({k: v for k, v in dict(self).items() if v is not None}, cls=BytesEncoder)


def openai_chat_request(project_id: str, params: OpenAIBody | dict, token: str):
    """
    DEPRECATED: Send a request to the specified OpenAI chat API.

    :param project_id: Project ID to log request to.
    :param params: The body parameters for the API (see docs).
    :param token: Authorization token for the API.
    :return: The response from the API.
    """
    warnings.warn(
        message="openai_chat_request() is deprecated and will be removed. Use the OpenAIGatewayLLM class instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    url_with_project_id = f"{ENDPOINT_URL}/openai-chat?project_id={project_id}"

    if isinstance(params, dict):
        params = OpenAIBody(**params)
    payload = params.to_json()
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    response = requests.request("POST", url_with_project_id, headers=headers, data=payload)
    response.raise_for_status()
    return response


def parse_openai_chat_response(response: requests.Response) -> str:
    """DEPRECATED: Parse a chat response to only the chat response."""
    warnings.warn(
        message="parse_openai_chat_response() is deprecated and will be removed. Use the OpenAIGatewayLLM class instead.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    if not isinstance(response, requests.Response):
        raise TypeError("Expected a requests.Response object")

    if response.status_code != requests.codes.ok:
        raise ValueError(f"Expected a successful HTTP response, got {response.status_code}")

    try:
        response_data = response.json()
    except json.JSONDecodeError:
        raise ValueError("Failed to decode JSON from response")

    if not response_data.get("choices") or not isinstance(response_data["choices"], list):
        raise ValueError('Response JSON does not contain "choices" list')

    if len(response_data["choices"]) == 0:
        raise ValueError('"choices" list is empty')

    choice = response_data["choices"][0]
    if "message" not in choice or "content" not in choice["message"]:
        raise ValueError('Expected "message" and "content" keys in the response')

    return choice["message"]["content"]
