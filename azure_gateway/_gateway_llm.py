import json
import logging
import os
import time
from typing import Callable, Any

import requests
import tiktoken
from langchain_core.language_models.llms import LLM
from openai import AzureOpenAI
from requests_sse import EventSource

from ._exceptions import OutOfQuotaError
from ._logger import LoggerWrapper


Callback = Callable[[str], None]
ServiceResponse = tuple[str, int, int]


class BaseGatewayLLM(LLM):
    """Base class for Gateway LLM"""

    project_id: str
    model_id: str
    model_type: str
    conversation_history: list
    gateway_auth_token: str
    alert_log_threshold: float = 0.9
    logger: logging.Logger | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = LoggerWrapper(self.logger)

    @property
    def _llm_type(self) -> str:
        return "custom"

    @property
    def _identifying_params(self):
        """Get the identifying parameters."""
        return {}

    def _call(self, prompt: str, callback: Callback | None, **_):
        self._query_gateway_safely(query_function=self._check_quota)

        self.log.info(f"PROMPT\n{self.log.indent_text(prompt)}")
        self.conversation_history.append({"role": "user", "content": prompt})

        response_text, prompt_tokens, completion_tokens = self._call_service(callback=callback)

        self._query_gateway_safely(
            query_function=lambda: self._update_quota(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )
        )
        self.log.info(f"RESPONSE\n{self.log.indent_text(response_text)}")

        self.conversation_history.append({"role": "assistant", "content": response_text})

        return response_text

    def _call_service(self, callback: Callback | None) -> ServiceResponse:
        raise NotImplementedError()

    def call(self, prompt: str, detail: str, callback: Callback | None) -> str:
        """Wrap-around for Langchain's invoke method."""
        start = time.time()
        response = self.invoke(input=prompt, config=None, callback=callback)
        self.log.info(f"CALLTIME - {detail} - {time.time() - start}")
        return response

    def _check_quota(self):
        """Check the current usage quota and raise an error if the project is out of quota."""
        quota_check_response = requests.get(
            os.environ["GATEWAY_HIGH_TRAFFIC_URL"],
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.gateway_auth_token}",
            },
            params={"project_id": self.project_id},
            timeout=10,
        )
        quota_check_response.raise_for_status()

        quota_check_json = json.loads(quota_check_response.text)
        if (
            "within_quota" not in quota_check_json
            or "quota" not in quota_check_json
            or "usage" not in quota_check_json
        ):
            raise ValueError("invalid response from quota-checking endpoint")

        within_quota = quota_check_json["within_quota"]
        if not within_quota:
            raise OutOfQuotaError(
                "The LLM usage quota has been exceeded and the service can no longer be used."
            )

        usage = quota_check_json["usage"]
        quota = quota_check_json["quota"]
        if usage / quota >= self.alert_log_threshold:
            self.log.warning(f"Gateway quota is almost used up: {usage}$ / {quota}$")

    def _update_quota(self, prompt_tokens: int, completion_tokens: int):
        """Update the usage quota with the usage tokens specified in the given chat completion."""
        additional_usage = [prompt_tokens, completion_tokens]

        quota_update_response = requests.post(
            os.environ["GATEWAY_HIGH_TRAFFIC_URL"],
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.gateway_auth_token}",
            },
            params={"project_id": self.project_id},
            json={"model_id": self.model_id, "usage_quantity": additional_usage},
            timeout=10,
        )
        quota_update_response.raise_for_status()

    def _query_gateway_safely(self, query_function: Callable):
        """
        Execute the given gateway query function in a safe way.

        If the Gateway call fails due to the quota being used up, raise OutOfQuotaError.
        If the Gateway call fails due to some internal problems with the Gateway, log the error and continue.
        """
        try:
            query_function()
        except OutOfQuotaError:
            raise
        except Exception as e:
            self.log.error("there was an error querying the gateway")
            self.log.error(e)

    @staticmethod
    def _parse_chat_response(response: requests.Response) -> str:
        """Parse a chat response to only the chat response."""
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


class OpenAIGatewayLLM(BaseGatewayLLM):
    client: AzureOpenAI
    temperature: float = 0.7

    def _call_service(self, callback: Callback | None) -> ServiceResponse:
        if callback is None:
            return self._regular_chat_completion()

        return self._streaming_chat_completion(callback)

    def _regular_chat_completion(self):
        self.log.info("CALL - regular chat completion")
        response = self.client.chat.completions.create(
            model=self.model_id, messages=self.conversation_history, temperature=self.temperature
        )
        req_response = requests.Response()
        req_response.status_code = 200
        req_response._content = response.model_dump_json().encode()
        response_text = self.parse_chat_response(req_response)
        return response_text, response.usage.prompt_tokens, response.usage.completion_tokens

    def _streaming_chat_completion(self, callback: Callable):
        """
        If there is a callback given, we stream the response.

        In this case, we need to compute the number of tokens ourselves, since OpenAI doesn't provide them.
        For the completion tokens, tiktoken is perfectly accurate.
        For the input tokens, unfortunately not.
        Here, we use a JSON dump of the conversation history.
        This is slightly inaccurate, but only by a few tokens.
        Also, this always leads to an overestimation, so the costs will still be under control.
        """
        self.log.info("CALL - streaming chat completion")
        response_text = ""
        encoding = tiktoken.get_encoding("cl100k_base")

        stream = self.client.chat.completions.create(
            model=self.model_id,
            messages=self.conversation_history,
            temperature=self.temperature,
            stream=True,
        )

        for chunk in stream:
            if len(chunk.choices) == 0:
                continue

            delta = chunk.choices[0].delta
            if delta is None or delta.content is None:
                continue

            chunk_text = delta.content
            response_text += chunk_text
            callback(chunk_text)

        prompt_tokens = encoding.encode(text=json.dumps(self.conversation_history))
        completion_tokens = encoding.encode(text=response_text)

        return response_text, len(prompt_tokens), len(completion_tokens)


class GenericGatewayLLM(BaseGatewayLLM):
    model_token: str

    def _call_service(self, callback: Callback | None) -> ServiceResponse:
        headers = {
            "Authorization": f"Bearer {self.model_token}",
            "Content-Type": "application/json",
        }

        body = {"messages": self.conversation_history}

        url = f"{self.model_url}chat/completions"

        if callback is None:
            return self._regular_chat_completion(url=url, headers=headers, body=body)

        return self._streaming_chat_completion(
            callback=callback, url=url, headers=headers, body=body
        )

    def _regular_chat_completion(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> ServiceResponse:
        response = requests.post(url=url, headers=headers, json=body)
        response_text = self.parse_chat_response(response)

        response_json = response.json()
        if "usage" not in response_json:
            self.log.error("no token usage in LLM response")
            prompt_tokens = 0
            completion_tokens = 0
        else:
            prompt_tokens = response_json["usage"]["prompt_tokens"]
            completion_tokens = response_json["usage"]["completion_tokens"]

        return response_text, prompt_tokens, completion_tokens

    def _streaming_chat_completion(
        self, callback: Callable, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> ServiceResponse:
        # In this case, we need to compute the number of tokens ourselves, since OpenAI doesn't provide them.
        # For the completion tokens, tiktoken is perfectly accurate.
        # For the input tokens, unfortunately not. Here, we use a JSON dump of the conversation history.
        # This is slightly inaccurate, but only by a few tokens.
        # Also, this always leads to an overestimation, so the costs will still be under control.

        response_text = ""
        encoding = tiktoken.get_encoding("cl100k_base")

        body = {**body, "stream": True}

        with EventSource(url=url, method="POST", headers=headers, json=body) as event_source:
            for event in event_source:
                if event.data == "[DONE]":
                    event_source.close()
                    break
                chunk = json.loads(event.data)

                if "choices" not in chunk or len(chunk["choices"]) == 0:
                    continue

                delta = chunk["choices"][0]["delta"]
                if delta is None or "content" not in delta or delta["content"] is None:
                    continue

                chunk_text = delta["content"]
                response_text += chunk_text
                callback(chunk_text)

        prompt_tokens = len(encoding.encode(text=json.dumps(self.conversation_history)))
        completion_tokens = len(encoding.encode(text=response_text))

        return response_text, prompt_tokens, completion_tokens
