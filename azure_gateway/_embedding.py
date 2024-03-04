import json

import requests

from ._constants import ENDPOINT_URL


def openai_embedding_request(
    project_id: str, text: str, token: str, user_id: str = None
) -> list[float]:
    """
    Send a request to the specified OpenAI embedding API.

    :param project_id: Project ID to log request to.
    :param text: Text to embed.
    :param token: Authorization token for the API.
    :param user_id: Fine grained logging by adding user to project.
    """
    url_with_project_id = f"{ENDPOINT_URL}/embedding?project_id={project_id}"
    body = dict(text=text, user_id=user_id)
    body = {k: v for k, v in body.items() if v is not None}
    payload = json.dumps(body)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    response = requests.request("POST", url_with_project_id, headers=headers, data=payload)
    response.raise_for_status()
    return response


def parse_openai_embedding_response(response: requests.Response) -> list[float]:
    """Parse an embedding response to only the actual embedding."""
    if not isinstance(response, requests.Response):
        raise ValueError("Invalid response object")

    try:
        data = response.json()
    except json.JSONDecodeError:
        raise ValueError("Response content is not valid JSON")

    if (
        not isinstance(data, dict)
        or "data" not in data
        or not data["data"]
        or "embedding" not in data["data"][0]
    ):
        raise ValueError("Unexpected JSON structure in response")

    try:
        embedding = data["data"][0]["embedding"]
    except (KeyError, TypeError, IndexError):
        raise ValueError("Error extracting embedding from response")

    return embedding
