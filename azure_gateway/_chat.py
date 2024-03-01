import json

import requests

from ._constants import ENDPOINT_URL


def openai_chat_request(project_id: str, params: dict, token: str):
    """
    Send a request to the specified OpenAI chat API.

    :param project_id: Project ID to log request to.
    :param params: The body parameters for the API (see docs).
    :param token: Authorization token for the API.
    :return: The response from the API.
    """
    url_with_project_id = f"{ENDPOINT_URL}/openai-chat?project_id={project_id}"

    payload = json.dumps(params)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    response = requests.request(
        "POST", url_with_project_id, headers=headers, data=payload
    )
    response.raise_for_status()
    return response
