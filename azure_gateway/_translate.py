import json

import requests

from ._constants import ENDPOINT_URL


def azure_translation_request(  # noqa: PLR0913
    project_id: str,
    text: str,
    target_languages: list[str],
    original_language: str,
    token: str,
    user_id: str = None,
) -> str:
    """
    Send a request to the specified Azure translation API.

    :param project_id: Project ID to log request to.
    :param text: Text to be translated.
    :param target_languages: Languages to translate text into.
    :param original_language: Language of origin (skipped if confident that isn't said language).
    :param token: Authorization token for the API.
    :param user_id: Fine grained logging by adding user to project.
    :return: The response from the API.
    """
    url_with_project_id = f"{ENDPOINT_URL}/translation?project_id={project_id}"
    body = dict(
        user_id=user_id,
        text=text,
        target_languages=target_languages,
        original_language=original_language,
    )
    body = {k: v for k, v in body.items() if v is not None}
    payload = json.dumps(body)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    response = requests.request("POST", url_with_project_id, headers=headers, data=payload)
    response.raise_for_status()
    return response


def parse_azure_translation_response(response: requests.Response) -> list[str]:
    """Parse a translation response to only the actual translations."""
    if not isinstance(response, requests.Response):
        raise ValueError("Invalid response object")

    try:
        data = response.json()
    except json.JSONDecodeError:
        raise ValueError("Response content is not valid JSON")

    if not isinstance(data, list) or len(data) == 0 or "translations" not in data[0]:
        raise ValueError("Unexpected JSON structure in response")

    try:
        translations = [t["text"] for t in data[0]["translations"] if "text" in t]
    except (KeyError, TypeError, IndexError):
        raise ValueError("Error extracting translations from response")

    return translations
