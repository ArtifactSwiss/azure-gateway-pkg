# Azure Model API Gateway Package

Package to wrap calling the [Azure Model API Gateway](https://github.com/ArtifactSwiss/azure-gateway).

## Installation

1. Install directly using pip

```bash
pip install git+https://github.com/ArtifactSwiss/azure-gateway-pkg.git
```

2. Use python package as `azure_gateway`

## Usage

- There are always two parts to one request type:
  1.  The `NAME_request` function
  2.  The `parse_NAME_response` function
- Currently supported `NAME`s are:
  - `openai_chat`
  - `openai_embedding`
  - `azure_translation`
- There is also a `LLMGateway` wrapper for langchain calls
- Import them as required and adjust the authentication tokens on a project level as described in the gateway docs

## Example

```python
import os
import azure_gateway

params = azure_gateway.OpenAIBody(
    model_id=os.environ["GATEWAY_MODEL"],
    messages=[{"role": "system", "content": "Greet everyone like they're pizza royalty"}],
)
res = azure_gateway.openai_chat_request(
    project_id=os.environ["GATEWAY_PROJECT"],
    params=params,
    token=os.environ["GATEWAY_TOKEN"]
)
parsed = azure_gateway.parse_openai_chat_response(res)
```
