# Azure Model API Gateway Package

Package to wrap calling the [Azure Model API Gateway](https://github.com/ArtifactSwiss/azure-gateway).

## Installation

1. Install directly using pip

```bash
pip install git+https://github.com/ArtifactSwiss/azure-gateway-pkg.git
```

2. Use python package as `azure_gateway`

If you want a specific version, append `@VERSION` to the installation above. Versions can be found in the [releases tab](https://github.com/ArtifactSwiss/azure-gateway-pkg/releases).

## Usage

### Option 1: direct calls

- There are always two parts to one request type:
  1.  The `NAME_request` function
  2.  The `parse_NAME_response` function
- Currently supported `NAME`s are:
  - `openai_chat`
  - `openai_embedding`
  - `azure_translation`
- Import them as required and adjust the authentication tokens on a project level as described in the gateway docs

```python
import os
import azure_gateway

params = azure_gateway.OpenAIBody(
    model_id=os.environ["GATEWAY_MODEL"],
    messages=[
        {"role": "system", "content": "Greet everyone like they're pizza royalty"}
    ],
)
res = azure_gateway.openai_chat_request(
    project_id=os.environ["GATEWAY_PROJECT"],
    params=params,
    token=os.environ["GATEWAY_TOKEN"]
)
parsed = azure_gateway.parse_openai_chat_response(res)
```

### Option 2: high traffic calls

- A high-traffic endpoint wrapper that performs all calls directly to Azure and only logs the usage to the gateway
- There are currently two types:
  1. `GenericGatewayLLM`
  2. `OpenAIGatewayLLM`
- Import them as required and pass a Azure client / custom call

```python
import os
import azure_gateway

client = AzureOpenAI(
    api_key=os.environ["AZURE_API_KEY"],
    api_version="2024-06-01",
    azure_endpoint=os.environ["AZURE_ENDPOINT"]
)
llm = azure_gateway.OpenAIGatewayLLM(
    project_id=os.environ["GATEWAY_PROJECT_ID"],
    model_id=os.environ["GATEWAY_MODEL_ID"],
    auth_token=os.environ["GATEWAY_AUTH_TOKEN"],
    client=client
)

response = llm.call("Say hello to the lovely people of the interwebs")
```
