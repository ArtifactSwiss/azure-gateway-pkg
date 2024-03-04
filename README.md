# Azure Model API Gateway Package

Package to wrap calling the [Azure Model API Gateway](https://github.com/ArtifactSwiss/azure-gateway).

## Installation

1. Create a personal access token with repo scope (see [docs](https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line#creating-a-token))
2. Install directly using pip

```bash
pip install git+https://{token}@github.com/ArtifactSwiss/azure-gateway-pkg.git
```

3. Use python package as `azure_gateway`

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
