from langchain_core.language_models.llms import LLM

from ._chat import openai_chat_request, parse_openai_chat_response


class GatewayLLM(LLM):
    """Custom langchain LLM wrapper to use gateway in chains."""

    project_id: str
    token: str
    model_id: str
    conversation_history: list

    @property
    def _llm_type(self) -> str:
        return "gateway"

    @property
    def _identifying_params(self):
        """Get the identifying parameters."""
        return {}

    def _call(self, prompt, **kwargs):
        self.conversation_history.append({"role": "user", "content": prompt})
        response = self.send_prompt_to_gateway(self.conversation_history)
        params = {"model_id": self.model_id, "messages": prompt, **kwargs}
        response = openai_chat_request(self.project_id, params, self.token)
        self.conversation_history.append({"role": "assistant", "content": response})
        return parse_openai_chat_response(response)
