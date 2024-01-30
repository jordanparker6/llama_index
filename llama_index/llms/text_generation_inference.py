from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import llm_chat_callback, llm_completion_callback
from llama_index.llms.custom import CustomLLM
from llama_index.prompts import PromptTemplate
from llama_index.types import BaseOutputParser, PydanticProgramMode

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_HUGGINGFACE_MODEL = "StabilityAI/stablelm-tuned-alpha-3b"


class TextGenerationInference(CustomLLM):
    base_url: str = Field(
        default=DEFAULT_URL,
        description="Base URL for the text generation service.",
    )
    auth_token: Optional[Union[str, Callable[[str], str]]] = Field(
        default=None,
        description="Authentication token or authentication token generating function for authenticated requests",
    )
    model_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The model name to use from HuggingFace. "
            "Unused if `model` is passed in directly."
        ),
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of tokens available for input.",
        gt=0,
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for generation.",
        gt=0,
    )
    system_prompt: str = Field(
        default="",
        description=(
            "The system prompt, containing any extra instructions or context. "
            "The model card on HuggingFace should specify if this is needed."
        ),
    )
    query_wrapper_prompt: PromptTemplate = Field(
        default=PromptTemplate("{query_str}"),
        description=(
            "The query wrapper prompt, containing the query placeholder. "
            "The model card on HuggingFace should specify if this is needed. "
            "Should contain a `{query_str}` placeholder."
        ),
    )
    tokenizer_name: str = Field(
        default=DEFAULT_HUGGINGFACE_MODEL,
        description=(
            "The name of the tokenizer to use from HuggingFace. "
            "Unused if `tokenizer` is passed in directly."
        ),
    )
    stopping_ids: List[int] = Field(
        default_factory=list,
        description=(
            "The stopping ids to use. "
            "Generation stops when these token IDs are predicted."
        ),
    )
    tokenizer_outputs_to_remove: list = Field(
        default_factory=list,
        description=(
            "The outputs to remove from the tokenizer. "
            "Sometimes huggingface tokenizers return extra inputs that cause errors."
        ),
    )
    generate_kwargs: dict = Field(
        default_factory=dict,
        description="The kwargs to pass to the model during generation.",
    )
    is_chat_model: bool = Field(
        default=False,
        description=(
            LLMMetadata.__fields__["is_chat_model"].field_info.description
            + " Be sure to verify that you either pass an appropriate tokenizer "
            "that can convert prompts to properly formatted chat messages or a "
            "`messages_to_prompt` that does so."
        ),
    )

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()
    _stopping_criteria: Any = PrivateAttr()

    _client: Any = PrivateAttr()

    timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for the request.",
    )

    def __init__(
        self,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        query_wrapper_prompt: Union[str, PromptTemplate] = "{query_str}",
        tokenizer_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model_name: str = DEFAULT_HUGGINGFACE_MODEL,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        stopping_ids: Optional[List[int]] = None,
        generate_kwargs: Optional[dict] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        is_chat_model: Optional[bool] = False,
        callback_manager: Optional[CallbackManager] = None,
        system_prompt: str = "",
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
        auth_token: Optional[Union[str, Callable[[str], str]]] = None,
    ) -> None:
        try:
            import httpx  # noqa
        except ImportError:
            raise ImportError(
                "TextGenerationInterface requires httpx to be installed.\n"
                "Please install httpx with `pip install httpx`."
            )

        super().__init__(
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=tokenizer_name,
            model_name=model_name,
            stopping_ids=stopping_ids or [],
            generate_kwargs=generate_kwargs or {},
            is_chat_model=is_chat_model,
            callback_manager=callback_manager,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
            temperature=temperature,
            auth_token=auth_token,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TextGenerationInference"

    def _call_api(self, endpoint: str, prompt: str, **kwargs) -> List[List[float]]:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.auth_token is not None:
            if callable(self.auth_token):
                headers["Authorization"] = self.auth_token(self.base_url)
            else:
                headers["Authorization"] = self.auth_token
        input_dict = self._get_input_dict(prompt, **kwargs)

        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/{endpoint}",
                headers=headers,
                json=input_dict,
                timeout=self.timeout,
            )
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}")
        return response.json()

    async def _acall_api(
        self, endpoint: str, prompt: str, **kwargs
    ) -> List[List[float]]:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.auth_token is not None:
            if callable(self.auth_token):
                headers["Authorization"] = self.auth_token(self.base_url)
            else:
                headers["Authorization"] = self.auth_token
        input_dict = self._get_input_dict(prompt, **kwargs)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/{endpoint}",
                headers=headers,
                json=input_dict,
                timeout=self.timeout,
            )
        if response.status_code != 200:
            raise Exception(f"Request failed with status code {response.status_code}")
        return response.json()

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_name,
        )

    def _get_input_dict(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "inputs": prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                **kwargs,
            },
        }

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError("Chat is not implemented for this LLM.")

    @llm_chat_callback()
    async def achat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError("Chat is not implemented for this LLM.")

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError("Chat is not implemented for this LLM.")

    @llm_completion_callback()
    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponse:
        raise NotImplementedError("Chat is not implemented for this LLM.")

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        if not formatted:
            prompt = self.completion_to_prompt(prompt)
        data = self._call_api("generate", prompt, **kwargs)
        return CompletionResponse(text=data["generated_text"])

    @llm_completion_callback()
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        if not formatted:
            prompt = self.completion_to_prompt(prompt)
        data = await self._acall_api("generate", prompt, **kwargs)
        return CompletionResponse(text=data["generated_text"])

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        if not formatted:
            prompt = self.completion_to_prompt(prompt)
        data = self._call_api("generate", prompt, **kwargs)
        return CompletionResponse(text=data["generated_text"])

    @llm_completion_callback()
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        if not formatted:
            prompt = self.completion_to_prompt(prompt)
        data = await self._acall_api("generate", prompt, **kwargs)
        return CompletionResponse(text=data["generated_text"])
