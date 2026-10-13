"""Optional native LangChain adapter retaining gateway routing metadata.

Install `vin-guard-sdk[langchain]` and import ChatGateway from this module.
The current server is non-streaming: BaseChatModel supplies a buffered
stream/astream fallback, while invoke/ainvoke use sync/native async HTTP.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from threading import Lock
from typing import Any

try:
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
    from langchain_core.caches import BaseCache
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, convert_to_openai_messages
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from pydantic import ConfigDict, Field, PrivateAttr, SecretStr, field_validator
except ImportError as exc:
    raise ImportError("Install the optional integration: pip install 'vin-guard-sdk[langchain]'") from exc

from .async_client import AsyncGatewayClient
from .client import ChatCompletion, GatewayClient
from .errors import GatewayError

_ADVANCED = {"stop", "tools", "tool_choice", "response_format", "top_p", "user", "seed"}
__all__ = ["ChatGateway"]


def _to_result(completion: ChatCompletion) -> ChatResult:
    if not completion.choices:
        raise GatewayError("Chat response contains no choices", code="invalid_response",
                           status=200, request_id=completion.gateway.request_id)
    gateway = {
        "requestId": completion.gateway.request_id, "tier": completion.gateway.tier,
        "policy": completion.gateway.policy, "confidence": completion.gateway.confidence,
        "costUsd": completion.gateway.cost_usd,
    }
    gateway.update(completion.raw.get("_gateway") or {})
    token_usage = asdict(completion.usage)
    generations = []
    for choice in completion.choices:
        wire_message = choice.get("message") or {}
        extra = {k: v for k, v in wire_message.items() if k not in ("role", "content")}
        metadata = {
            "model_name": completion.model, "finish_reason": choice.get("finish_reason"),
            "token_usage": token_usage, "_gateway": gateway,
            "gateway_headers": completion.headers,
        }
        message = AIMessage(
            content=wire_message.get("content") or "", additional_kwargs=extra,
            response_metadata=metadata, id=completion.id or None,
            usage_metadata={"input_tokens": completion.usage.prompt_tokens,
                            "output_tokens": completion.usage.completion_tokens,
                            "total_tokens": completion.usage.total_tokens},
        )
        generations.append(ChatGeneration(message=message,
                                           generation_info={"finish_reason": choice.get("finish_reason")}))
    return ChatResult(generations=generations,
                      llm_output={"model_name": completion.model, "token_usage": token_usage,
                                  "_gateway": gateway, "gateway_headers": completion.headers})


class ChatGateway(BaseChatModel):
    """A Runnable chat model for the gateway's chat-completions endpoint.

    gateway_tags/gateway_metadata are wire headers. LangChain's tags/metadata
    and RunnableConfig are tracing configuration and are not sent as headers.
    Caller-supplied SDK clients remain caller-owned.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: SecretStr = Field(exclude=True, repr=False)
    model: str = "auto"
    base_url: str = "https://gateway.example.com/v1"
    timeout: float = 30.0
    temperature: float | None = None
    max_tokens: int | None = None
    gateway_tags: dict[str, str] | str | None = None
    gateway_metadata: dict[str, Any] | None = None
    route_policy: str | None = None
    no_store: bool = False
    allow_advanced_options: bool = False
    cache: BaseCache | bool | None = False
    client: GatewayClient | None = Field(default=None, exclude=True, repr=False)
    async_client: AsyncGatewayClient | None = Field(default=None, exclude=True, repr=False)

    _owned_client: GatewayClient | None = PrivateAttr(default=None)
    _owned_async_client: AsyncGatewayClient | None = PrivateAttr(default=None)
    _client_lock: Lock = PrivateAttr(default_factory=Lock)

    @field_validator("api_key")
    @classmethod
    def _validate_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("api_key is required")
        return value

    @property
    def _llm_type(self) -> str:
        return "ai-gateway"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url,
                "temperature": self.temperature, "max_tokens": self.max_tokens,
                "route_policy": self.route_policy, "no_store": self.no_store,
                "gateway_tags": self.gateway_tags, "gateway_metadata": self.gateway_metadata,
                "allow_advanced_options": self.allow_advanced_options}

    def _sync(self) -> GatewayClient:
        if self.client is not None:
            return self.client
        with self._client_lock:
            if self._owned_client is None:
                self._owned_client = GatewayClient(api_key=self.api_key.get_secret_value(),
                                                    base_url=self.base_url, timeout=self.timeout)
            return self._owned_client

    def _async(self) -> AsyncGatewayClient:
        if self.async_client is not None:
            return self.async_client
        with self._client_lock:
            if self._owned_async_client is None:
                self._owned_async_client = AsyncGatewayClient(api_key=self.api_key.get_secret_value(),
                                                              base_url=self.base_url, timeout=self.timeout)
            return self._owned_async_client

    def _request(self, messages: list[BaseMessage], stop: list[str] | None,
                 kwargs: dict[str, Any]) -> dict[str, Any]:
        options = {"temperature": self.temperature, "max_tokens": self.max_tokens,
                   "tags": self.gateway_tags, "metadata": self.gateway_metadata,
                   "route_policy": self.route_policy, "no_store": self.no_store}
        overrides = dict(kwargs)
        # BaseChatModel.with_structured_output supplies this tracing hint.
        overrides.pop("ls_structured_output_format", None)
        for source, target in (("gateway_tags", "tags"), ("gateway_metadata", "metadata")):
            if source in overrides:
                overrides[target] = overrides.pop(source)
        options.update(overrides)
        if stop:
            options["stop"] = stop
        advanced = [key for key in _ADVANCED if options.get(key) is not None]
        wire_messages = convert_to_openai_messages(messages)
        has_tools = any(m.get("role") in ("tool", "function") or m.get("tool_calls")
                        or m.get("function_call") for m in wire_messages)
        if not self.allow_advanced_options and (advanced or has_tools):
            raise ValueError("The current gateway backend does not implement tools/stop/structured output "
                             "or other advanced chat options. Set allow_advanced_options=True only "
                             "for a backend that supports them.")
        if options.get("stream"):
            raise ValueError("ChatGateway uses buffered responses; server SSE is not implemented")
        return {"model": self.model, "messages": wire_messages, **options}

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        request = self._request(messages, stop, kwargs)
        return _to_result(self._sync().chat_completions(**request))

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None,
                         run_manager: AsyncCallbackManagerForLLMRun | None = None,
                         **kwargs: Any) -> ChatResult:
        request = self._request(messages, stop, kwargs)
        return _to_result(await self._async().chat_completions(**request))

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: str | dict | None = None,
                   strict: bool | None = None, **kwargs: Any):
        """Opt-in schema binding for a future backend with tool support."""
        if not self.allow_advanced_options:
            raise NotImplementedError("Tool calling is not implemented by the current gateway backend")
        schemas = [convert_to_openai_tool(tool, strict=strict) for tool in tools]
        if tool_choice == "any":
            tool_choice = "required"
        elif isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
            tool_choice = {"type": "function", "function": {"name": tool_choice}}
        return self.bind(tools=schemas, tool_choice=tool_choice, **kwargs)

    def close(self) -> None:
        """Close the owned synchronous HTTP client.

        Sync context managers (``with ChatGateway(...)``) do not clean up the
        owned async client because ``AsyncClient.aclose()`` must be awaited.
        Use ``await aclose()`` (or ``async with``) when both ``invoke`` and
        ``ainvoke`` were used.
        """
        if self._owned_client is not None:
            self._owned_client.close()

    async def aclose(self) -> None:
        """Close both owned HTTP clients, leaving injected clients untouched."""
        self.close()
        if self._owned_async_client is not None:
            await self._owned_async_client.aclose()

    def __enter__(self) -> ChatGateway:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    async def __aenter__(self) -> ChatGateway:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
