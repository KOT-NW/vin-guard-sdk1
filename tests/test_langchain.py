"""Runnable integration tests using actual LangChain and mocked HTTP.

The LangChain extra is required to collect this module. A plain import (not
``pytest.importorskip``) ensures a missing dependency fails the adapter CI job
instead of silently skipping into a green run. Core-only jobs exclude this file.
"""

import asyncio
import json

import httpx
import pytest

from ai_gateway import AsyncGatewayClient, GatewayClient, GatewayError
from ai_gateway.langchain import ChatGateway
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel


def response(*, content="hello", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return httpx.Response(200, json={
        "id": "chatcmpl_test", "model": "openai/stub-demo",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        "_gateway": {"requestId": "req_test", "tier": "easy", "policyVersion": 7,
                     "costUsd": 0.002, "fallback": False},
    }, headers={"x-gateway-request-id": "req_test", "x-gateway-provider": "openai"})


def test_invoke_maps_supported_payload_and_preserves_gateway_metadata():
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "policy/default-auto"
        assert body["max_tokens"] == 17
        assert "max_completion_tokens" not in body
        assert body["stream"] is False
        assert body["messages"] == [{"role": "system", "content": "be brief"},
                                    {"role": "user", "content": "hi"}]
        assert request.headers["x-gateway-tags"] == "team=sdk"
        assert json.loads(request.headers["x-gateway-metadata"]) == {"ticket": "t_1"}
        assert request.headers["x-gateway-no-store"] == "true"
        return response()

    with GatewayClient(api_key="gw_live_secret", transport=httpx.MockTransport(handler)) as client:
        with ChatGateway(api_key="gw_live_secret", client=client, model="policy/default-auto",
                         max_tokens=17, gateway_tags={"team": "sdk"}, no_store=True) as model:
            result = model.invoke([SystemMessage(content="be brief"), HumanMessage(content="hi")],
                                  config={"tags": ["trace-only"], "metadata": {"trace": "only"}},
                                  gateway_metadata={"ticket": "t_1"})
        assert not client._http.is_closed  # injected clients remain caller-owned
    assert isinstance(result, AIMessage)
    assert result.content == "hello"
    assert result.response_metadata["_gateway"]["policyVersion"] == 7
    assert result.response_metadata["_gateway"]["costUsd"] == 0.002
    assert result.response_metadata["gateway_headers"]["x-gateway-provider"] == "openai"
    assert result.response_metadata["model_name"] == "openai/stub-demo"
    assert result.response_metadata["finish_reason"] == "stop"
    assert result.usage_metadata == {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}


def test_prompt_pipeline_and_batch_use_runnable_contract():
    seen = []

    def handler(request):
        prompt = json.loads(request.content)["messages"][-1]["content"]
        seen.append(prompt)
        return response(content=prompt.upper())

    with GatewayClient(api_key="x", transport=httpx.MockTransport(handler)) as client:
        model = ChatGateway(api_key="x", client=client)
        chain = ChatPromptTemplate.from_template("Explain {topic}") | model | StrOutputParser()
        assert chain.invoke({"topic": "routing"}) == "EXPLAIN ROUTING"
        assert chain.batch([{"topic": "cost"}, {"topic": "limits"}],
                           config={"max_concurrency": 2}) == ["EXPLAIN COST", "EXPLAIN LIMITS"]
    assert len(seen) == 3


def test_native_async_ainvoke_abatch_and_buffered_astream():
    seen = []

    async def handler(request):
        seen.append(json.loads(request.content))
        await asyncio.sleep(0)
        return response()

    async def run():
        async with AsyncGatewayClient(api_key="x", transport=httpx.MockTransport(handler)) as client:
            async with ChatGateway(api_key="x", async_client=client) as model:
                assert (await model.ainvoke("hi")).content == "hello"
                assert len(await model.abatch(["a", "b"], config={"max_concurrency": 2})) == 2
                chunks = [chunk async for chunk in model.astream("hi")]
                assert len(chunks) == 1 and chunks[0].content == "hello"
                assert model._owned_client is None
            assert not client._http.is_closed

    asyncio.run(run())
    assert len(seen) == 4
    assert all(body["stream"] is False for body in seen)


def test_sync_stream_is_buffered_and_callbacks_observe_usage():
    class Recorder(BaseCallbackHandler):
        results = []

        def on_llm_end(self, result, **kwargs):
            self.results.append(result)

    recorder = Recorder()
    with GatewayClient(api_key="x", transport=httpx.MockTransport(lambda request: response())) as client:
        model = ChatGateway(api_key="x", client=client)
        chunks = list(model.stream("hi", config={"callbacks": [recorder]}))
    assert len(chunks) == 1
    assert chunks[0].response_metadata["_gateway"]["requestId"] == "req_test"
    assert recorder.results[0].generations[0][0].message.usage_metadata["total_tokens"] == 8


@pytest.mark.parametrize("options", [
    {"stop": ["END"]}, {"tools": []}, {"response_format": {"type": "json_object"}},
    {"top_p": 0.8}, {"seed": 7}, {"stream": True},
])
def test_unsupported_features_fail_before_network(options):
    def handler(request):
        pytest.fail("Unsupported feature reached the backend")

    with GatewayClient(api_key="x", transport=httpx.MockTransport(handler)) as client:
        model = ChatGateway(api_key="x", client=client)
        with pytest.raises(ValueError):
            model.invoke("hi", **options)


def test_tool_binding_requires_opt_in():
    model = ChatGateway(api_key="x")
    with pytest.raises(NotImplementedError, match="current gateway"):
        model.bind_tools([])
    with pytest.raises(ValueError, match="current gateway"):
        model.invoke([ToolMessage(content="result", tool_call_id="call_1")])
    assert model._owned_client is None


def test_future_backend_tool_schema_and_tool_call_conversion():
    class Lookup(BaseModel):
        """Look up a city."""
        city: str

    def handler(request):
        body = json.loads(request.content)
        assert body["tools"][0]["function"]["name"] == "Lookup"
        assert body["tool_choice"] == {"type": "function", "function": {"name": "Lookup"}}
        return response(content=None, tool_calls=[{
            "id": "call_1", "type": "function", "function": {"name": "Lookup", "arguments": '{"city":"Hanoi"}'},
        }])

    with GatewayClient(api_key="x", transport=httpx.MockTransport(handler)) as client:
        model = ChatGateway(api_key="x", client=client, allow_advanced_options=True)
        result = model.bind_tools([Lookup], tool_choice="Lookup").invoke("Where is Hanoi?")
    assert result.content == ""
    assert result.tool_calls[0]["name"] == "Lookup"
    assert result.tool_calls[0]["args"] == {"city": "Hanoi"}
    assert result.response_metadata["_gateway"]["tier"] == "easy"


def test_future_backend_tool_history_serialization_and_structured_output():
    class Lookup(BaseModel):
        """Look up a city."""
        city: str

    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return response(content=None, tool_calls=[{
            "id": "call_1", "type": "function", "function": {"name": "Lookup", "arguments": '{"city":"Hanoi"}'},
        }])

    with GatewayClient(api_key="x", transport=httpx.MockTransport(handler)) as client:
        model = ChatGateway(api_key="x", client=client, allow_advanced_options=True)
        result = model.with_structured_output(Lookup).invoke("Look up Hanoi")
        assert result.city == "Hanoi"
        model.invoke([HumanMessage(content="Where?"),
                      AIMessage(content="", tool_calls=[{"id": "call_1", "name": "Lookup", "args": {"city": "Hanoi"}}]),
                      ToolMessage(content="Vietnam", tool_call_id="call_1")])
    assert bodies[1]["messages"][1]["tool_calls"][0]["function"]["name"] == "Lookup"
    assert bodies[1]["messages"][2] == {"role": "tool", "content": "Vietnam", "tool_call_id": "call_1"}


def test_errors_propagate_and_secrets_stay_out_of_serialization():
    def handler(request):
        return httpx.Response(429, json={"error": {"code": "budget_exceeded", "message": "budget"}},
                              headers={"x-gateway-request-id": "req_error"})

    secret = "gw_live_key_secret"
    with GatewayClient(api_key=secret, transport=httpx.MockTransport(handler)) as client:
        model = ChatGateway(api_key=secret, client=client)
        assert secret not in repr(model)
        assert secret not in json.dumps(model.model_dump(), default=str)
        # ``asdict`` is a LangChain convenience absent on the 1.0.x floor.
        asdict = getattr(model, "asdict", None)
        if callable(asdict):
            assert secret not in json.dumps(asdict(), default=str)
        else:
            assert secret not in json.dumps(model.dict(), default=str)
        assert secret not in json.dumps(model.to_json(), default=str)
        assert model.cache is False
        with pytest.raises(GatewayError) as exc:
            model.invoke("hi")
    assert exc.value.code == "budget_exceeded"
    assert exc.value.request_id == "req_error"


def test_owned_clients_close_and_empty_response_has_actionable_error():
    model = ChatGateway(api_key="x")
    sync = model._sync()
    model.close()
    assert sync._http.is_closed

    async def run():
        model = ChatGateway(api_key="x")
        client = model._async()
        await model.aclose()
        assert client._http.is_closed

    asyncio.run(run())
    with GatewayClient(api_key="x", transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))) as client:
        with pytest.raises(GatewayError, match="no choices"):
            ChatGateway(api_key="x", client=client).invoke("hi")
