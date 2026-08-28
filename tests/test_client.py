import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pydantic import BaseModel, Field

from gento import (
    LLMClient,
    GenerateResponse,
    UnsupportedModelError,
)


class UserProfile(BaseModel):
    name: str
    age: int
    interests: list[str] = Field(default_factory=list)


@pytest.mark.asyncio
async def test_client_default_model_initialization():
    client = LLMClient()
    assert client.default_model == "google/gemini-3.5-flash"

    custom_client = LLMClient("openai/gpt-4o")
    assert custom_client.default_model == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_client_generate_calls_adapter():
    client = LLMClient("google/gemini-3.5-flash")
    mock_response = GenerateResponse(content="Hello world", raw_response={})

    mock_adapter = MagicMock()
    mock_adapter.generate = AsyncMock(return_value=mock_response)

    with patch.object(client, "_get_adapter", return_value=mock_adapter):
        resp = await client.generate("Say hello")
        assert resp.content == "Hello world"
        mock_adapter.generate.assert_called_once()
        req = mock_adapter.generate.call_args[0][0]
        assert req.model == "google/gemini-3.5-flash"
        assert req.prompt == "Say hello"


@pytest.mark.asyncio
async def test_client_generate_structured_success():
    client = LLMClient("openai/gpt-4o")
    expected_profile = UserProfile(name="Alice", age=30, interests=["coding"])
    mock_response = GenerateResponse(
        content='{"name":"Alice","age":30,"interests":["coding"]}',
        parsed=expected_profile,
    )

    mock_adapter = MagicMock()
    mock_adapter.generate = AsyncMock(return_value=mock_response)

    with patch.object(client, "_get_adapter", return_value=mock_adapter):
        profile = await client.generate_structured(
            "Get profile",
            response_schema=UserProfile,
        )
        assert profile.name == "Alice"
        assert profile.age == 30
        assert profile.interests == ["coding"]


@pytest.mark.asyncio
async def test_client_generate_dict_success():
    client = LLMClient("volcengine/doubao-1.5-pro-32k")
    expected_profile = UserProfile(name="Bob", age=25)
    mock_response = GenerateResponse(
        content='{"name":"Bob","age":25}', parsed=expected_profile
    )

    mock_adapter = MagicMock()
    mock_adapter.generate = AsyncMock(return_value=mock_response)

    with patch.object(client, "_get_adapter", return_value=mock_adapter):
        data = await client.generate_dict(
            "Get profile",
            response_schema=UserProfile,
        )
        assert isinstance(data, dict)
        assert data["name"] == "Bob"
        assert data["age"] == 25


@pytest.mark.asyncio
async def test_client_unsupported_model():
    client = LLMClient()
    with pytest.raises(UnsupportedModelError):
        await client.generate("Hi", model="unknown-provider/foo-bar")


@pytest.mark.asyncio
async def test_client_get_adapter_resolves_env_keys():
    import os
    from gento.adapters.gemini import GeminiAdapter
    from gento.adapters.openai import OpenAIAdapter
    from gento.adapters.ark import ArkAdapter

    client = LLMClient()
    with patch.dict(os.environ, {"GEMINI_API_KEY": "env-gemini-key"}):
        adapter = client._get_adapter("google/gemini-3.5-flash")
        assert isinstance(adapter, GeminiAdapter)
        assert adapter.api_key == "env-gemini-key"

    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-openai-key"}):
        adapter_openai = client._get_adapter("openai/gpt-4o")
        assert isinstance(adapter_openai, OpenAIAdapter)
        assert adapter_openai.api_key == "env-openai-key"

    with patch.dict(os.environ, {"VOLC_API_KEY": "env-volc-key"}):
        adapter_ark = client._get_adapter("volcengine/doubao-1.5-pro-32k")
        assert isinstance(adapter_ark, ArkAdapter)
        assert adapter_ark.api_key == "env-volc-key"
        assert adapter_ark.base_url == "https://ark.cn-beijing.volces.com/api/v3"
