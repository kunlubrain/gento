import pytest

from gento.adapters.ark import ArkAdapter
from gento.adapters.gemini import GeminiAdapter
from gento.adapters.openai import OpenAIAdapter
from gento.exceptions import UnsupportedModelError
from gento.registry import create_adapter, resolve_model_definition


def test_resolve_known_models():
    def_gemini = resolve_model_definition("google/gemini-3.5-flash")
    assert def_gemini.adapter_class == GeminiAdapter

    def_openai = resolve_model_definition("openai/gpt-4o")
    assert def_openai.adapter_class == OpenAIAdapter

    def_ark = resolve_model_definition("volcengine/doubao-1.5-pro-32k")
    assert def_ark.adapter_class == ArkAdapter


def test_resolve_dynamic_models():
    # Modern / custom names
    assert resolve_model_definition("gemini-3-flash").adapter_class == GeminiAdapter
    assert resolve_model_definition("gpt-5").adapter_class == OpenAIAdapter
    assert resolve_model_definition("o3-mini").adapter_class == OpenAIAdapter
    assert resolve_model_definition("ep-20240101-abcde").adapter_class == ArkAdapter
    assert resolve_model_definition("doubao-vision-pro").adapter_class == ArkAdapter


def test_resolve_unknown_model_raises():
    with pytest.raises(UnsupportedModelError):
        resolve_model_definition("custom-model-without-prefix")


def test_create_adapter():
    adapter = create_adapter("google/gemini-3.5-flash", api_key="test_key")
    assert isinstance(adapter, GeminiAdapter)
