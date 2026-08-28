import os
import pytest
from unittest.mock import patch, MagicMock

from gento.adapters.gemini import GeminiAdapter
from gento.adapters.openai import OpenAIAdapter
from gento.adapters.ark import ArkAdapter
from gento.models import GenerateRequest


def test_gemini_adapter_init_missing_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiAdapter()


def test_gemini_adapter_init_with_key():
    adapter = GeminiAdapter(api_key="test-gemini-key")
    assert adapter.client is not None
    assert adapter.normalize_model_name("google/gemini-3.5-flash") == "gemini-3.5-flash"


def test_openai_adapter_init_missing_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIAdapter()


def test_openai_adapter_init_with_key():
    adapter = OpenAIAdapter(api_key="test-openai-key")
    assert adapter.client is not None
    assert adapter.normalize_model_name("openai/gpt-4o") == "gpt-4o"


def test_ark_adapter_init_missing_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="VOLC_API_KEY"):
            ArkAdapter()


def test_ark_adapter_init_with_key():
    adapter = ArkAdapter(api_key="test-ark-key")
    assert adapter.client is not None
    assert adapter.normalize_model_name("volcengine/doubao-1.5-pro-32k") == "doubao-1.5-pro-32k"
    assert adapter.normalize_model_name("ark/ep-20240101") == "ep-20240101"
