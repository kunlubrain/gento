import os
from unittest.mock import patch

import pytest

from gento.adapters.ark import ArkAdapter
from gento.adapters.gemini import GeminiAdapter
from gento.adapters.openai import OpenAIAdapter
from gento.exceptions import UnsupportedModelError
from gento.registry import create_adapter, resolve_model_definition


def test_resolve_known_models():
    def_gemini = resolve_model_definition("google/gemini-3.5-flash")
    assert def_gemini.adapter_class == GeminiAdapter
    assert def_gemini.provider == "google"
    assert "GEMINI_API_KEY" in def_gemini.api_key_env_vars

    def_openai = resolve_model_definition("openai/gpt-4o")
    assert def_openai.adapter_class == OpenAIAdapter
    assert def_openai.provider == "openai"
    assert "OPENAI_API_KEY" in def_openai.api_key_env_vars

    def_ark = resolve_model_definition("volcengine/doubao-1.5-pro-32k")
    assert def_ark.adapter_class == ArkAdapter
    assert def_ark.provider == "volcengine"
    assert "VOLC_API_KEY" in def_ark.api_key_env_vars
    assert def_ark.default_base_url == "https://ark.cn-beijing.volces.com/api/v3"


def test_resolve_dynamic_models():
    gemini_def = resolve_model_definition("google/gemini-3.5-flash")
    assert gemini_def.adapter_class == GeminiAdapter
    assert "GEMINI_API_KEY" in gemini_def.api_key_env_vars

    gpt_def = resolve_model_definition("openai/gpt-5")
    assert gpt_def.adapter_class == OpenAIAdapter
    assert "OPENAI_API_KEY" in gpt_def.api_key_env_vars

    o4_def = resolve_model_definition("openai/gpt-4o-mini")
    assert o4_def.adapter_class == OpenAIAdapter
    assert "OPENAI_API_KEY" in o4_def.api_key_env_vars

    doubao_def = resolve_model_definition("volcengine/doubao-pro-32k")
    assert doubao_def.adapter_class == ArkAdapter
    assert "VOLC_API_KEY" in doubao_def.api_key_env_vars


def test_resolve_unknown_model_raises():
    with pytest.raises(UnsupportedModelError):
        resolve_model_definition("custom-model-without-prefix")


def test_model_definition_resolve_env_keys():
    gemini_def = resolve_model_definition("google/gemini-3.5-flash")
    with patch.dict(os.environ, {"GEMINI_API_KEY": "env-gemini-key"}, clear=True):
        assert gemini_def.resolve_api_key() == "env-gemini-key"

    ark_def = resolve_model_definition("volcengine/doubao-1.5-pro-32k")
    with patch.dict(os.environ, {"ARK_API_KEY": "env-ark-key"}, clear=True):
        assert ark_def.resolve_api_key() == "env-ark-key"
        assert ark_def.resolve_base_url() == "https://ark.cn-beijing.volces.com/api/v3"


def test_create_adapter_with_env_vars():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_env_key"}):
        adapter = create_adapter("google/gemini-3.5-flash")
        assert isinstance(adapter, GeminiAdapter)
        assert adapter.api_key == "test_env_key"

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_openai_key"}):
        adapter = create_adapter("openai/gpt-4o")
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter.api_key == "test_openai_key"

    with patch.dict(os.environ, {"VOLC_API_KEY": "test_volc_key"}):
        adapter = create_adapter("volcengine/doubao-1.5-pro-32k")
        assert isinstance(adapter, ArkAdapter)
        assert adapter.api_key == "test_volc_key"
        assert adapter.base_url == "https://ark.cn-beijing.volces.com/api/v3"


def test_create_adapter_missing_key_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="Missing API key"):
            create_adapter("google/gemini-3.5-flash")
