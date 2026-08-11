import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "wdoc"
    / "utils"
    / "customs"
    / "add_extra_litellm_models_metadata.py"
)
SPEC = importlib.util.spec_from_file_location("extra_litellm_metadata", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
extra_litellm_metadata = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extra_litellm_metadata)


def test_registers_minimax_models_with_current_metadata():
    registered_models = {}
    fake_litellm = SimpleNamespace(
        model_cost={},
        models_by_provider={},
        register_model=registered_models.update,
    )

    extra_litellm_metadata.add_extra_models_metadata(fake_litellm)

    expected = {
        "minimax/MiniMax-M3": {
            "max_tokens": 1_000_000,
            "input_cost_per_token": 0.6 / 1_000_000,
            "output_cost_per_token": 2.4 / 1_000_000,
            "cache_read_input_token_cost": 0.12 / 1_000_000,
            "cache_creation_input_token_cost": None,
            "input_modalities": ["text", "image", "video"],
            "thinking": ["adaptive", "disabled"],
        },
        "minimax/MiniMax-M2.7": {
            "max_tokens": 204_800,
            "input_cost_per_token": 0.3 / 1_000_000,
            "output_cost_per_token": 1.2 / 1_000_000,
            "cache_read_input_token_cost": 0.06 / 1_000_000,
            "cache_creation_input_token_cost": 0.375 / 1_000_000,
            "input_modalities": ["text"],
            "thinking": ["always_on"],
        },
    }

    for model_id, metadata in expected.items():
        assert model_id in fake_litellm.models_by_provider["minimax"]
        registered = registered_models[model_id]
        assert registered["litellm_provider"] == "minimax"
        assert registered["mode"] == "chat"
        for key, value in metadata.items():
            assert registered[key] == value


def test_does_not_replace_existing_litellm_metadata():
    existing_metadata = {"source": "litellm"}
    registered_models = {}
    fake_litellm = SimpleNamespace(
        model_cost={"minimax/MiniMax-M3": existing_metadata},
        models_by_provider={"minimax": {"minimax/MiniMax-M3"}},
        register_model=registered_models.update,
    )

    extra_litellm_metadata.add_extra_models_metadata(fake_litellm)

    assert fake_litellm.model_cost["minimax/MiniMax-M3"] is existing_metadata
    assert "minimax/MiniMax-M3" not in registered_models
    assert "minimax/MiniMax-M2.7" in registered_models


def test_registration_failure_only_logs_a_warning(monkeypatch):
    warnings = []
    fake_litellm = SimpleNamespace(model_cost={})
    monkeypatch.setattr(extra_litellm_metadata.logger, "warning", warnings.append)
    monkeypatch.setattr(
        extra_litellm_metadata,
        "_add_extra_models_metadata",
        lambda litellm: (_ for _ in ()).throw(RuntimeError("registration failed")),
    )

    extra_litellm_metadata.add_extra_models_metadata(fake_litellm)

    assert warnings == [
        "Could not add extra LiteLLM model metadata: registration failed"
    ]


@pytest.mark.parametrize(
    ("region", "protocol", "expected"),
    [
        ("global_en", "openai", "https://api.minimax.io/v1"),
        ("global_en", "anthropic", "https://api.minimax.io/anthropic"),
        ("cn_zh", "openai", "https://api.minimaxi.com/v1"),
        ("cn_zh", "anthropic", "https://api.minimaxi.com/anthropic"),
    ],
)
def test_minimax_endpoint_recipes(region, protocol, expected):
    assert extra_litellm_metadata.get_minimax_api_base(region, protocol) == expected


def test_minimax_endpoint_recipe_rejects_unknown_selection():
    with pytest.raises(ValueError, match="Unsupported MiniMax endpoint selection"):
        extra_litellm_metadata.get_minimax_api_base("unknown", "openai")
