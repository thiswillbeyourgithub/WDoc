import importlib.util
from pathlib import Path

import litellm
import pytest


MODULE_PATH = (
    Path(__file__).parents[1] / "wdoc" / "utils" / "customs" / "litellm_models.py"
)
SPEC = importlib.util.spec_from_file_location("wdoc_litellm_models", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
litellm_models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(litellm_models)


def test_registers_minimax_models_with_current_metadata():
    litellm_models.register_wdoc_models(litellm)

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
        assert model_id in litellm.models_by_provider["minimax"]
        registered = litellm.model_cost[model_id]
        assert registered["litellm_provider"] == "minimax"
        assert registered["mode"] == "chat"
        for key, value in metadata.items():
            assert registered[key] == value


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
    assert litellm_models.get_minimax_api_base(region, protocol) == expected


def test_minimax_endpoint_recipe_rejects_unknown_selection():
    with pytest.raises(ValueError, match="Unsupported MiniMax endpoint selection"):
        litellm_models.get_minimax_api_base("unknown", "openai")
