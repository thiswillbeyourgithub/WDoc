"""Add model metadata that is not yet available in LiteLLM."""

from typing import Any, Literal

from loguru import logger


MINIMAX_PROVIDER = "minimax"

# Endpoint sources:
# https://platform.minimax.io/docs
# https://platform.minimaxi.com/docs
MINIMAX_ENDPOINTS = {
    "global_en": {
        "openai_base_url": "https://api.minimax.io/v1",
        "anthropic_base_url": "https://api.minimax.io/anthropic",
    },
    "cn_zh": {
        "openai_base_url": "https://api.minimaxi.com/v1",
        "anthropic_base_url": "https://api.minimaxi.com/anthropic",
    },
}

# Model metadata sources:
# https://platform.minimax.io/docs/api-reference/api-overview
# https://platform.minimaxi.com/docs/api-reference/api-overview
MINIMAX_MODELS = {
    "minimax/MiniMax-M3": {
        "litellm_provider": MINIMAX_PROVIDER,
        "mode": "chat",
        "max_tokens": 1_000_000,
        "max_input_tokens": 1_000_000,
        "input_cost_per_token": 0.6 / 1_000_000,
        "output_cost_per_token": 2.4 / 1_000_000,
        "cache_read_input_token_cost": 0.12 / 1_000_000,
        "cache_creation_input_token_cost": None,
        "input_modalities": ["text", "image", "video"],
        "thinking": ["adaptive", "disabled"],
        "supports_vision": True,
        "supports_reasoning": True,
        "supports_adaptive_thinking": True,
    },
    "minimax/MiniMax-M2.7": {
        "litellm_provider": MINIMAX_PROVIDER,
        "mode": "chat",
        "max_tokens": 204_800,
        "max_input_tokens": 204_800,
        "input_cost_per_token": 0.3 / 1_000_000,
        "output_cost_per_token": 1.2 / 1_000_000,
        "cache_read_input_token_cost": 0.06 / 1_000_000,
        "cache_creation_input_token_cost": 0.375 / 1_000_000,
        "input_modalities": ["text"],
        "thinking": ["always_on"],
        "supports_reasoning": True,
    },
}


def get_minimax_api_base(
    region: Literal["global_en", "cn_zh"],
    protocol: Literal["openai", "anthropic"] = "openai",
) -> str:
    """Return the configured MiniMax base URL for a region and protocol."""
    try:
        return MINIMAX_ENDPOINTS[region][f"{protocol}_base_url"]
    except KeyError as err:
        raise ValueError(
            f"Unsupported MiniMax endpoint selection: {region}/{protocol}"
        ) from err


def _add_extra_models_metadata(litellm: Any) -> None:
    models_to_add = {
        model_id: metadata
        for model_id, metadata in MINIMAX_MODELS.items()
        if model_id not in litellm.model_cost
    }
    if not models_to_add:
        return

    litellm.register_model(models_to_add)
    provider_models = litellm.models_by_provider.setdefault(MINIMAX_PROVIDER, set())
    provider_models.update(models_to_add)


def add_extra_models_metadata(litellm: Any) -> None:
    """Add missing model metadata without preventing wdoc from starting."""
    try:
        _add_extra_models_metadata(litellm)
    except Exception as err:
        logger.warning(f"Could not add extra LiteLLM model metadata: {err}")
