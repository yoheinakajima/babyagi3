"""Lazy loader for optional API-integrated tools.

Modules here are imported only when their required API keys are present,
so optional tools stay hidden/off by default.
"""

from __future__ import annotations

import importlib
import os

_OPTIONAL_MODULES = {
    "tools.optional.peopledatalabs": ["PEOPLEDATALABS_API_KEY"],
    "tools.optional.voilanorbert": ["VOILANORBERT_API_KEY"],
    "tools.optional.hunter": ["HUNTER_API_KEY"],
    "tools.optional.exa": ["EXA_API_KEY"],
    "tools.optional.happenstance": ["HAPPENSTANCE_API_KEY"],
    "tools.optional.x_api": ["X_API_BEARER_TOKEN"],
    "tools.optional.runwayml": ["RUNWAY_API_KEY"],
    "tools.optional.elevenlabs": ["ELEVENLABS_API_KEY"],
    "tools.optional.videodb": ["VIDEODB_API_KEY"],
    "tools.optional.godaddy": ["GODADDY_API_KEY", "GODADDY_API_SECRET"],
    "tools.optional.shopify": ["SHOPIFY_ACCESS_TOKEN", "SHOPIFY_STORE_DOMAIN"],
    "tools.optional.printful": ["PRINTFUL_API_KEY"],
    "tools.optional.github_api": ["GITHUB_TOKEN"],
}

_LOADED_MODULES: set[str] = set()


def load_optional_tools() -> list[str]:
    """Import optional tool modules when required env vars are set.

    Safe to call repeatedly. Returns modules newly loaded in this call.
    """
    loaded_now = []

    for module_name, required_env in _OPTIONAL_MODULES.items():
        if module_name in _LOADED_MODULES:
            continue
        if all(os.getenv(var) for var in required_env):
            importlib.import_module(module_name)
            _LOADED_MODULES.add(module_name)
            loaded_now.append(module_name)

    return loaded_now
