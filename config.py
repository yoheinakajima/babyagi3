"""
Configuration loader for BabyAGI.

Loads configuration from YAML file with environment variable substitution.
"""

import os
import re
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config.yaml in current dir.

    Returns:
        Configuration dict with env vars substituted.
    """
    # Find config file
    if config_path is None:
        config_path = os.environ.get("BABYAGI_CONFIG", "config.yaml")

    path = Path(config_path)
    if not path.exists():
        # Return minimal default config
        return _default_config()

    # Load YAML
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed. Using default config.")
        print("Install with: pip install pyyaml")
        return _default_config()

    with open(path) as f:
        content = f.read()

    # Substitute environment variables: ${VAR_NAME} or ${VAR_NAME:default}
    content = _substitute_env_vars(content)

    config = yaml.safe_load(content) or {}

    # Merge with defaults
    return _merge_with_defaults(config)


def _substitute_env_vars(content: str) -> str:
    """Replace ${VAR} and ${VAR:default} with environment values."""

    def replace(match):
        var_expr = match.group(1)
        if ":" in var_expr:
            var_name, default = var_expr.split(":", 1)
        else:
            var_name, default = var_expr, ""
        return os.environ.get(var_name, default)

    # Match ${VAR} or ${VAR:default}
    pattern = r"\$\{([^}]+)\}"
    return re.sub(pattern, replace, content)


def _default_config() -> dict:
    """Return minimal default configuration."""
    return {
        "owner": {
            "id": os.environ.get("OWNER_ID", "owner"),
            "email": os.environ.get("OWNER_EMAIL", ""),
            "contacts": {
                "email": os.environ.get("OWNER_EMAIL", ""),
            }
        },
        "channels": {
            "cli": {"enabled": True},
            "email": {
                "enabled": bool(os.environ.get("AGENTMAIL_API_KEY")),
                "poll_interval": 60,
            },
            "voice": {"enabled": False},
        },
        "agent": {
            "model": "claude-sonnet-4-20250514",
            "name": os.environ.get("AGENT_NAME", "Assistant"),
            "description": os.environ.get("AGENT_DESCRIPTION", "a helpful AI assistant"),
            "objective": os.environ.get(
                "AGENT_OBJECTIVE",
                "Help my owner with tasks, manage their digital presence, and handle communications on their behalf."
            ),
            "behavior": {
                "spending": {
                    "require_approval": True,
                    "auto_approve_limit": 0.0,
                },
                "external_policy": {
                    "respond_to_unknown": True,
                    "consult_owner_threshold": "medium",
                },
                "accounts": {
                    "use_agent_email": True,
                    "check_existing_first": True,
                },
            },
        },
    }


def _merge_with_defaults(config: dict) -> dict:
    """Merge user config with defaults."""
    defaults = _default_config()

    # Deep merge
    def merge(base, override):
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result

    return merge(defaults, config)


def get_channel_config(config: dict, channel: str) -> dict:
    """Get configuration for a specific channel.

    Args:
        config: Full configuration dict
        channel: Channel name (cli, email, voice, etc.)

    Returns:
        Channel configuration dict, or empty dict if not found.
    """
    return config.get("channels", {}).get(channel, {})


def is_channel_enabled(config: dict, channel: str) -> bool:
    """Check if a channel is enabled.

    Args:
        config: Full configuration dict
        channel: Channel name

    Returns:
        True if channel is enabled.
    """
    channel_config = get_channel_config(config, channel)
    return channel_config.get("enabled", False)
