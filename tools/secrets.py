"""
Secrets Management Tools

Secure storage and retrieval of API keys and credentials.

Uses the keyring library which integrates with:
- macOS Keychain
- Windows Credential Locker
- Linux Secret Service (GNOME Keyring, KWallet)
- Encrypted file fallback

Install:
    pip install keyring keyrings.cryptfile

The agent can:
- Request API keys from the user securely
- Store retrieved API keys (e.g., from browser automation)
- Retrieve stored keys for tool usage
"""

import os
from tools import tool

import logging

logger = logging.getLogger(__name__)

# Service name for keyring storage
KEYRING_SERVICE = "babyagi"

# Track pending key requests (for user interaction)
_pending_requests = {}


def _get_keyring():
    """Get keyring module, with fallback to environment variables."""
    try:
        import keyring
        return keyring
    except ImportError:
        return None


def _mask_key(key: str) -> str:
    """Mask a key for display (show first 4 and last 4 chars)."""
    if not key or len(key) < 12:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def _check_secret(name: str) -> dict:
    """Internal helper to check for a secret (used by multiple tools)."""
    # Normalize name
    name_upper = name.upper().replace("-", "_")

    # Check environment variables first
    for env_name in [name_upper, f"{name_upper}_API_KEY", f"{name_upper}_KEY"]:
        value = os.environ.get(env_name)
        if value:
            return {
                "found": True,
                "name": name,
                "source": "environment",
                "masked_value": _mask_key(value),
                "env_var": env_name
            }

    # Check keyring - try the exact name first, then with _API_KEY suffix
    keyring = _get_keyring()
    if keyring:
        for keyring_name in [name_upper, f"{name_upper}_API_KEY"]:
            try:
                value = keyring.get_password(KEYRING_SERVICE, keyring_name)
                if value:
                    # Also set in environment for tool usage
                    os.environ[name_upper] = value
                    return {
                        "found": True,
                        "name": name,
                        "source": "keyring",
                        "masked_value": _mask_key(value),
                        "env_var": name_upper
                    }
            except Exception as e:
                return {"error": f"Keyring error: {e}"}

    return {
        "found": False,
        "name": name,
        "message": f"No secret found for '{name}'. Use request_api_key to ask the user, or store_secret if you have the value."
    }




def _refresh_optional_tools_if_needed(agent=None) -> list[str]:
    """Load optional tool modules and register any newly available tools."""
    try:
        from tools.optional import load_optional_tools
        loaded_modules = load_optional_tools()
    except Exception:
        return []

    if not agent:
        return loaded_modules

    # Best effort: if we have an agent instance, refresh tool registrations
    # so newly configured optional tools are callable immediately.
    try:
        from agent import Tool
        from tools import get_all_tools

        existing = set(agent.tools.keys()) if hasattr(agent, "tools") else set()
        for tool in get_all_tools(Tool):
            if hasattr(agent, "register"):
                agent.register(tool)
        updated = set(agent.tools.keys()) if hasattr(agent, "tools") else set()
        return sorted(updated - existing)
    except Exception:
        return loaded_modules

@tool(packages=["keyring"])
def get_secret(name: str) -> dict:
    """Retrieve a stored secret/API key.

    Checks in order:
    1. Environment variables (NAME as-is, or with _API_KEY suffix)
    2. Keyring storage

    Args:
        name: Name of the secret (e.g., "SERPAPI", "GITHUB_TOKEN")
    """
    return _check_secret(name)


@tool(packages=["keyring"])
def store_secret(name: str, value: str, agent=None) -> dict:
    """Store a secret/API key securely.

    Use this after retrieving an API key (e.g., from browser automation)
    or when the user provides one.

    The key will be:
    1. Stored in keyring for persistence
    2. Set as environment variable for immediate use

    IMPORTANT: Use the exact environment variable name expected by the system.
    For example: "SENDBLUE_API_KEY", "AGENTMAIL_API_KEY", "SENDBLUE_PHONE_NUMBER",
    "OWNER_PHONE". Do NOT use short names like "SENDBLUE" - use the full env var name.

    Args:
        name: Full environment variable name (e.g., "SENDBLUE_API_KEY", "AGENTMAIL_API_KEY", "OWNER_PHONE")
        value: The actual secret value to store
    """
    # Normalize name
    name_upper = name.upper().replace("-", "_")

    # Determine the environment variable name.
    # If the name already looks like a full env var (contains _API_KEY, _API_SECRET,
    # _SECRET, _TOKEN, _PHONE, _NUMBER, _EMAIL, _ID, _URL, _NAME), use it as-is.
    # Otherwise, append _API_KEY for backwards compatibility.
    known_suffixes = (
        "_API_KEY", "_API_SECRET", "_SECRET", "_TOKEN", "_KEY",
        "_PHONE", "_PHONE_NUMBER", "_NUMBER", "_EMAIL", "_ID",
        "_URL", "_NAME", "_BIO", "_GOAL", "_TIMEZONE",
    )
    if any(name_upper.endswith(suffix) for suffix in known_suffixes):
        env_var = name_upper
    else:
        env_var = f"{name_upper}_API_KEY"

    # Set environment variable immediately
    os.environ[env_var] = value

    # Store in keyring for persistence (using the full env var name as the key)
    keyring = _get_keyring()
    if keyring:
        try:
            keyring.set_password(KEYRING_SERVICE, env_var, value)
            newly_enabled_tools = _refresh_optional_tools_if_needed(agent)
            return {
                "stored": True,
                "name": name,
                "location": "keyring + environment",
                "env_var": env_var,
                "masked_value": _mask_key(value),
                "newly_enabled_tools": newly_enabled_tools,
            }
        except Exception as e:
            # Keyring failed, but env var is set
            newly_enabled_tools = _refresh_optional_tools_if_needed(agent)
            return {
                "stored": True,
                "name": name,
                "location": "environment only (keyring failed)",
                "env_var": env_var,
                "warning": str(e),
                "newly_enabled_tools": newly_enabled_tools,
            }
    else:
        newly_enabled_tools = _refresh_optional_tools_if_needed(agent)
        return {
            "stored": True,
            "name": name,
            "location": "environment only (keyring not installed)",
            "env_var": env_var,
            "hint": "pip install keyring for persistent storage",
            "newly_enabled_tools": newly_enabled_tools,
        }


@tool(packages=["keyring"])
def delete_secret(name: str) -> dict:
    """Delete a stored secret.

    Removes from both keyring and environment.

    Args:
        name: Name of the secret to delete
    """
    name_upper = name.upper().replace("-", "_")

    deleted_from = []

    # Remove from environment
    for env_name in [name_upper, f"{name_upper}_API_KEY", f"{name_upper}_KEY"]:
        if env_name in os.environ:
            del os.environ[env_name]
            deleted_from.append(f"env:{env_name}")

    # Remove from keyring
    keyring = _get_keyring()
    if keyring:
        try:
            keyring.delete_password(KEYRING_SERVICE, name_upper)
            deleted_from.append("keyring")
        except Exception as e:
            logger.debug("Could not delete secret '%s' from keyring (may not exist): %s", name, e)

    if deleted_from:
        return {"deleted": True, "name": name, "from": deleted_from}
    return {"deleted": False, "name": name, "message": "Secret not found"}


@tool  # No external deps - just checks environment
def list_secrets() -> dict:
    """List all known secrets (with masked values).

    Shows secrets from environment variables and keyring.
    Actual values are never shown - only masked previews.
    """
    secrets = []

    # Common API key patterns in environment
    api_key_patterns = ["_API_KEY", "_KEY", "_TOKEN", "_SECRET"]

    for key, value in os.environ.items():
        if any(pattern in key for pattern in api_key_patterns):
            # Skip internal/system keys
            if key.startswith("_") or key in ["PATH", "HOME", "USER"]:
                continue
            secrets.append({
                "name": key,
                "source": "environment",
                "masked_value": _mask_key(value)
            })

    return {
        "count": len(secrets),
        "secrets": secrets,
        "note": "Values are masked for security. Use get_secret(name) to access."
    }


@tool(packages=["keyring"])
def request_api_key(
    name: str,
    service: str,
    purpose: str,
    signup_url: str = None
) -> dict:
    """Request an API key from the user.

    Use this when a tool needs an API key that isn't available.
    This creates a structured request that clearly explains what's needed.

    The agent should then wait for the user to provide the key,
    or offer to use browser automation to obtain it.

    Args:
        name: Key name to store as (e.g., "SERPAPI")
        service: Name of the service (e.g., "SerpAPI")
        purpose: Why the agent needs this key
        signup_url: URL where user can get the key (optional)
    """
    name_upper = name.upper().replace("-", "_")

    # Check if we actually need it
    existing = _check_secret(name)
    if existing.get("found"):
        return {
            "already_available": True,
            "name": name,
            "message": f"API key for {service} is already available"
        }

    request = {
        "type": "api_key_request",
        "name": name_upper,
        "service": service,
        "purpose": purpose,
        "signup_url": signup_url,
        "how_to_provide": f"Provide the key and I'll store it securely using store_secret('{name_upper}', 'your-key-here')",
        "alternatives": []
    }

    if signup_url:
        request["alternatives"].append(
            f"I can try to obtain this automatically using browser automation (browse to {signup_url})"
        )

    _pending_requests[name_upper] = request

    return request


@tool(packages=["keyring"])
def check_required_secrets(required: list) -> dict:
    """Check if all required secrets are available.

    Useful before running a workflow that needs multiple API keys.

    Args:
        required: List of secret names to check (e.g., ["OPENAI", "SERPAPI"])
    """
    results = {
        "all_available": True,
        "available": [],
        "missing": []
    }

    for name in required:
        check = _check_secret(name)
        if check.get("found"):
            results["available"].append(name)
        else:
            results["missing"].append(name)
            results["all_available"] = False

    if results["missing"]:
        results["message"] = f"Missing API keys: {', '.join(results['missing'])}"
        results["hint"] = "Use request_api_key() for each missing key"

    return results


# =============================================================================
# Configuration mapping: env var name -> config dict path
# =============================================================================

_CONFIG_FIELDS = {
    # Owner fields
    "OWNER_NAME": ("owner", "name"),
    "OWNER_EMAIL": ("owner", "email"),
    "OWNER_BIO": ("owner", "bio"),
    "OWNER_GOAL": ("owner", "goal"),
    "OWNER_PHONE": ("owner", "phone"),
    "OWNER_TIMEZONE": ("owner", "timezone"),
    # Agent fields
    "AGENT_NAME": ("agent", "name"),
    # Channel: SendBlue
    "SENDBLUE_API_KEY": ("channels", "sendblue", "api_key"),
    "SENDBLUE_API_SECRET": ("channels", "sendblue", "api_secret"),
    "SENDBLUE_PHONE_NUMBER": ("channels", "sendblue", "from_number"),
    # Channel: Email (AgentMail)
    "AGENTMAIL_API_KEY": ("channels", "email", "api_key"),
    "AGENTMAIL_INBOX_ID": ("channels", "email", "inbox_id"),
    # Composio integrations
    "COMPOSIO_API_KEY": ("composio", "api_key"),
}


def _set_nested(d: dict, path: tuple, value):
    """Set a value in a nested dict by key path, creating intermediates."""
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


@tool
def update_config(key: str, value: str, agent=None) -> dict:
    """Update a BabyAGI configuration value at runtime.

    Use this to change phone numbers, API keys, owner info, or channel settings
    AFTER initialization. The change takes effect immediately — environment variable
    is set, the in-memory config is updated, and sender/listener credentials are
    refreshed so the new value is used on the next message.

    Supported keys (use exact names):
      Owner info:     OWNER_NAME, OWNER_EMAIL, OWNER_BIO, OWNER_GOAL, OWNER_PHONE, OWNER_TIMEZONE
      Agent:          AGENT_NAME
      SendBlue:       SENDBLUE_API_KEY, SENDBLUE_API_SECRET, SENDBLUE_PHONE_NUMBER
      AgentMail:      AGENTMAIL_API_KEY, AGENTMAIL_INBOX_ID
      Composio:       COMPOSIO_API_KEY

    For API keys and secrets, this also persists the value to keyring storage
    (same as store_secret). For non-secret config (names, bios), it only sets
    the environment variable and updates the running config.

    Args:
        key: Configuration key (e.g., "SENDBLUE_PHONE_NUMBER", "OWNER_PHONE")
        value: The new value to set
    """
    key_upper = key.upper().replace("-", "_")

    # Validate key
    if key_upper not in _CONFIG_FIELDS:
        return {
            "error": f"Unknown config key: {key}",
            "valid_keys": sorted(_CONFIG_FIELDS.keys()),
        }

    # 1. Set environment variable
    os.environ[key_upper] = value

    # 2. Update the running config dict (if agent is available)
    config_updated = False
    if agent and hasattr(agent, "config"):
        path = _CONFIG_FIELDS[key_upper]
        _set_nested(agent.config, path, value)
        config_updated = True

        # Also update owner.contacts mirror for phone/email
        if key_upper == "OWNER_PHONE":
            _set_nested(agent.config, ("owner", "contacts", "phone"), value)
        elif key_upper == "OWNER_EMAIL":
            _set_nested(agent.config, ("owner", "contacts", "email"), value)

    # 3. Persist secrets to keyring (API keys, secrets, phone numbers — anything sensitive)
    secret_keys = {
        "SENDBLUE_API_KEY", "SENDBLUE_API_SECRET", "SENDBLUE_PHONE_NUMBER",
        "AGENTMAIL_API_KEY", "AGENTMAIL_INBOX_ID",
        "COMPOSIO_API_KEY",
    }
    persisted = False
    if key_upper in secret_keys:
        keyring = _get_keyring()
        if keyring:
            try:
                keyring.set_password(KEYRING_SERVICE, key_upper, value)
                persisted = True
            except Exception as e:
                logger.warning("Could not persist %s to keyring: %s", key_upper, e)

    # 4. Invalidate cached credentials in senders so they re-read on next send
    if agent and hasattr(agent, "senders"):
        if key_upper.startswith("SENDBLUE") and "sendblue" in agent.senders:
            sender = agent.senders["sendblue"]
            sender._api_key = None
            sender._api_secret = None
            sender._from_number = None
        if key_upper.startswith("AGENTMAIL") and "email" in agent.senders:
            sender = agent.senders["email"]
            sender._client = None
            sender._inbox_id = None

    # Determine if value should be masked in response
    is_sensitive = "KEY" in key_upper or "SECRET" in key_upper
    display_value = _mask_key(value) if is_sensitive else value

    newly_enabled_tools = _refresh_optional_tools_if_needed(agent)

    return {
        "updated": True,
        "key": key_upper,
        "value": display_value,
        "env_set": True,
        "config_updated": config_updated,
        "persisted_to_keyring": persisted,
        "newly_enabled_tools": newly_enabled_tools,
    }


# =============================================================================
# Startup helper: restore keyring secrets into environment
# =============================================================================

# All keys that may be stored in keyring and should be loaded at startup.
_KEYRING_KNOWN_KEYS = [
    "AGENTMAIL_API_KEY",
    "SENDBLUE_API_KEY",
    "SENDBLUE_API_SECRET",
    "COMPOSIO_API_KEY",
    # Optional tools
    "PEOPLEDATALABS_API_KEY",
    "VOILANORBERT_API_KEY",
    "HUNTER_API_KEY",
    "EXA_API_KEY",
    "HAPPENSTANCE_API_KEY",
    "X_API_BEARER_TOKEN",
    "RUNWAY_API_KEY",
    "ELEVENLABS_API_KEY",
    "VIDEODB_API_KEY",
    "GODADDY_API_KEY",
    "GODADDY_API_SECRET",
    "SHOPIFY_ACCESS_TOKEN",
    "SHOPIFY_STORE_DOMAIN",
    "PRINTFUL_API_KEY",
    "GITHUB_TOKEN",
]


def load_keyring_secrets():
    """Load secrets from keyring into environment variables at startup.

    Called early in ``main.py`` so that keys persisted during initialization
    (or via ``store_secret`` / ``update_config``) are available to SDKs like
    Composio that read directly from ``os.environ``.

    Only populates env vars that are not already set — explicit env vars,
    ``.env`` files, and Replit secrets always take precedence.
    """
    keyring = _get_keyring()
    if not keyring:
        return

    for key in _KEYRING_KNOWN_KEYS:
        if os.environ.get(key):
            continue  # already set — don't overwrite
        try:
            value = keyring.get_password(KEYRING_SERVICE, key)
            if value:
                os.environ[key] = value
                logger.debug("Loaded %s from keyring", key)
        except Exception as e:
            logger.debug("Could not load %s from keyring: %s", key, e)
