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


@tool
def get_secret(name: str) -> dict:
    """Retrieve a stored secret/API key.

    Checks in order:
    1. Environment variables (NAME as-is, or with _API_KEY suffix)
    2. Keyring storage

    Args:
        name: Name of the secret (e.g., "SERPAPI", "GITHUB_TOKEN")
    """
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
                # Don't return the actual value - tools should read from env directly
                "env_var": env_name
            }

    # Check keyring
    keyring = _get_keyring()
    if keyring:
        try:
            value = keyring.get_password(KEYRING_SERVICE, name_upper)
            if value:
                # Also set in environment for tool usage
                os.environ[f"{name_upper}_API_KEY"] = value
                return {
                    "found": True,
                    "name": name,
                    "source": "keyring",
                    "masked_value": _mask_key(value),
                    "env_var": f"{name_upper}_API_KEY"
                }
        except Exception as e:
            return {"error": f"Keyring error: {e}"}

    return {
        "found": False,
        "name": name,
        "message": f"No secret found for '{name}'. Use request_api_key to ask the user, or store_secret if you have the value."
    }


@tool
def store_secret(name: str, value: str) -> dict:
    """Store a secret/API key securely.

    Use this after retrieving an API key (e.g., from browser automation)
    or when the user provides one.

    The key will be:
    1. Stored in keyring for persistence
    2. Set as environment variable for immediate use

    Args:
        name: Name for the secret (e.g., "SERPAPI", "OPENAI")
        value: The actual secret value to store
    """
    # Normalize name
    name_upper = name.upper().replace("-", "_")

    # Set environment variable immediately
    env_var = f"{name_upper}_API_KEY"
    os.environ[env_var] = value

    # Store in keyring for persistence
    keyring = _get_keyring()
    if keyring:
        try:
            keyring.set_password(KEYRING_SERVICE, name_upper, value)
            return {
                "stored": True,
                "name": name,
                "location": "keyring + environment",
                "env_var": env_var,
                "masked_value": _mask_key(value)
            }
        except Exception as e:
            # Keyring failed, but env var is set
            return {
                "stored": True,
                "name": name,
                "location": "environment only (keyring failed)",
                "env_var": env_var,
                "warning": str(e)
            }
    else:
        return {
            "stored": True,
            "name": name,
            "location": "environment only (keyring not installed)",
            "env_var": env_var,
            "hint": "pip install keyring for persistent storage"
        }


@tool
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
        except Exception:
            pass  # Key might not exist

    if deleted_from:
        return {"deleted": True, "name": name, "from": deleted_from}
    return {"deleted": False, "name": name, "message": "Secret not found"}


@tool
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


@tool
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
    existing = get_secret.__wrapped__(name)  # Call underlying function
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


@tool
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
        check = get_secret.__wrapped__(name)
        if check.get("found"):
            results["available"].append(name)
        else:
            results["missing"].append(name)
            results["all_available"] = False

    if results["missing"]:
        results["message"] = f"Missing API keys: {', '.join(results['missing'])}"
        results["hint"] = "Use request_api_key() for each missing key"

    return results
