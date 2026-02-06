"""
Skills and Composio integration tools.

This module provides tools for:
1. Acquiring skills from SKILL.md files (behavioral instructions)
2. Setting up Composio integrations (external app connections)

Tool Types:
- "executable": Python code that runs directly (default)
- "skill": Returns behavioral instructions when called
- "composio": Thin wrapper that calls Composio library
"""

import re
import yaml
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent import Agent, Tool
    from memory.models import ToolDefinition


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


def create_skill_tool(tool_def: "ToolDefinition", Tool: type) -> "Tool":
    """
    Create a Tool instance from a skill-type ToolDefinition.

    When called, skill tools return their instructions as context
    for the AI to follow in subsequent actions.
    """

    def skill_fn(params: dict, agent: "Agent") -> dict:
        action = params.get("action", "activate")

        if action == "activate":
            return {
                "type": "skill_instructions",
                "skill": tool_def.name,
                "instructions": tool_def.skill_content,
                "description": tool_def.description,
                "guidance": (
                    "I've loaded the skill instructions above. "
                    "Follow these guidelines for your next actions."
                ),
            }

        elif action == "check":
            # Verify required tools exist
            required = _extract_required_tools(tool_def.skill_content or "")
            available = [t for t in required if t in agent.tools]
            missing = [t for t in required if t not in agent.tools]

            return {
                "skill": tool_def.name,
                "ready": len(missing) == 0,
                "available_tools": available,
                "missing_tools": missing,
                "suggestion": (
                    f"Missing tools: {missing}. Set these up first." if missing else None
                ),
            }

        elif action == "info":
            return {
                "name": tool_def.name,
                "description": tool_def.description,
                "type": "skill",
                "usage_count": tool_def.usage_count,
                "depends_on": tool_def.depends_on,
            }

        return {"error": f"Unknown action: {action}. Use 'activate', 'check', or 'info'."}

    return Tool(
        name=tool_def.name,
        description=f"Skill: {tool_def.description}. Use action='activate' to get behavioral instructions.",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["activate", "check", "info"],
                    "description": (
                        "activate: Get instructions to follow. "
                        "check: Verify required tools exist. "
                        "info: Get skill metadata."
                    ),
                }
            },
        },
        fn=skill_fn,
    )


def _extract_required_tools(skill_content: str) -> list[str]:
    """Extract tool names mentioned in skill content."""
    # Look for patterns like `tool_name` or tool_name()
    pattern = r"`([a-z_][a-z0-9_]*)`|([a-z_][a-z0-9_]*)\(\)"
    matches = re.findall(pattern, skill_content.lower())
    tools = set()
    for match in matches:
        tool_name = match[0] or match[1]
        if tool_name and not tool_name.startswith(("if", "for", "while", "def", "class")):
            tools.add(tool_name)
    return list(tools)


def parse_skill_md(content: str) -> tuple[dict, str]:
    """
    Parse a SKILL.md file into metadata and instructions.

    Returns:
        (metadata_dict, instructions_markdown)
    """
    # Check for YAML frontmatter
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1]) or {}
                instructions = parts[2].strip()
                return metadata, instructions
            except yaml.YAMLError:
                pass

    # No frontmatter - extract name from first heading
    lines = content.strip().split("\n")
    name = "unnamed-skill"
    description = ""

    for line in lines:
        if line.startswith("# "):
            name = line[2:].strip().lower().replace(" ", "-")
            break

    # Use first paragraph as description
    in_paragraph = False
    for line in lines:
        if line.startswith("#"):
            continue
        if line.strip():
            if not in_paragraph:
                in_paragraph = True
                description = line.strip()
            else:
                description += " " + line.strip()
        elif in_paragraph:
            break

    return {"name": name, "description": description[:200]}, content


def _acquire_skill_tool(agent: "Agent", Tool: type) -> "Tool":
    """
    Create the acquire_skill tool for learning new skills from SKILL.md files.
    """

    def fn(params: dict, ag: "Agent") -> dict:
        source = params["source"]
        auto_enable = params.get("auto_enable", False)
        skip_safety = params.get("skip_safety", False)

        # 1. Fetch content
        try:
            if source.startswith("http"):
                import httpx

                response = httpx.get(source, follow_redirects=True, timeout=30)
                response.raise_for_status()
                content = response.text
            else:
                from pathlib import Path

                content = Path(source).expanduser().read_text()
        except Exception as e:
            return {"success": False, "error": f"Failed to fetch skill: {e}"}

        # 2. Parse SKILL.md
        metadata, instructions = parse_skill_md(content)
        name = metadata.get("name", "unnamed-skill")
        description = metadata.get("description", "No description provided")

        # Prefix with skill_ if not already
        if not name.startswith("skill_"):
            name = f"skill_{name}"

        # 3. Safety scan (unless skipped)
        safety_result = {"safety_score": 100, "risk_level": "low", "findings": []}
        if not skip_safety:
            safety_result = _scan_skill_safety(content, ag)

            if safety_result["risk_level"] == "critical":
                return {
                    "success": False,
                    "reason": "Skill rejected - dangerous patterns detected",
                    "findings": safety_result["findings"],
                }

        # 4. Determine if we should enable
        should_enable = auto_enable and safety_result["safety_score"] >= 80

        # 5. Save to database
        if ag.memory is None:
            return {"success": False, "error": "Memory system not available"}

        tool_def = ag.memory.store.save_tool_definition(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["activate", "check", "info"],
                    }
                },
            },
            tool_type="skill",
            skill_content=instructions,
            category="skill",
            is_dynamic=True,
        )

        # 6. Enable/disable based on safety
        if should_enable:
            ag.memory.store.enable_tool(name)
            # Register with agent
            skill_tool = create_skill_tool(tool_def, Tool)
            ag.tools[skill_tool.name] = skill_tool
        else:
            ag.memory.store.disable_tool(name, reason="Pending safety review")

        return {
            "success": True,
            "skill_id": tool_def.id,
            "name": name,
            "enabled": should_enable,
            "safety_score": safety_result["safety_score"],
            "risk_level": safety_result["risk_level"],
            "requires_review": not should_enable,
            "message": (
                f"Skill '{name}' acquired and activated."
                if should_enable
                else f"Skill '{name}' saved but requires review (safety_score={safety_result['safety_score']})"
            ),
        }

    return Tool(
        name="acquire_skill",
        description="""Acquire a skill from a URL or file path containing a SKILL.md file.

Skills are behavioral instructions that guide how to perform tasks.
They're different from executable tools - skills teach patterns and behaviors.

The skill will be scanned for safety before activation:
- Auto-enabled if safe (score >= 80) and auto_enable=True
- Otherwise saved but disabled pending review

Examples:
- acquire_skill(source="https://example.com/skills/code-review/SKILL.md")
- acquire_skill(source="~/my-skills/email-triage.md", auto_enable=True)""",
        parameters={
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "URL or file path to the SKILL.md file",
                },
                "auto_enable": {
                    "type": "boolean",
                    "description": "If True and safe, enable immediately (default: False)",
                },
                "skip_safety": {
                    "type": "boolean",
                    "description": "Skip safety scan (use with caution, default: False)",
                },
            },
            "required": ["source"],
        },
        fn=fn,
    )


def _scan_skill_safety(content: str, agent: "Agent") -> dict:
    """
    Scan a skill for safety issues.

    Uses heuristics and optionally AI analysis.
    """
    findings = []
    score = 100

    # Check for dangerous patterns
    dangerous_patterns = [
        (r"eval\s*\(", "Uses eval() - potential code injection", 30),
        (r"exec\s*\(", "Uses exec() - potential code injection", 30),
        (r"__import__", "Uses __import__ - potential module injection", 20),
        (r"os\.system", "Uses os.system - potential command injection", 25),
        (r"subprocess", "Uses subprocess - potential command execution", 15),
        (r"rm\s+-rf", "Contains rm -rf command", 40),
        (r"curl.*\|\s*sh", "Pipes curl to shell - dangerous", 50),
        (r"password|secret|api.?key|token", "References credentials", 5),
        (r"send.*to.*external", "May send data externally", 10),
        (r"ignore.*previous.*instructions", "Prompt injection attempt", 50),
        (r"disregard.*above", "Prompt injection attempt", 50),
    ]

    content_lower = content.lower()
    for pattern, description, penalty in dangerous_patterns:
        if re.search(pattern, content_lower, re.IGNORECASE):
            findings.append(description)
            score -= penalty

    # Ensure score doesn't go negative
    score = max(0, score)

    # Determine risk level
    if score >= 80:
        risk_level = "low"
    elif score >= 50:
        risk_level = "medium"
    elif score >= 20:
        risk_level = "high"
    else:
        risk_level = "critical"

    return {
        "safety_score": score,
        "risk_level": risk_level,
        "findings": findings,
        "scanned_at": __import__("datetime").datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSIO TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level lock to prevent duplicate connection race conditions (Fix #1)
import threading
_connect_locks: dict[str, threading.Lock] = {}
_connect_locks_guard = threading.Lock()


def _get_connect_lock(key: str) -> threading.Lock:
    """Get or create a per-app connection lock to prevent duplicate connections."""
    with _connect_locks_guard:
        if key not in _connect_locks:
            _connect_locks[key] = threading.Lock()
        return _connect_locks[key]


def create_composio_tool(tool_def: "ToolDefinition", Tool: type, composio_client) -> "Tool":
    """
    Create a Tool instance from a composio-type ToolDefinition.

    Uses SDK 0.11+ API: client.tools.execute(slug, arguments, user_id=...).
    Pre-loads tool schema into the SDK's internal cache to avoid 404 errors.
    """

    # Pre-load the tool schema into the SDK's cache so execute() can find it (Fix #3)
    _ensure_tool_schema_cached(composio_client, tool_def.composio_action)

    def composio_fn(params: dict, agent: "Agent") -> dict:
        try:
            user_id = agent.config.get("composio_entity_id", "default")
            result = composio_client.tools.execute(
                slug=tool_def.composio_action,
                arguments=params,
                user_id=user_id,
                dangerously_skip_version_check=True,
            )
            return result
        except Exception as e:
            error_str = str(e)
            # If we get a 404, try re-caching the schema and retry once
            if "404" in error_str or "not found" in error_str.lower():
                try:
                    _ensure_tool_schema_cached(composio_client, tool_def.composio_action, force=True)
                    user_id = agent.config.get("composio_entity_id", "default")
                    return composio_client.tools.execute(
                        slug=tool_def.composio_action,
                        arguments=params,
                        user_id=user_id,
                        dangerously_skip_version_check=True,
                    )
                except Exception as retry_err:
                    return {"error": f"Composio execution failed after retry: {retry_err}"}
            return {"error": f"Composio execution failed: {e}"}

    return Tool(
        name=tool_def.name,
        description=tool_def.description,
        parameters=tool_def.parameters,
        fn=composio_fn,
        packages=["composio"],
    )


def _ensure_tool_schema_cached(composio_client, action_slug: str, force: bool = False) -> None:
    """Pre-load a tool's schema into the SDK's internal _tool_schemas cache.

    The SDK's tools.execute() looks up the schema in _tool_schemas first. If
    not found, it falls back to client.tools.retrieve() which can 404 if the
    slug format doesn't match exactly. By pre-loading via get_raw_composio_tools,
    we ensure the schema is cached correctly. (Fix #3)
    """
    try:
        cache = composio_client.tools._tool_schemas
        if not force and action_slug in cache:
            return
        # Load the single tool by slug
        tools = composio_client.tools.get_raw_composio_tools(tools=[action_slug])
        for tool in tools:
            slug = getattr(tool, 'slug', None) or getattr(tool, 'name', None)
            if slug:
                cache[slug] = tool
    except Exception as e:
        logger.debug("Could not pre-cache tool schema for '%s': %s", action_slug, e)


def _composio_setup_tool(agent: "Agent", Tool: type) -> "Tool":
    """
    Create the composio_setup tool for managing Composio integrations.

    This tool provides full API-first Composio management:
    - Discover available apps and their authentication requirements
    - Check existing connections and their status
    - Initiate OAuth flows or collect API credentials
    - Poll for OAuth completion
    - Enable/disable Composio actions as agent tools
    """

    def fn(params: dict, ag: "Agent") -> dict:
        action = params["action"]
        app = params.get("app")
        actions = params.get("actions")
        credentials = params.get("credentials")  # For API key/bearer token auth
        connection_id = params.get("connection_id")  # For polling specific connections
        account_label = params.get("account_label")  # For multiple accounts per platform

        # Import composio (new SDK v0.10+)
        try:
            from composio import Composio
        except ImportError:
            return {
                "error": "Composio not installed. Run: pip install composio",
                "suggestion": "Install composio with: pip install composio",
            }

        try:
            client = Composio()
        except Exception as e:
            return {
                "error": f"Failed to initialize Composio client: {e}",
                "suggestion": "Make sure COMPOSIO_API_KEY is set. Get your API key from https://app.composio.dev/settings",
            }

        # user_id replaces entity_id in SDK v0.10+
        # Support account_label for multiple accounts on same platform
        base_user_id = ag.config.get("composio_entity_id", "default")
        user_id = f"{base_user_id}_{account_label}" if account_label else base_user_id

        def get_or_create_auth_config(toolkit_name: str, custom_credentials: dict | None = None) -> str | None:
            """Get existing auth_config_id or create one via SDK.

            For OAuth apps, uses Composio's managed OAuth (no client credentials needed).
            For API key apps, creates a custom auth config with provided credentials.
            """
            toolkit = toolkit_name.upper()

            # Check if we have a stored auth config
            if ag.memory:
                stored = ag.memory.store.get_composio_auth_config(toolkit)
                if stored:
                    return stored["auth_config_id"]

            # Create via SDK (0.11+: client.auth_configs.create)
            try:
                if custom_credentials:
                    if "api_key" in custom_credentials:
                        auth_scheme = "API_KEY"
                    elif "token" in custom_credentials:
                        auth_scheme = "BEARER_TOKEN"
                    elif "username" in custom_credentials and "password" in custom_credentials:
                        auth_scheme = "BASIC"
                    else:
                        auth_scheme = "API_KEY"

                    config = client.auth_configs.create(
                        toolkit=toolkit,
                        options={
                            "type": "use_custom_auth",
                            "auth_scheme": auth_scheme,
                            "credentials": custom_credentials,
                        }
                    )
                else:
                    config = client.auth_configs.create(
                        toolkit=toolkit,
                        options={"type": "use_composio_managed_auth"}
                    )

                auth_config_id = getattr(config, "id", None) or getattr(config, "auth_config_id", None)
                if auth_config_id:
                    if ag.memory:
                        ag.memory.store.save_composio_auth_config(
                            toolkit=toolkit,
                            auth_config_id=auth_config_id,
                            auth_type="custom" if custom_credentials else "managed",
                        )
                    return auth_config_id
            except Exception as e:
                logger.debug("Could not create Composio auth config for '%s': %s", toolkit, e)

            return None

        def _get_account_app(account) -> str:
            """Extract app/toolkit name from a connected account object.

            SDK 0.11+: account.toolkit is an object with .slug attribute.
            """
            return account.toolkit.slug

        def _get_account_id(account) -> str:
            """Extract connection ID from a connected account object."""
            return account.id

        def _get_account_status(account) -> str:
            """Extract status string from a connected account object.

            SDK 0.11+ returns a Literal type (e.g. "ACTIVE", "EXPIRED").
            """
            return account.status

        def _list_connected_accounts(uid: str, toolkit_slugs: list | None = None,
                                     statuses: list[str] | None = None) -> list:
            """List connected accounts for a user with optional filters."""
            kwargs: dict = {"user_ids": [uid]}
            if toolkit_slugs:
                kwargs["toolkit_slugs"] = toolkit_slugs
            if statuses:
                kwargs["statuses"] = statuses
            resp = client.connected_accounts.list(**kwargs)
            return resp.items

        def _list_apps_via_api() -> list[dict]:
            """List apps via REST API since SDK doesn't expose apps.list()."""
            import os
            import httpx

            api_key = os.environ.get("COMPOSIO_API_KEY", "")
            if not api_key:
                return []

            try:
                response = httpx.get(
                    "https://backend.composio.dev/api/v1/apps",
                    headers={"x-api-key": api_key},
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    # Handle both array response and {items: [...]} response
                    apps = data if isinstance(data, list) else data.get("items", data.get("apps", []))
                    return [
                        {
                            "name": app.get("name", ""),
                            "key": app.get("key", app.get("name", "")),
                            "description": app.get("description", ""),
                        }
                        for app in apps
                    ]
            except Exception as e:
                logger.debug("Could not list Composio apps via API: %s", e)
            return []

        # ═══════════════════════════════════════════════════════════════════
        # LIST_APPS - Show all available Composio apps
        # ═══════════════════════════════════════════════════════════════════
        if action == "list_apps":
            try:
                apps = _list_apps_via_api()
                if not apps:
                    return {
                        "error": "Could not list apps. Ensure COMPOSIO_API_KEY is set.",
                        "suggestion": "Set COMPOSIO_API_KEY environment variable or visit https://composio.dev/toolkits for available apps.",
                    }
                return {
                    "apps": apps,
                    "count": len(apps),
                }
            except Exception as e:
                return {"error": f"Failed to list apps: {e}"}

        # ═══════════════════════════════════════════════════════════════════
        # LIST_ACTIONS - Show actions for a specific app
        # ═══════════════════════════════════════════════════════════════════
        elif action == "list_actions":
            if not app:
                return {"error": "app parameter required for list_actions"}
            try:
                app_actions = client.tools.get_raw_composio_tools(toolkits=[app], limit=500)
                return {
                    "app": app,
                    "actions": [
                        {"name": getattr(a, 'slug', None) or a.name, "description": a.description}
                        for a in app_actions
                    ],
                    "count": len(app_actions),
                }
            except Exception as e:
                return {"error": f"Failed to list actions for {app}: {e}"}

        # ═══════════════════════════════════════════════════════════════════
        # GET_AUTH_PARAMS - Get authentication requirements for an app
        # ═══════════════════════════════════════════════════════════════════
        elif action == "get_auth_params":
            if not app:
                return {"error": "app parameter required for get_auth_params"}
            try:
                # Get app details via REST API
                import os
                import httpx

                api_key = os.environ.get("COMPOSIO_API_KEY", "")
                auth_schemes = None
                target_app = None

                if api_key:
                    try:
                        # Get specific app details
                        response = httpx.get(
                            f"https://backend.composio.dev/api/v1/apps/{app.upper()}",
                            headers={"x-api-key": api_key},
                            timeout=30,
                        )
                        if response.status_code == 200:
                            target_app = response.json()
                            auth_schemes = target_app.get("auth_schemes") or target_app.get("authSchemes")
                    except Exception as e:
                        logger.debug("Could not fetch auth schemes for app '%s': %s", app, e)

                if not target_app:
                    # App not found or API error - provide generic OAuth guidance
                    return {
                        "app": app,
                        "auth_type": "oauth2",
                        "expected_params": [{
                            "type": "oauth2",
                            "message": "Use connect action to initiate OAuth flow",
                            "flow": "connect"
                        }],
                        "instructions": _get_auth_instructions("oauth2", app, []),
                        "note": "Could not fetch app details. Assuming OAuth flow."
                    }

                # Get expected parameters (what credentials the user needs to provide)
                expected_params = []
                auth_type = "unknown"

                if auth_schemes:
                    # Parse auth schemes to understand what's needed
                    for scheme in auth_schemes:
                        scheme_type = scheme.get('auth_mode') or scheme.get('type') or scheme.get('mode', '')
                        scheme_type = str(scheme_type).upper()

                        if 'OAUTH' in scheme_type:
                            auth_type = "oauth2"
                            expected_params.append({
                                "type": "oauth2",
                                "message": "OAuth authentication - user will be redirected to authorize",
                                "flow": "Use connect action to get authorization URL"
                            })
                        elif 'API_KEY' in scheme_type or 'APIKEY' in scheme_type:
                            auth_type = "api_key"
                            fields = scheme.get('fields', []) or scheme.get('expected_params', [])
                            for field in fields:
                                field_name = field.get('name') or field.get('displayName', 'api_key')
                                expected_params.append({
                                    "name": field_name,
                                    "type": "api_key",
                                    "required": field.get('required', True),
                                    "description": field.get('description', f"API key for {app}")
                                })
                        elif 'BEARER' in scheme_type:
                            auth_type = "bearer_token"
                            expected_params.append({
                                "name": "token",
                                "type": "bearer_token",
                                "required": True,
                                "description": f"Bearer token for {app}"
                            })
                        elif 'BASIC' in scheme_type:
                            auth_type = "basic"
                            expected_params.extend([
                                {"name": "username", "type": "string", "required": True},
                                {"name": "password", "type": "string", "required": True, "sensitive": True}
                            ])

                # If we couldn't determine auth type, check for existing auth configs
                if not expected_params:
                    try:
                        configs = client.auth_configs.list(toolkit_slugs=[app.upper()])
                        if configs and getattr(configs, 'items', None):
                            auth_type = "has_config"
                            expected_params.append({
                                "type": "preconfigured",
                                "message": f"Auth config exists for {app}. Use connect action."
                            })
                    except Exception as e:
                        logger.debug("Could not check auth configs for app '%s': %s", app, e)

                    if not expected_params:
                        # Default: assume OAuth
                        auth_type = "oauth2"
                        expected_params.append({
                            "type": "oauth2",
                            "message": "Use connect action to initiate OAuth flow",
                            "flow": "connect"
                        })

                return {
                    "app": app,
                    "auth_type": auth_type,
                    "expected_params": expected_params,
                    "instructions": _get_auth_instructions(auth_type, app, expected_params)
                }
            except Exception as e:
                return {"error": f"Failed to get auth params for {app}: {e}"}

        # ═══════════════════════════════════════════════════════════════════
        # CHECK_CONNECTIONS - List all active connections for this user
        # ═══════════════════════════════════════════════════════════════════
        elif action == "check_connections":
            try:
                connected_accounts = _list_connected_accounts(user_id)

                connections = []
                for account in connected_accounts:
                    conn_info = {
                        "id": _get_account_id(account),
                        "app": _get_account_app(account),
                        "status": _get_account_status(account),
                        "created_at": account.created_at,
                    }
                    connections.append(conn_info)

                # Also check local database for pending connections
                local_pending = []
                if ag.memory:
                    local_conns = ag.memory.store.list_credentials(credential_type="composio_connection")
                    for cred in local_conns:
                        if cred.metadata and cred.metadata.get("status") == "pending":
                            local_pending.append({
                                "id": cred.metadata.get("connection_id"),
                                "app": cred.service,
                                "status": "pending",
                                "redirect_url": cred.metadata.get("redirect_url"),
                            })

                # Also show stored auth configs
                auth_configs = []
                if ag.memory:
                    auth_configs = ag.memory.store.list_composio_auth_configs()

                active_apps = [c["app"] for c in connections if c["status"] == "ACTIVE"]
                expired_apps = [c["app"] for c in connections if c["status"] == "EXPIRED"]

                result: dict = {
                    "user_id": user_id,
                    "connections": connections,
                    "active_apps": active_apps,
                    "pending_local": local_pending,
                    "stored_auth_configs": auth_configs,
                    "total_active": len(active_apps),
                }

                # Proactively flag expired connections and suggest refresh (Fix #4)
                if expired_apps:
                    result["expired_apps"] = expired_apps
                    result["suggestion"] = (
                        f"Expired connections found for: {', '.join(expired_apps)}. "
                        f"Use: composio_setup(action='refresh_connection', app='APP_NAME') to refresh."
                    )

                return result
            except Exception as e:
                return {"error": f"Failed to check connections: {e}"}

        # ═══════════════════════════════════════════════════════════════════
        # CONNECT - Initiate authentication for an app
        # ═══════════════════════════════════════════════════════════════════
        elif action == "connect":
            if not app:
                return {"error": "app parameter required for connect"}

            # Acquire per-app lock to prevent duplicate connections from rapid calls (Fix #1)
            lock_key = f"{user_id}:{app.upper()}"
            lock = _get_connect_lock(lock_key)
            if not lock.acquire(timeout=5):
                return {
                    "status": "in_progress",
                    "app": app,
                    "message": f"A connection attempt for {app} is already in progress. Please wait.",
                }
            try:
                # Check if already connected or has a pending connection
                if not account_label:
                    existing = _list_connected_accounts(user_id, toolkit_slugs=[app.upper()])
                    for account in existing:
                        acc_status = _get_account_status(account)
                        acc_id = _get_account_id(account)
                        if acc_status == 'ACTIVE':
                            return {
                                "status": "already_connected",
                                "app": app,
                                "connection_id": acc_id,
                                "message": f"{app} is already connected and active. You can enable tools now.",
                                "next_step": f"Use: composio_setup(action='enable', app='{app}')"
                            }
                        if acc_status in ('INITIATED', 'INITIALIZING'):
                            redirect_url = None
                            # Try to get redirect_url from local DB
                            if ag.memory:
                                local_conns = ag.memory.store.list_credentials(credential_type="composio_connection")
                                for cred in local_conns:
                                    if cred.metadata and cred.metadata.get("connection_id") == acc_id:
                                        redirect_url = cred.metadata.get("redirect_url")
                                        break
                            return {
                                "status": "pending",
                                "app": app,
                                "connection_id": acc_id,
                                "redirect_url": redirect_url,
                                "message": f"A pending connection already exists for {app}. Please complete the authorization at the URL above, then poll for completion.",
                                "next_step": f"After authorizing, use: composio_setup(action='poll_connection', app='{app}', connection_id='{acc_id}')"
                            }

                # Get or create auth_config_id
                auth_config_id = get_or_create_auth_config(app, credentials if credentials else None)

                if not auth_config_id:
                    return {
                        "error": f"Could not create auth config for {app}",
                        "suggestion": (
                            f"Create an auth config manually in the Composio dashboard at https://app.composio.dev, "
                            f"then store it with: composio_setup(action='status') to verify. "
                            f"Alternatively, ensure COMPOSIO_API_KEY is set correctly."
                        ),
                    }

                # Build initiate params — credentials go in config for API key / bearer auth
                initiate_kwargs: dict = {
                    "user_id": user_id,
                    "auth_config_id": auth_config_id,
                }
                if credentials:
                    if "api_key" in credentials:
                        initiate_kwargs["config"] = {"api_key": credentials["api_key"]}
                    elif "token" in credentials:
                        initiate_kwargs["config"] = {"token": credentials["token"]}
                    elif "username" in credentials:
                        initiate_kwargs["config"] = {
                            "username": credentials["username"],
                            "password": credentials.get("password", ""),
                        }

                try:
                    connection = client.connected_accounts.initiate(**initiate_kwargs)
                except Exception as init_error:
                    error_str = str(init_error)
                    # Handle "multiple connected accounts" — find the existing active one
                    if "multiple" in error_str.lower() and "connected account" in error_str.lower():
                        active = _list_connected_accounts(
                            user_id, toolkit_slugs=[app.upper()], statuses=["ACTIVE"]
                        )
                        if active:
                            return {
                                "status": "already_connected",
                                "app": app,
                                "connection_id": _get_account_id(active[0]),
                                "message": f"{app} is already connected and active. You can enable tools now.",
                                "next_step": f"Use: composio_setup(action='enable', app='{app}')"
                            }
                    return {
                        "error": f"Failed to initiate connection for {app}: {init_error}",
                        "suggestion": f"This app may require specific credentials. Try: composio_setup(action='get_auth_params', app='{app}')"
                    }

                # ConnectionRequest has .id, .status, .redirect_url
                conn_id = connection.id
                redirect_url = connection.redirect_url

                # For credential-based auth, connection is usually immediately active
                if credentials and connection.status == "ACTIVE":
                    if ag.memory:
                        ag.memory.store.store_credential(
                            credential_type="composio_connection",
                            service=app.upper(),
                            metadata={
                                "connection_id": conn_id,
                                "status": "active",
                                "user_id": user_id,
                                "auth_type": "api_key",
                                "account_label": account_label,
                            }
                        )
                    return {
                        "status": "connected",
                        "app": app,
                        "connection_id": conn_id,
                        "user_id": user_id,
                        "message": f"Successfully connected to {app} with provided credentials",
                        "next_step": f"Use: composio_setup(action='enable', app='{app}')"
                    }

                # OAuth flow — store pending and return redirect URL
                if ag.memory and conn_id:
                    ag.memory.store.store_credential(
                        credential_type="composio_connection",
                        service=app.upper(),
                        metadata={
                            "connection_id": conn_id,
                            "status": "pending",
                            "user_id": user_id,
                            "redirect_url": redirect_url,
                            "account_label": account_label,
                        }
                    )

                return {
                    "status": "pending",
                    "app": app,
                    "connection_id": conn_id,
                    "redirect_url": redirect_url,
                    "user_id": user_id,
                    "message": f"Please visit the URL below to authorize {app}. Once done, I can poll for completion or you can tell me when you're done.",
                    "next_step": f"After authorizing, use: composio_setup(action='poll_connection', app='{app}', connection_id='{conn_id}')"
                }
            except Exception as e:
                error_msg = str(e)
                if "auth" in error_msg.lower() or "config" in error_msg.lower():
                    return {
                        "error": f"Failed to initiate connection for {app}: {e}",
                        "suggestion": f"This app may require specific credentials. Try: composio_setup(action='get_auth_params', app='{app}')"
                    }
                return {"error": f"Failed to initiate connection for {app}: {e}"}
            finally:
                lock.release()

        # ═══════════════════════════════════════════════════════════════════
        # POLL_CONNECTION - Check if a pending OAuth connection is complete
        # ═══════════════════════════════════════════════════════════════════
        elif action == "poll_connection":
            if not app and not connection_id:
                return {"error": "app or connection_id parameter required for poll_connection"}

            try:
                # If we have connection_id, check it directly
                if connection_id:
                    account = client.connected_accounts.get(nanoid=connection_id)
                    status = account.status
                    acc_app = account.toolkit.slug if hasattr(account, 'toolkit') else (app or 'unknown')

                    if status == 'ACTIVE':
                        if ag.memory:
                            ag.memory.store.store_credential(
                                credential_type="composio_connection",
                                service=acc_app.upper(),
                                metadata={
                                    "connection_id": connection_id,
                                    "status": "active",
                                    "user_id": user_id,
                                }
                            )
                        return {
                            "status": "active",
                            "app": acc_app,
                            "connection_id": connection_id,
                            "message": "Connection is now active! You can enable tools.",
                            "next_step": f"Use: composio_setup(action='enable', app='{acc_app}')"
                        }
                    elif status in ('INITIATED', 'INITIALIZING'):
                        return {
                            "status": "pending",
                            "app": acc_app,
                            "connection_id": connection_id,
                            "message": "Still waiting for authorization. Please complete the OAuth flow.",
                        }
                    else:
                        return {
                            "status": status.lower(),
                            "app": acc_app,
                            "connection_id": connection_id,
                            "message": f"Connection status: {status}",
                        }

                # Otherwise, check by app name
                if app:
                    accounts = _list_connected_accounts(user_id, toolkit_slugs=[app.upper()])

                    for account in accounts:
                        status = _get_account_status(account)
                        conn_id = _get_account_id(account)

                        if status == 'ACTIVE':
                            if ag.memory:
                                ag.memory.store.store_credential(
                                    credential_type="composio_connection",
                                    service=app.upper(),
                                    metadata={
                                        "connection_id": conn_id,
                                        "status": "active",
                                        "user_id": user_id,
                                    }
                                )
                            return {
                                "status": "active",
                                "app": app,
                                "connection_id": conn_id,
                                "message": f"{app} is now connected! You can enable tools.",
                                "next_step": f"Use: composio_setup(action='enable', app='{app}')"
                            }
                        else:
                            return {
                                "status": status.lower(),
                                "app": app,
                                "connection_id": conn_id,
                                "message": f"Connection status: {status}",
                            }

                    return {
                        "status": "not_found",
                        "app": app,
                        "message": f"No connection found for {app}. Use connect action first.",
                    }

                return {"error": "Could not poll connection - no app or connection_id"}
            except Exception as e:
                return {"error": f"Failed to poll connection: {e}"}

        # ═══════════════════════════════════════════════════════════════════
        # REFRESH_CONNECTION - Refresh expired OAuth connections (Fix #4)
        # ═══════════════════════════════════════════════════════════════════
        elif action == "refresh_connection":
            if not app and not connection_id:
                return {"error": "app or connection_id parameter required for refresh_connection"}

            try:
                # Find the expired connection(s) to refresh
                targets = []
                if connection_id:
                    targets.append(connection_id)
                elif app:
                    expired = _list_connected_accounts(
                        user_id, toolkit_slugs=[app.upper()], statuses=["EXPIRED"]
                    )
                    targets = [_get_account_id(a) for a in expired]
                    if not targets:
                        # Check if already active
                        active = _list_connected_accounts(
                            user_id, toolkit_slugs=[app.upper()], statuses=["ACTIVE"]
                        )
                        if active:
                            return {
                                "status": "already_active",
                                "app": app,
                                "connection_id": _get_account_id(active[0]),
                                "message": f"{app} connection is already active, no refresh needed.",
                            }
                        return {
                            "status": "not_found",
                            "app": app,
                            "message": f"No expired connection found for {app}.",
                        }

                refreshed = []
                for target_id in targets:
                    resp = client.connected_accounts.refresh(nanoid=target_id)
                    new_status = getattr(resp, 'status', None)
                    redirect_url = getattr(resp, 'redirect_url', None)
                    refreshed.append({
                        "connection_id": target_id,
                        "status": new_status or "refreshing",
                        "redirect_url": redirect_url,
                    })

                # If OAuth re-auth is needed, there will be a redirect_url
                needs_reauth = [r for r in refreshed if r.get("redirect_url")]
                if needs_reauth:
                    return {
                        "status": "reauth_required",
                        "app": app or "unknown",
                        "refreshed": refreshed,
                        "message": "OAuth tokens expired. Please visit the redirect URL to re-authorize.",
                        "next_step": "After re-authorizing, use: composio_setup(action='poll_connection', ...)"
                    }

                return {
                    "status": "refreshed",
                    "app": app or "unknown",
                    "refreshed": refreshed,
                    "message": f"Refreshed {len(refreshed)} connection(s).",
                }
            except Exception as e:
                return {"error": f"Failed to refresh connection: {e}"}

        # ═══════════════════════════════════════════════════════════════════
        # DISCONNECT - Remove expired or unwanted connections
        # ═══════════════════════════════════════════════════════════════════
        elif action == "disconnect":
            if not app and not connection_id:
                return {"error": "app or connection_id parameter required for disconnect"}

            try:
                targets = []
                if connection_id:
                    targets.append(connection_id)
                elif app:
                    # Find all connections for this app (expired + active)
                    all_accounts = _list_connected_accounts(
                        user_id, toolkit_slugs=[app.upper()]
                    )
                    # Default: only delete expired connections
                    expired = [a for a in all_accounts if _get_account_status(a) == "EXPIRED"]
                    if expired:
                        targets = [_get_account_id(a) for a in expired]
                    else:
                        # If no expired, delete all connections for this app
                        targets = [_get_account_id(a) for a in all_accounts]

                    if not targets:
                        return {
                            "status": "not_found",
                            "app": app,
                            "message": f"No connections found for {app}.",
                        }

                deleted = []
                for target_id in targets:
                    try:
                        client.connected_accounts.delete(nanoid=target_id)
                        deleted.append(target_id)
                    except Exception as del_err:
                        logger.debug("Could not delete connection '%s': %s", target_id, del_err)

                return {
                    "status": "disconnected",
                    "app": app or "unknown",
                    "deleted": deleted,
                    "count": len(deleted),
                    "message": f"Removed {len(deleted)} connection(s)." + (
                        f" Use composio_setup(action='connect', app='{app}') to reconnect."
                        if app else ""
                    ),
                }
            except Exception as e:
                return {"error": f"Failed to disconnect: {e}"}

        # ═══════════════════════════════════════════════════════════════════
        # ENABLE - Register Composio actions as agent tools
        # ═══════════════════════════════════════════════════════════════════
        elif action == "enable":
            if not app:
                return {"error": "app parameter required for enable"}

            if ag.memory is None:
                return {"error": "Memory system not available"}

            try:
                # First check if the app is connected
                is_connected = False
                active_accounts = _list_connected_accounts(
                    user_id, toolkit_slugs=[app.upper()], statuses=["ACTIVE"]
                )
                is_connected = len(active_accounts) > 0

                if not is_connected:
                    return {
                        "error": f"{app} is not connected. Connect first before enabling tools.",
                        "suggestion": f"Use: composio_setup(action='connect', app='{app}')"
                    }

                # Get action schemas from Composio (use limit=500 to include
                # read/list/get actions, not just the default subset)
                if actions:
                    action_list = client.tools.get_raw_composio_tools(tools=actions)
                else:
                    action_list = client.tools.get_raw_composio_tools(toolkits=[app], limit=500)

                # Pre-populate the SDK's internal schema cache with all fetched tools.
                # This is critical: tools.execute() needs the schema in _tool_schemas
                # to avoid 404 errors from client.tools.retrieve(). (Fix #3)
                for schema in action_list:
                    slug = schema.slug
                    client.tools._tool_schemas[slug] = schema

                registered = []
                for action_schema in action_list:
                    action_name = action_schema.slug
                    tool_name = action_name.lower()

                    # Build parameters from schema
                    tool_params = getattr(action_schema, 'input_parameters', None) or getattr(action_schema, 'parameters', None) or {
                        "type": "object",
                        "properties": {},
                    }

                    # Save to our database
                    tool_def = ag.memory.store.save_tool_definition(
                        name=tool_name,
                        description=action_schema.description or f"{app} action: {action_name}",
                        parameters=tool_params,
                        tool_type="composio",
                        composio_app=app,
                        composio_action=action_name,
                        category="composio",
                        is_dynamic=True,
                    )

                    # Register with agent
                    tool = create_composio_tool(tool_def, Tool, client)
                    ag.tools[tool.name] = tool
                    registered.append(tool_name)

                return {
                    "success": True,
                    "app": app,
                    "registered": registered,
                    "count": len(registered),
                    "message": f"Enabled {len(registered)} tools from {app}. You can now use these tools directly.",
                }
            except Exception as e:
                return {"error": f"Failed to enable {app} tools: {e}"}

        # ═══════════════════════════════════════════════════════════════════
        # DISABLE - Remove Composio tools
        # ═══════════════════════════════════════════════════════════════════
        elif action == "disable":
            if not app and not actions:
                return {"error": "app or actions parameter required for disable"}

            if ag.memory is None:
                return {"error": "Memory system not available"}

            disabled = []
            if actions:
                for action_name in actions:
                    ag.memory.store.disable_tool(action_name, reason="User disabled")
                    if action_name in ag.tools:
                        del ag.tools[action_name]
                    disabled.append(action_name)
            elif app:
                composio_tools = ag.memory.store.get_composio_tools(app=app)
                for tool_def in composio_tools:
                    ag.memory.store.disable_tool(tool_def.name, reason="User disabled")
                    if tool_def.name in ag.tools:
                        del ag.tools[tool_def.name]
                    disabled.append(tool_def.name)

            return {"disabled": disabled, "count": len(disabled)}

        # ═══════════════════════════════════════════════════════════════════
        # STATUS - Show what's connected and enabled
        # ═══════════════════════════════════════════════════════════════════
        elif action == "status":
            if ag.memory is None:
                return {"error": "Memory system not available"}

            # Get local tool status
            composio_tools = ag.memory.store.get_composio_tools()
            by_app = {}
            for tool_def in composio_tools:
                app_name = tool_def.composio_app or "unknown"
                if app_name not in by_app:
                    by_app[app_name] = []
                by_app[app_name].append({
                    "name": tool_def.name,
                    "enabled": tool_def.is_enabled,
                    "usage_count": tool_def.usage_count,
                })

            # Also get connection status from Composio
            connections = []
            try:
                all_accounts = _list_connected_accounts(user_id)
                for account in all_accounts:
                    connections.append({
                        "app": _get_account_app(account),
                        "status": _get_account_status(account),
                        "id": _get_account_id(account),
                    })
            except Exception as e:
                logger.debug("Could not list Composio connections: %s", e)

            # Get stored auth configs
            auth_configs = ag.memory.store.list_composio_auth_configs()

            return {
                "user_id": user_id,
                "connections": connections,
                "enabled_apps": list(by_app.keys()),
                "tools_by_app": by_app,
                "total_tools": len(composio_tools),
                "stored_auth_configs": auth_configs,
            }

        return {"error": f"Unknown action: {action}. Valid actions: list_apps, list_actions, get_auth_params, check_connections, connect, poll_connection, refresh_connection, enable, disable, status"}

    return Tool(
        name="composio_setup",
        description="""Set up and manage Composio integrations (250+ apps via API).

This tool handles ALL Composio operations via API - no CLI or dashboard needed.
Auth configs are created programmatically and persisted across sessions.
Once connected, Composio actions become tools you can use directly.

Actions:
- list_apps: Show all 250+ available Composio apps
- list_actions: Show available actions for an app
- get_auth_params: Check what authentication an app needs (OAuth, API key, etc.)
- check_connections: List all active/pending connections and stored auth configs
- connect: Start OAuth flow OR connect with API credentials
- poll_connection: Check if pending OAuth authorization is complete
- refresh_connection: Refresh expired OAuth connections (re-auth if needed)
- enable: Register Composio actions as usable tools (requires active connection)
- disable: Remove tools from agent (keeps Composio auth)
- status: Show connections, enabled tools, and stored auth configs

Typical workflow:
1. check_connections - See what's already connected
2. get_auth_params - See what auth the app needs
3. connect - Start OAuth (gives you URL) or pass credentials directly
4. poll_connection - Verify OAuth completed (if OAuth flow)
5. enable - Activate the app's actions as tools
6. Use the tools directly!

Multiple accounts: Use account_label to connect multiple accounts for the same app.

Examples:
- composio_setup(action="check_connections")
- composio_setup(action="get_auth_params", app="ATTIO")
- composio_setup(action="connect", app="SLACK")
- composio_setup(action="connect", app="SLACK", account_label="work")  # Second Slack account
- composio_setup(action="connect", app="OPENAI", credentials={"api_key": "sk-..."})
- composio_setup(action="poll_connection", app="SLACK")
- composio_setup(action="enable", app="GITHUB")
- composio_setup(action="refresh_connection", app="GMAIL")  # Refresh expired OAuth
- composio_setup(action="disconnect", app="ATTIO")  # Remove expired connections
- composio_setup(action="enable", app="SLACK", actions=["SLACK_SEND_MESSAGE"])""",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_apps", "list_actions", "get_auth_params", "check_connections", "connect", "poll_connection", "refresh_connection", "disconnect", "enable", "disable", "status"],
                    "description": "What to do",
                },
                "app": {
                    "type": "string",
                    "description": "App name (e.g., SLACK, GITHUB, NOTION, ATTIO, GOOGLESHEETS)",
                },
                "actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific action names to enable/disable",
                },
                "credentials": {
                    "type": "object",
                    "description": "Credentials for API key/bearer token auth (e.g., {\"api_key\": \"...\"})",
                },
                "connection_id": {
                    "type": "string",
                    "description": "Connection ID for polling a specific connection",
                },
                "account_label": {
                    "type": "string",
                    "description": "Label for connecting multiple accounts on the same platform (e.g., 'work', 'personal')",
                },
            },
            "required": ["action"],
        },
        fn=fn,
    )


def _get_auth_instructions(auth_type: str, app: str, params: list) -> str:
    """Generate human-readable auth instructions based on auth type."""
    if auth_type == "oauth2":
        return f"""To connect {app}:
1. Use: composio_setup(action="connect", app="{app}")
2. You'll receive an authorization URL
3. Visit the URL and authorize the app
4. Use: composio_setup(action="poll_connection", app="{app}") to verify completion
5. Use: composio_setup(action="enable", app="{app}") to activate tools"""

    elif auth_type == "api_key":
        param_names = [p.get("name", "api_key") for p in params if p.get("type") == "api_key"]
        creds_example = ", ".join([f'"{p}": "your_{p}"' for p in param_names])
        return f"""To connect {app}:
1. Get your API key from {app}'s settings/dashboard
2. Use: composio_setup(action="connect", app="{app}", credentials={{{creds_example}}})
3. Use: composio_setup(action="enable", app="{app}") to activate tools"""

    elif auth_type == "bearer_token":
        return f"""To connect {app}:
1. Get your bearer token from {app}
2. Use: composio_setup(action="connect", app="{app}", credentials={{"token": "your_token"}})
3. Use: composio_setup(action="enable", app="{app}") to activate tools"""

    elif auth_type == "basic":
        return f"""To connect {app}:
1. Get your username and password for {app}
2. Use: composio_setup(action="connect", app="{app}", credentials={{"username": "...", "password": "..."}})
3. Use: composio_setup(action="enable", app="{app}") to activate tools"""

    else:
        return f"Use composio_setup(action='connect', app='{app}') to initiate connection."


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════


def get_skill_tools(agent: "Agent", Tool: type) -> list["Tool"]:
    """Get all skill-related tools to register with the agent."""
    return [
        _acquire_skill_tool(agent, Tool),
        _composio_setup_tool(agent, Tool),
    ]
