"""
Skills and Composio integration tools.

This module provides tools for:
1. Acquiring skills from SKILL.md files (behavioral instructions)
2. Setting up Composio integrations (external app connections)
3. Creating workflow tools that combine skills and other tools

Tool Types:
- "executable": Python code that runs directly (default)
- "skill": Returns behavioral instructions when called
- "composio": Thin wrapper that calls Composio library
"""

import re
import yaml
from typing import TYPE_CHECKING

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
They're different from executable tools - skills teach patterns and workflows.

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


def create_composio_tool(tool_def: "ToolDefinition", Tool: type, composio_client) -> "Tool":
    """
    Create a Tool instance from a composio-type ToolDefinition.

    The tool wraps Composio's execute_action, passing through to their library.
    Compatible with both old SDK (entity_id) and new SDK (user_id).
    """

    def composio_fn(params: dict, agent: "Agent") -> dict:
        try:
            user_id = agent.config.get("composio_entity_id", "default")

            # SDK 0.11+: client.tools.execute(slug, arguments, user_id=...)
            try:
                result = composio_client.tools.execute(
                    slug=tool_def.composio_action,
                    arguments=params,
                    user_id=user_id,
                )
            except (TypeError, AttributeError):
                # Fallback for older SDK: client.execute_action(...)
                try:
                    result = composio_client.execute_action(
                        action=tool_def.composio_action,
                        params=params,
                        user_id=user_id,
                    )
                except TypeError:
                    result = composio_client.execute_action(
                        action=tool_def.composio_action,
                        params=params,
                        entity_id=user_id,
                    )
            return result
        except Exception as e:
            return {"error": f"Composio execution failed: {e}"}

    return Tool(
        name=tool_def.name,
        description=tool_def.description,
        parameters=tool_def.parameters,
        fn=composio_fn,
        packages=["composio"],
    )


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
            """Get existing auth_config_id or create one programmatically.

            For OAuth apps, uses Composio's managed OAuth (no client credentials needed).
            For API key apps, creates a custom auth config with provided credentials.
            """
            import os
            import httpx

            toolkit = toolkit_name.upper()

            # Check if we have a stored auth config
            if ag.memory:
                stored = ag.memory.store.get_composio_auth_config(toolkit)
                if stored:
                    return stored["auth_config_id"]

            # Try SDK method first
            try:
                auth_configs = getattr(client, "auth_configs", None)
                if auth_configs:
                    if custom_credentials:
                        if "api_key" in custom_credentials:
                            auth_scheme = "API_KEY"
                        elif "token" in custom_credentials:
                            auth_scheme = "BEARER_TOKEN"
                        elif "username" in custom_credentials and "password" in custom_credentials:
                            auth_scheme = "BASIC"
                        else:
                            auth_scheme = "API_KEY"

                        config = auth_configs.create(
                            toolkit=toolkit,
                            options={
                                "type": "use_custom_auth",
                                "auth_scheme": auth_scheme,
                                "credentials": custom_credentials,
                            }
                        )
                    else:
                        config = auth_configs.create(
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
            except Exception:
                pass

            # Fallback: Try REST API to create auth config
            api_key = os.environ.get("COMPOSIO_API_KEY", "")
            if api_key:
                try:
                    payload = {
                        "toolkit": toolkit,
                        "type": "use_composio_managed_auth" if not custom_credentials else "use_custom_auth",
                    }
                    if custom_credentials:
                        if "api_key" in custom_credentials:
                            payload["auth_scheme"] = "API_KEY"
                        elif "token" in custom_credentials:
                            payload["auth_scheme"] = "BEARER_TOKEN"
                        payload["credentials"] = custom_credentials

                    response = httpx.post(
                        "https://backend.composio.dev/api/v1/auth-configs",
                        headers={"x-api-key": api_key, "Content-Type": "application/json"},
                        json=payload,
                        timeout=30,
                    )
                    if response.status_code in (200, 201):
                        data = response.json()
                        auth_config_id = data.get("id") or data.get("auth_config_id")
                        if auth_config_id and ag.memory:
                            ag.memory.store.save_composio_auth_config(
                                toolkit=toolkit,
                                auth_config_id=auth_config_id,
                                auth_type="custom" if custom_credentials else "managed",
                            )
                        return auth_config_id
                except Exception:
                    pass

            return None

        def _get_account_app(account) -> str:
            """Extract app/toolkit name from a connected account object."""
            # SDK 0.11+: account.toolkit is an object with .slug
            toolkit = getattr(account, 'toolkit', None)
            if toolkit is not None and hasattr(toolkit, 'slug'):
                return toolkit.slug
            # Fallback: toolkit might be a string directly
            if isinstance(toolkit, str):
                return toolkit
            # Older SDK attribute names
            return getattr(account, 'app_name', None) or getattr(account, 'appName', '') or ''

        def _get_account_id(account) -> str:
            """Extract connection ID from a connected account object."""
            return getattr(account, 'id', None) or getattr(account, 'connectedAccountId', None) or getattr(account, 'connection_id', 'unknown')

        def _list_connected_accounts(uid: str, toolkit_slugs: list | None = None) -> list:
            """Helper to list connected accounts with SDK compatibility."""
            try:
                kwargs = {"user_ids": [uid]}
                if toolkit_slugs:
                    kwargs["toolkit_slugs"] = toolkit_slugs
                resp = client.connected_accounts.list(**kwargs)
                return resp.items if hasattr(resp, 'items') else list(resp)
            except TypeError:
                try:
                    resp = client.connected_accounts.list(user_id=uid)
                    return resp.items if hasattr(resp, 'items') else list(resp)
                except TypeError:
                    resp = client.connected_accounts.list(entity_ids=[uid])
                    return resp.items if hasattr(resp, 'items') else list(resp)

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
            except Exception:
                pass
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
                app_actions = client.tools.get_raw_composio_tools(toolkits=[app])
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
                    except Exception:
                        pass

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

                # If we couldn't determine auth type, try connecting to see what happens
                if not expected_params:
                    try:
                        # Try to get auth config info
                        auth_configs = getattr(client, 'auth_configs', None)
                        if auth_configs:
                            configs = auth_configs.list(toolkit=app)
                            if configs:
                                auth_type = "has_config"
                                expected_params.append({
                                    "type": "preconfigured",
                                    "message": f"Auth config exists for {app}. Use connect action."
                                })
                    except Exception:
                        pass

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
                        "status": getattr(account, 'status', 'unknown'),
                        "created_at": str(getattr(account, 'created_at', None) or getattr(account, 'createdAt', '')),
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

                active_apps = [c["app"] for c in connections if c["status"] in ("ACTIVE", "active")]

                return {
                    "user_id": user_id,
                    "connections": connections,
                    "active_apps": active_apps,
                    "pending_local": local_pending,
                    "stored_auth_configs": auth_configs,
                    "total_active": len(active_apps),
                }
            except Exception as e:
                return {"error": f"Failed to check connections: {e}"}

        # ═══════════════════════════════════════════════════════════════════
        # CONNECT - Initiate authentication for an app
        # ═══════════════════════════════════════════════════════════════════
        elif action == "connect":
            if not app:
                return {"error": "app parameter required for connect"}
            try:
                # Check if already connected or has a pending connection
                if not account_label:
                    try:
                        connected_accounts = _list_connected_accounts(user_id, toolkit_slugs=[app.upper()])
                        for account in connected_accounts:
                            acc_app = _get_account_app(account)
                            acc_status = str(getattr(account, 'status', '')).upper()
                            acc_id = _get_account_id(account)
                            if acc_app.upper() == app.upper():
                                if acc_status == 'ACTIVE':
                                    return {
                                        "status": "already_connected",
                                        "app": app,
                                        "connection_id": acc_id,
                                        "message": f"{app} is already connected and active. You can enable tools now.",
                                        "next_step": f"Use: composio_setup(action='enable', app='{app}')"
                                    }
                                if acc_status in ('INITIATED', 'PENDING', 'INITIALIZING'):
                                    # Reuse existing pending connection instead of creating a duplicate
                                    redirect_url = getattr(account, 'redirect_url', None) or getattr(account, 'redirectUrl', None)
                                    # Try to get redirect_url from local DB if not on the account object
                                    if not redirect_url and ag.memory:
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
                    except Exception:
                        pass  # Continue with connection attempt

                # Get or create auth_config_id (required for SDK 0.10+)
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

                # If credentials provided (for API key / bearer token auth)
                if credentials:
                    try:
                        initiate_params = {
                            "user_id": user_id,
                            "auth_config_id": auth_config_id,
                        }
                        # Pass credentials as config
                        if "api_key" in credentials:
                            initiate_params["config"] = {"api_key": credentials["api_key"]}
                        elif "token" in credentials:
                            initiate_params["config"] = {"token": credentials["token"]}
                        elif "username" in credentials:
                            initiate_params["config"] = {
                                "username": credentials["username"],
                                "password": credentials.get("password", ""),
                            }

                        connection = client.connected_accounts.initiate(**initiate_params)
                        conn_id = getattr(connection, 'id', None) or getattr(connection, 'connectedAccountId', None) or getattr(connection, 'connectionId', None)

                        # Store connection locally
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
                    except Exception as e:
                        return {
                            "error": f"Failed to connect with provided credentials: {e}",
                            "suggestion": "Check that credentials are correct. For API key auth, use: credentials={\"api_key\": \"your-key\"}"
                        }

                # OAuth flow - initiate connection with auth_config_id
                try:
                    connection = client.connected_accounts.initiate(
                        user_id=user_id,
                        auth_config_id=auth_config_id,
                    )
                except Exception as init_error:
                    error_str = str(init_error)
                    # Handle "multiple connected accounts" error by finding existing connection
                    if "multiple" in error_str.lower() and "connected account" in error_str.lower():
                        try:
                            existing = _list_connected_accounts(user_id, toolkit_slugs=[app.upper()])
                            for account in existing:
                                acc_status = str(getattr(account, 'status', '')).upper()
                                if acc_status == 'ACTIVE':
                                    return {
                                        "status": "already_connected",
                                        "app": app,
                                        "connection_id": _get_account_id(account),
                                        "message": f"{app} is already connected and active. You can enable tools now.",
                                        "next_step": f"Use: composio_setup(action='enable', app='{app}')"
                                    }
                        except Exception:
                            pass
                    return {
                        "error": f"Failed to initiate OAuth for {app}: {init_error}",
                        "suggestion": f"This app may require specific credentials. Try: composio_setup(action='get_auth_params', app='{app}')"
                    }

                # Extract connection info (handle both old and new response formats)
                conn_id = getattr(connection, 'id', None) or getattr(connection, 'connectedAccountId', None) or getattr(connection, 'connectionId', None)
                redirect_url = getattr(connection, 'redirectUrl', None) or getattr(connection, 'redirect_url', None)

                # Store pending connection in local database for tracking
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
                # Check if this is an auth config issue
                if "auth" in error_msg.lower() or "config" in error_msg.lower():
                    return {
                        "error": f"Failed to initiate connection for {app}: {e}",
                        "suggestion": f"This app may require specific credentials. Try: composio_setup(action='get_auth_params', app='{app}')"
                    }
                return {"error": f"Failed to initiate connection for {app}: {e}"}

        # ═══════════════════════════════════════════════════════════════════
        # POLL_CONNECTION - Check if a pending OAuth connection is complete
        # ═══════════════════════════════════════════════════════════════════
        elif action == "poll_connection":
            if not app and not connection_id:
                return {"error": "app or connection_id parameter required for poll_connection"}

            try:
                # If we have connection_id, check it directly
                if connection_id:
                    try:
                        account = client.connected_accounts.get(nanoid=connection_id)
                        status = str(getattr(account, 'status', 'unknown')).upper()
                        acc_app = _get_account_app(account) or app or 'unknown'

                        if status == 'ACTIVE':
                            # Update local record
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
                        elif status in ('PENDING', 'INITIATED', 'INITIALIZING'):
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
                    except Exception as e:
                        return {"error": f"Failed to check connection {connection_id}: {e}"}

                # Otherwise, check by app name
                if app:
                    connected_accounts = _list_connected_accounts(user_id, toolkit_slugs=[app.upper()])

                    for account in connected_accounts:
                        acc_app = _get_account_app(account)
                        if acc_app.upper() == app.upper():
                            status = str(getattr(account, 'status', 'unknown')).upper()
                            conn_id = _get_account_id(account)

                            if status == 'ACTIVE':
                                # Update local record
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
                try:
                    connected_accounts = _list_connected_accounts(user_id, toolkit_slugs=[app.upper()])
                    for account in connected_accounts:
                        acc_app = _get_account_app(account)
                        acc_status = str(getattr(account, 'status', '')).upper()
                        if acc_app.upper() == app.upper() and acc_status == 'ACTIVE':
                            is_connected = True
                            break
                except Exception:
                    pass

                if not is_connected:
                    return {
                        "error": f"{app} is not connected. Connect first before enabling tools.",
                        "suggestion": f"Use: composio_setup(action='connect', app='{app}')"
                    }

                # Get action schemas from Composio
                if actions:
                    action_list = client.tools.get_raw_composio_tools(tools=actions)
                else:
                    action_list = client.tools.get_raw_composio_tools(toolkits=[app])

                registered = []
                for action_schema in action_list:
                    # Create tool name - use slug (e.g. "ATTIO_LIST_CONTACTS") as canonical name
                    tool_name = (getattr(action_schema, 'slug', None) or action_schema.name).lower()

                    # Build parameters from schema
                    tool_params = getattr(action_schema, 'input_parameters', None) or getattr(action_schema, 'parameters', None) or {
                        "type": "object",
                        "properties": {},
                    }

                    # Canonical action name (slug preferred, fallback to name)
                    action_name = getattr(action_schema, 'slug', None) or action_schema.name

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
                connected_accounts = _list_connected_accounts(user_id)
                for account in connected_accounts:
                    connections.append({
                        "app": _get_account_app(account),
                        "status": str(getattr(account, 'status', 'unknown')),
                        "id": _get_account_id(account),
                    })
            except Exception:
                pass

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

        return {"error": f"Unknown action: {action}. Valid actions: list_apps, list_actions, get_auth_params, check_connections, connect, poll_connection, enable, disable, status"}

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
- composio_setup(action="enable", app="SLACK", actions=["SLACK_SEND_MESSAGE"])""",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_apps", "list_actions", "get_auth_params", "check_connections", "connect", "poll_connection", "enable", "disable", "status"],
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
# WORKFLOW TOOLS
# ═══════════════════════════════════════════════════════════════════════════════


def _create_workflow_tool(agent: "Agent", Tool: type) -> "Tool":
    """
    Create a tool for building workflow tools that combine skills and other tools.
    """

    def fn(params: dict, ag: "Agent") -> dict:
        name = params["name"]
        description = params["description"]
        depends_on = params.get("depends_on", [])
        steps = params.get("steps", [])

        if ag.memory is None:
            return {"error": "Memory system not available"}

        # Verify dependencies exist
        missing = [d for d in depends_on if d not in ag.tools]
        if missing:
            return {
                "error": f"Missing dependencies: {missing}",
                "suggestion": "Enable these tools first, then create the workflow.",
            }

        # Generate workflow code
        code = _generate_workflow_code(name, description, depends_on, steps)

        # Save as executable tool
        tool_def = ag.memory.store.save_tool_definition(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input for the workflow",
                    }
                },
            },
            source_code=code,
            tool_type="executable",
            depends_on=depends_on,
            category="workflow",
            is_dynamic=True,
        )

        return {
            "success": True,
            "name": name,
            "depends_on": depends_on,
            "message": f"Workflow '{name}' created. Note: Complex workflows may need manual implementation.",
        }

    return Tool(
        name="create_workflow",
        description="""Create a workflow tool that orchestrates other tools and skills.

Workflows combine multiple tools/skills into a single reusable tool.
This is useful for multi-step processes that you do repeatedly.

Example:
create_workflow(
    name="daily_standup",
    description="Post daily standup to Slack",
    depends_on=["skill_standup_format", "slack_send_message"],
    steps=["Activate standup skill", "Format update", "Post to #standup"]
)""",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the workflow tool",
                },
                "description": {
                    "type": "string",
                    "description": "What this workflow does",
                },
                "depends_on": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tool/skill names this workflow uses",
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "High-level steps the workflow performs",
                },
            },
            "required": ["name", "description"],
        },
        fn=fn,
    )


def _generate_workflow_code(name: str, description: str, depends_on: list, steps: list) -> str:
    """Generate Python code for a workflow tool."""
    steps_comment = "\n".join(f"    # {i+1}. {step}" for i, step in enumerate(steps))
    depends_comment = ", ".join(depends_on) if depends_on else "none"

    return f'''
def workflow_fn(params: dict, agent) -> dict:
    """
    {description}

    Dependencies: {depends_comment}
    Steps:
{steps_comment}
    """
    input_data = params.get("input", "")

    # TODO: Implement workflow logic
    # Access dependent tools via agent.tools[tool_name].execute(params, agent)

    return {{
        "workflow": "{name}",
        "status": "not_implemented",
        "message": "This workflow needs manual implementation. Use the steps above as a guide.",
        "depends_on": {depends_on!r},
    }}

{name}_tool = Tool(
    name="{name}",
    description="""{description}""",
    parameters={{
        "type": "object",
        "properties": {{
            "input": {{"type": "string", "description": "Input for the workflow"}}
        }}
    }},
    fn=workflow_fn
)
'''


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════


def get_skill_tools(agent: "Agent", Tool: type) -> list["Tool"]:
    """Get all skill-related tools to register with the agent."""
    return [
        _acquire_skill_tool(agent, Tool),
        _composio_setup_tool(agent, Tool),
        _create_workflow_tool(agent, Tool),
    ]
