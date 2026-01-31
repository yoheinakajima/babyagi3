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
    """

    def composio_fn(params: dict, agent: "Agent") -> dict:
        try:
            entity_id = agent.config.get("composio_entity_id", "default")
            result = composio_client.execute_action(
                action=tool_def.composio_action,
                params=params,
                entity_id=entity_id,
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
    """

    def fn(params: dict, ag: "Agent") -> dict:
        action = params["action"]
        app = params.get("app")
        actions = params.get("actions")

        # Import composio
        try:
            from composio import Composio
            from composio.client.exceptions import ComposioClientError
        except ImportError:
            return {
                "error": "Composio not installed. Run: pip install composio-core",
                "suggestion": "Install composio with: pip install composio-core",
            }

        try:
            client = Composio()
        except Exception as e:
            return {
                "error": f"Failed to initialize Composio client: {e}",
                "suggestion": "Make sure COMPOSIO_API_KEY is set or run: composio login",
            }

        if action == "list_apps":
            try:
                apps = client.apps.list()
                return {
                    "apps": [{"name": a.name, "description": a.description} for a in apps],
                    "count": len(apps),
                }
            except Exception as e:
                return {"error": f"Failed to list apps: {e}"}

        elif action == "list_actions":
            if not app:
                return {"error": "app parameter required for list_actions"}
            try:
                app_actions = client.actions.list(app=app)
                return {
                    "app": app,
                    "actions": [
                        {"name": a.name, "description": a.description}
                        for a in app_actions
                    ],
                    "count": len(app_actions),
                }
            except Exception as e:
                return {"error": f"Failed to list actions for {app}: {e}"}

        elif action == "connect":
            if not app:
                return {"error": "app parameter required for connect"}
            try:
                entity_id = ag.config.get("composio_entity_id", "default")
                connection = client.connected_accounts.initiate(
                    app_name=app,
                    entity_id=entity_id,
                )
                return {
                    "status": "pending",
                    "app": app,
                    "redirect_url": connection.redirectUrl,
                    "message": f"Visit the URL to authorize {app}",
                }
            except Exception as e:
                return {"error": f"Failed to initiate connection for {app}: {e}"}

        elif action == "enable":
            if not app:
                return {"error": "app parameter required for enable"}

            if ag.memory is None:
                return {"error": "Memory system not available"}

            try:
                # Get action schemas from Composio
                if actions:
                    action_list = [client.actions.get(action=a) for a in actions]
                else:
                    action_list = client.actions.list(app=app)

                registered = []
                for action_schema in action_list:
                    # Create tool name
                    tool_name = action_schema.name.lower().replace("_", "_")

                    # Build parameters from schema
                    tool_params = action_schema.parameters or {
                        "type": "object",
                        "properties": {},
                    }

                    # Save to our database
                    tool_def = ag.memory.store.save_tool_definition(
                        name=tool_name,
                        description=action_schema.description or f"{app} action: {action_schema.name}",
                        parameters=tool_params,
                        tool_type="composio",
                        composio_app=app,
                        composio_action=action_schema.name,
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
                    "message": f"Enabled {len(registered)} tools from {app}",
                }
            except Exception as e:
                return {"error": f"Failed to enable {app} tools: {e}"}

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

        elif action == "status":
            if ag.memory is None:
                return {"error": "Memory system not available"}

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

            return {
                "connected_apps": list(by_app.keys()),
                "tools_by_app": by_app,
                "total_tools": len(composio_tools),
            }

        return {"error": f"Unknown action: {action}"}

    return Tool(
        name="composio_setup",
        description="""Set up and manage Composio integrations.

Composio provides 250+ app integrations (Slack, GitHub, Notion, etc.) via OAuth.
Once enabled, Composio actions become tools in your toolkit.

Actions:
- list_apps: Show all available Composio apps
- list_actions: Show actions for a specific app
- connect: Start OAuth flow to authorize an app
- enable: Register Composio actions as tools
- disable: Remove tools (keeps Composio auth)
- status: Show what's connected and enabled

Examples:
- composio_setup(action="list_apps")
- composio_setup(action="connect", app="SLACK")
- composio_setup(action="enable", app="GITHUB")
- composio_setup(action="enable", app="SLACK", actions=["SLACK_SEND_MESSAGE"])""",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_apps", "list_actions", "connect", "enable", "disable", "status"],
                    "description": "What to do",
                },
                "app": {
                    "type": "string",
                    "description": "App name (e.g., SLACK, GITHUB, NOTION)",
                },
                "actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific action names to enable/disable",
                },
            },
            "required": ["action"],
        },
        fn=fn,
    )


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
