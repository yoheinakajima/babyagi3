# Open Source Release TODO

Items that need to be addressed before public release. Organized by priority.

> Generated from a full codebase audit. See [ARCHITECTURE.md](ARCHITECTURE.md) for system overview.

---

## Critical (Must Fix)

### Files tool not wired into agent
- **Location**: `tools/__init__.py` — `get_all_tools()` function
- **Issue**: `tools/files/__init__.py` defines a `@tool` decorated `files()` function, but the `files` module is not imported in `get_all_tools()`. The file processing, creation, and storage tools are unreachable.
- **Fix**: Add `files` to the import statement in `get_all_tools()`:
  ```python
  from tools import web, email, secrets, verbose, credentials, metrics, research_agent, meeting, files
  ```

### SendBlue listener not exported
- **Location**: `listeners/__init__.py`
- **Issue**: `run_sendblue_listener` is defined in `listeners/sendblue.py` but not exported from `listeners/__init__.py`. The other three listeners (cli, email, voice) are exported.
- **Fix**: Add to `listeners/__init__.py`:
  ```python
  from listeners.sendblue import run_sendblue_listener
  ```
  Note: `main.py` imports it directly, so this is a consistency issue rather than a runtime break.

---

## High Priority

### No test suite
- **Issue**: No `tests/` directory exists. Zero automated tests.
- **Impact**: External contributors can't verify correctness. Regressions go undetected.
- **Recommended**: Add at minimum:
  - Unit tests for `scheduler.py` (schedule parsing, task lifecycle)
  - Unit tests for `config.py` (env var substitution, defaults)
  - Unit tests for `tools/__init__.py` (schema generation, health checks)
  - Integration tests for `agent.py` (tool execution loop, thread management)
  - Unit tests for `memory/models.py` (dataclass construction)
  - Unit tests for `metrics/costs.py` (cost calculation)

### Broad exception handling (44 instances)
- **Locations**: Throughout `agent.py` (9), `tools/web.py`, `tools/research.py`, `listeners/email.py`, `memory/` modules
- **Issue**: `except Exception:` with `pass` masks errors silently. Open source users will struggle to debug.
- **Fix**: Replace with specific exception types (`except httpx.HTTPError`, `except json.JSONDecodeError`, etc.) and add `logging.warning()` or `logging.debug()` calls.

### Voice listener platform-specific code
- **Location**: `listeners/voice.py:265`
- **Issue**: Uses `afplay` (macOS only) for audio playback. Fails silently on Linux/Windows.
- **Fix**: Add platform detection:
  ```python
  import sys
  if sys.platform == "darwin":
      cmd = ["afplay", f.name]
  elif sys.platform == "linux":
      cmd = ["paplay", f.name]
  ```
  Or use a cross-platform library.

### Voice listener silence detection
- **Location**: `listeners/voice.py:103`
- **Issue**: `# TODO: Implement silence detection for automatic stop` — currently records for a fixed `max_duration` (10 seconds) regardless of when the user stops speaking.
- **Impact**: Poor UX — users wait the full duration.

### .gitignore gaps
- **Location**: `.gitignore`
- **Issues**:
  - Missing `*.db`, `*.sqlite`, `*.sqlite3` patterns (memory database)
  - Typo: `attaches_assets/` should be `attached_assets/`
- **Fix**: Add database patterns and fix typo.

---

## Medium Priority

### Workflow tool is a stub
- **Location**: `tools/skills.py:1375`
- **Issue**: `# TODO: Implement workflow logic` — the workflow creation tool is defined but the execution logic is placeholder.
- **Impact**: Users can "create" workflows but they don't actually do anything.

### File search semantic integration
- **Location**: `tools/files/__init__.py:271`
- **Issue**: `# TODO: Integrate with fact embeddings for semantic search` — file search is keyword-only, no vector search.

### Missing optional dependency group
- **Location**: `pyproject.toml`
- **Issue**: Voice listener requires `sounddevice`, `numpy`, `openai-whisper`, `pyttsx3`, `scipy` but these aren't listed as an optional dependency group.
- **Fix**: Add to `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  voice = ["sounddevice", "numpy", "openai-whisper", "pyttsx3", "scipy"]
  ```

### Unused import
- **Location**: `tools/research.py:44`
- **Issue**: `import asyncio` is imported but never used (all functions are synchronous).

### Missing LICENSE file
- **Issue**: README says "MIT" license but there's no `LICENSE` file in the repo root.
- **Fix**: Add a standard MIT LICENSE file.

---

## Low Priority

### Review pass statements (54 instances)
- **Issue**: Many `pass` statements in exception handlers throughout the codebase. Some are intentional (catching expected failures gracefully), others may be hiding bugs.
- **Recommendation**: Review each and add a comment explaining why the exception is intentionally swallowed, or add logging.

### Logging infrastructure
- **Issue**: The codebase uses `print()` and `console.*` for output but has no structured logging (`import logging`). For production use, structured logs (JSON) would be valuable.
- **Recommendation**: Add `logging.getLogger(__name__)` to key modules with configurable log levels.

### Type annotations
- **Issue**: Some functions lack return type annotations, especially in `tools/` modules.
- **Impact**: IDE support and static analysis are reduced.

### Duplicate README data model
- **Location**: Old README had data models inline. Now covered by [MODELS.md](MODELS.md).
- **Status**: Resolved in this documentation update.

---

## Nice to Have (Post-Release)

### WebSocket support
- The `server.py` has endpoints but no WebSocket for real-time event streaming.
- The `EventEmitter` system is ready for this — just needs a WebSocket subscriber.

### Multi-user support
- Currently single-owner. The memory system's `is_owner` flag is binary.
- Would need: per-user threads, per-user memory namespaces, auth on API endpoints.

### Learning system conflict resolution
- `docs/DESIGN_SELF_IMPROVEMENT.md` mentions "Learning Conflicts" as a future extension.
- Contradictory learnings (e.g., "user likes long emails" vs "user likes short emails") aren't detected.

### Learning decay
- Learnings don't lose confidence over time without reinforcement.
- Old preferences may become stale.

### Export/import for memory
- No way to export the SQLite memory database in a portable format.
- No way to import from another agent instance.

### Rate limiting on API server
- `server.py` has no rate limiting or authentication.
- Fine for local use, dangerous if exposed publicly.

### Docker/containerization
- No `Dockerfile` or `docker-compose.yml`.
- Would simplify deployment for users who don't want to manage Python environments.

---

## Checklist

- [ ] Wire `tools/files` into `tools/__init__.py`
- [ ] Export `run_sendblue_listener` from `listeners/__init__.py`
- [ ] Add test suite (`tests/`)
- [ ] Replace broad `except Exception` handlers
- [ ] Fix voice listener platform issues
- [ ] Fix .gitignore (add `*.db`, fix typo)
- [ ] Add `LICENSE` file
- [ ] Add optional dependency group for voice
- [ ] Remove unused `import asyncio` in `tools/research.py`
- [ ] Implement workflow tool logic
- [ ] Implement file search semantic integration
- [ ] Add structured logging
