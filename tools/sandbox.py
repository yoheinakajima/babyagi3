"""
Sandbox execution via e2b for code with external dependencies.

Uses AST to detect imports, filters stdlib, runs in sandbox if external packages found.
"""

import ast
import sys

# Reusable sandbox instance
_sandbox = None

# Python stdlib modules (3.12+)
STDLIB = frozenset(sys.stdlib_module_names)


def detect_imports(code: str) -> set[str]:
    """Extract external package names from code using AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split('.')[0])

    return imports - STDLIB


def get_sandbox():
    """Get or create reusable sandbox instance."""
    global _sandbox
    if _sandbox is None:
        from e2b_code_interpreter import Sandbox
        _sandbox = Sandbox()
    return _sandbox


def run_in_sandbox(code: str, packages: set[str] = None) -> dict:
    """Execute code in sandbox, auto-installing packages."""
    sandbox = get_sandbox()

    # Install packages if needed
    installed = []
    if packages:
        for pkg in packages:
            try:
                sandbox.commands.run(f"pip install -q {pkg}")
                installed.append(pkg)
            except Exception:
                pass

    # Execute code
    result = sandbox.run_code(code)

    return {
        "output": result.text if hasattr(result, 'text') else str(result),
        "packages_installed": installed,
        "sandboxed": True
    }


def execute_smart(code: str) -> dict:
    """
    Smart execution: sandbox if external packages detected, else local.

    Returns dict with output, sandboxed flag, and packages installed.
    """
    external = detect_imports(code)

    if external:
        try:
            return run_in_sandbox(code, external)
        except ImportError:
            # e2b not available, fall through to local
            pass

    # Run locally
    import io
    from contextlib import redirect_stdout, redirect_stderr

    stdout, stderr = io.StringIO(), io.StringIO()
    result = None

    try:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exec_globals = {}
            exec(code, exec_globals)
            # Get last expression value if any
            result = exec_globals.get('_result', exec_globals.get('result'))
    except Exception as e:
        return {"output": f"Error: {e}", "sandboxed": False}

    output = stdout.getvalue() or stderr.getvalue() or str(result) if result else "Done"
    return {"output": output, "sandboxed": False}
