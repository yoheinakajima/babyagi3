"""
Sandbox Execution via e2b

Provides isolated code execution with automatic package installation.
Uses AST to detect imports and installs them in the sandbox.
"""

import ast

# Lazy-loaded sandbox instance (reused across calls)
_sandbox = None


def detect_imports(code: str) -> list[str]:
    """
    Use AST to detect all imports in Python code.

    Returns list of top-level package names (e.g., 'requests' not 'requests.auth').
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Get top-level package (e.g., 'os.path' -> 'os')
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])

    # Filter out standard library modules
    stdlib = {
        'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio',
        'asyncore', 'atexit', 'audioop', 'base64', 'bdb', 'binascii',
        'binhex', 'bisect', 'builtins', 'bz2', 'calendar', 'cgi', 'cgitb',
        'chunk', 'cmath', 'cmd', 'code', 'codecs', 'codeop', 'collections',
        'colorsys', 'compileall', 'concurrent', 'configparser', 'contextlib',
        'contextvars', 'copy', 'copyreg', 'cProfile', 'crypt', 'csv',
        'ctypes', 'curses', 'dataclasses', 'datetime', 'dbm', 'decimal',
        'difflib', 'dis', 'distutils', 'doctest', 'email', 'encodings',
        'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput',
        'fnmatch', 'fractions', 'ftplib', 'functools', 'gc', 'getopt',
        'getpass', 'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib',
        'heapq', 'hmac', 'html', 'http', 'idlelib', 'imaplib', 'imghdr',
        'imp', 'importlib', 'inspect', 'io', 'ipaddress', 'itertools',
        'json', 'keyword', 'lib2to3', 'linecache', 'locale', 'logging',
        'lzma', 'mailbox', 'mailcap', 'marshal', 'math', 'mimetypes',
        'mmap', 'modulefinder', 'multiprocessing', 'netrc', 'nis', 'nntplib',
        'numbers', 'operator', 'optparse', 'os', 'ossaudiodev', 'pathlib',
        'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform',
        'plistlib', 'poplib', 'posix', 'posixpath', 'pprint', 'profile',
        'pstats', 'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue',
        'quopri', 'random', 're', 'readline', 'reprlib', 'resource',
        'rlcompleter', 'runpy', 'sched', 'secrets', 'select', 'selectors',
        'shelve', 'shlex', 'shutil', 'signal', 'site', 'smtpd', 'smtplib',
        'sndhdr', 'socket', 'socketserver', 'spwd', 'sqlite3', 'ssl',
        'stat', 'statistics', 'string', 'stringprep', 'struct', 'subprocess',
        'sunau', 'symtable', 'sys', 'sysconfig', 'syslog', 'tabnanny',
        'tarfile', 'telnetlib', 'tempfile', 'termios', 'test', 'textwrap',
        'threading', 'time', 'timeit', 'tkinter', 'token', 'tokenize',
        'tomllib', 'trace', 'traceback', 'tracemalloc', 'tty', 'turtle',
        'turtledemo', 'types', 'typing', 'unicodedata', 'unittest', 'urllib',
        'uu', 'uuid', 'venv', 'warnings', 'wave', 'weakref', 'webbrowser',
        'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml', 'xmlrpc',
        'zipapp', 'zipfile', 'zipimport', 'zlib', '_thread'
    }

    return [pkg for pkg in imports if pkg not in stdlib]


def get_sandbox():
    """Get or create a reusable sandbox instance."""
    global _sandbox
    if _sandbox is None:
        from e2b_code_interpreter import Sandbox
        _sandbox = Sandbox()
    return _sandbox


def run_in_sandbox(code: str, packages: list[str] = None) -> dict:
    """
    Execute code in an isolated e2b sandbox.

    Args:
        code: Python code to execute
        packages: Optional list of packages to install first

    Returns:
        dict with 'result' or 'error' key
    """
    sandbox = get_sandbox()

    # Install packages if specified
    if packages:
        for pkg in packages:
            sandbox.commands.run(f"pip install -q {pkg}")

    # Execute the code
    execution = sandbox.run_code(code)

    if execution.error:
        return {"error": str(execution.error)}

    # Return the last result or stdout
    if execution.results:
        last = execution.results[-1]
        return {"result": last.text if hasattr(last, 'text') else str(last)}

    return {"result": execution.logs.stdout if execution.logs else None}


def close_sandbox():
    """Close the sandbox when done."""
    global _sandbox
    if _sandbox is not None:
        _sandbox.close()
        _sandbox = None
