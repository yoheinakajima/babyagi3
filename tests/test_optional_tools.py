from tools import _registered_tools
from tools.optional import load_optional_tools


def test_optional_tools_not_loaded_without_keys(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    before = len(_registered_tools)
    loaded = load_optional_tools()
    after = len(_registered_tools)

    assert loaded == [] or all("github_api" not in m for m in loaded)
    assert after >= before


def test_optional_tools_load_with_key(monkeypatch):
    # Reset loader state by reloading module to keep test deterministic
    import importlib
    import tools.optional as opt

    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    opt = importlib.reload(opt)

    before = len(_registered_tools)
    loaded = opt.load_optional_tools()
    after = len(_registered_tools)

    assert "tools.optional.github_api" in loaded
    assert after >= before
    assert any(t["name"] == "github_search_repositories" for t in _registered_tools)


def test_optional_tools_can_load_later_after_key_is_added(monkeypatch):
    import importlib
    import tools.optional as opt

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    opt = importlib.reload(opt)

    first = opt.load_optional_tools()
    assert "tools.optional.github_api" not in first

    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    second = opt.load_optional_tools()
    assert "tools.optional.github_api" in second
