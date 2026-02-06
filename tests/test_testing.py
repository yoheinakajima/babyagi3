"""
Tests for tools/testing.py (run_tests tool).
"""

import os
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from tools.testing import run_tests


class TestRunTests:
    """Test the run_tests tool function."""

    def test_run_all_tests_returns_structured_result(self):
        """Running tests should return pass/fail counts."""
        # Target a single small file to avoid recursive test execution and
        # timeout in CI (running the full suite spawns a nested pytest that
        # includes this file, causing a >120s timeout).
        result = run_tests._tool_info["fn"]({"path": "tests/test_utils.py", "keyword": "", "max_failures": 0}, agent=None)
        assert "success" in result
        assert "passed" in result
        assert "failed" in result
        assert isinstance(result["passed"], int)

    def test_run_specific_file(self):
        """Can target a specific test file."""
        result = run_tests._tool_info["fn"]({"path": "tests/test_utils.py", "keyword": "", "max_failures": 0}, agent=None)
        assert "success" in result
        assert result["passed"] >= 0

    def test_keyword_filter(self):
        """Keyword filter narrows which tests run."""
        result = run_tests._tool_info["fn"]({"path": "", "keyword": "test_datetime_serialization", "max_failures": 0}, agent=None)
        assert "success" in result
        # Should match at most a few tests
        assert result["passed"] <= 5

    def test_max_failures(self):
        """max_failures parameter is accepted without error."""
        result = run_tests._tool_info["fn"]({"path": "tests/test_agent.py", "keyword": "", "max_failures": 1}, agent=None)
        assert "success" in result

    @patch("tools.testing.subprocess.run")
    def test_timeout_handling(self, mock_run):
        """Timeout is reported cleanly."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=120)
        result = run_tests._tool_info["fn"]({"path": "", "keyword": "", "max_failures": 0}, agent=None)
        assert result["success"] is False
        assert "timed out" in result["error"]

    @patch("tools.testing.subprocess.run")
    def test_pytest_not_found(self, mock_run):
        """Missing pytest is reported cleanly."""
        mock_run.side_effect = FileNotFoundError()
        result = run_tests._tool_info["fn"]({"path": "", "keyword": "", "max_failures": 0}, agent=None)
        assert result["success"] is False
        assert "pytest not found" in result["error"]

    @patch("tools.testing.subprocess.run")
    def test_parses_summary_line(self, mock_run):
        """Correctly parses pytest summary output."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="FAILED tests/test_foo.py::test_bar\n10 passed, 2 failed in 3.45s\n",
            stderr="",
        )
        result = run_tests._tool_info["fn"]({"path": "", "keyword": "", "max_failures": 0}, agent=None)
        assert result["passed"] == 10
        assert result["failed"] == 2
        assert result["success"] is False
        assert any("FAILED" in f for f in result["failures"])

    @patch("tools.testing.subprocess.run")
    def test_output_truncation(self, mock_run):
        """Large output is truncated to last 3000 chars."""
        big_output = "x" * 5000
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=big_output,
            stderr="",
        )
        result = run_tests._tool_info["fn"]({"path": "", "keyword": "", "max_failures": 0}, agent=None)
        assert len(result["output"]) <= 3000
