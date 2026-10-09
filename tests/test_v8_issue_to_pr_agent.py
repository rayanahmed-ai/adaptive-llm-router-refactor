from __future__ import annotations

import json
import subprocess

import pytest

from app.agents.v8_issue_to_pr import V84IssueToPRAgent


def action(**data):
    return json.dumps(data)


class FakeBedrock:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, model, prompt, **kwargs):
        self.calls.append((model, prompt))
        return {
            "success": True,
            "response": self.responses.pop(0),
        }


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text(
        "VALUE = 'old'\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "."], check=True
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "initial"],
        check=True,
    )
    return tmp_path


@pytest.fixture
def agent(repo, monkeypatch):
    monkeypatch.setenv("V84_TEST_COMMAND", "python -m pytest -q tests")
    return V84IssueToPRAgent(str(repo), dry_run=True, max_turns=1)


def test_structured_action_json_and_paths_are_validated(agent):
    with pytest.raises(ValueError, match="valid JSON"):
        agent._parse_action("{")
    with pytest.raises(ValueError, match="Absolute and traversal"):
        agent.execute_action(
            {"action": "read_file", "path": "../outside.py"}
        )
    with pytest.raises(ValueError, match="Absolute and traversal"):
        agent.execute_action(
            {"action": "read_file", "path": "/etc/passwd"}
        )
    with pytest.raises(ValueError, match="Protected path"):
        agent.execute_action(
            {"action": "write_file", "path": "config/models.yaml", "content": ""}
        )
    with pytest.raises(ValueError, match="Protected path"):
        agent.execute_action({
            "action": "write_file",
            "path": "app/agents/v8_issue_to_pr.py",
            "content": "",
        })


def test_file_reads_and_complete_writes_need_no_patch_parser(agent):
    original = agent.read_file("src/module.py")
    assert original == "VALUE = 'old'\n"
    result = agent.execute_action({
        "action": "write_file",
        "path": "src/module.py",
        "content": "VALUE = 'new'\n",
    })
    assert "Wrote src/module.py" in result
    assert agent.read_file("src/module.py") == "VALUE = 'new'\n"
    assert not hasattr(agent, "edits_to_patch")
    assert not hasattr(agent, "validate_patch")


def test_read_result_is_returned_before_model_can_request_edit(
    agent, monkeypatch
):
    bedrock = FakeBedrock([
        action(action="read_file", path="src/module.py"),
        action(action="write_file", path="src/module.py", content="VALUE = 'new'\n"),
        action(action="finish", summary="Updated module."),
    ])
    output = agent._tool_loop(
        bedrock,
        "small_model",
        12,
        "Update value",
        "Change the module constant.",
        {
            "selected_model": "small_model",
            "decision_type": "EXPLORATION",
            "final_score": 0.7,
        },
        [],
    )
    assert output == "Updated module."
    assert len(bedrock.calls) == 3
    assert all(model == "small_model" for model, _ in bedrock.calls)
    assert "VALUE = 'old'" in bedrock.calls[1][1]
    assert agent.read_file("src/module.py") == "VALUE = 'new'\n"


def test_issue_and_routing_metadata_reach_same_selected_model(agent, monkeypatch):
    bedrock = FakeBedrock([
        action(action="read_file", path="src/module.py"),
        action(action="write_file", path="src/module.py", content="VALUE = 'new'\n"),
        action(action="finish", summary="done"),
    ])
    route_args = []
    monkeypatch.setattr(
        agent,
        "_route_issue",
        lambda number, title, body: route_args.append((number, title, body))
        or {
            "selected_model": "large_model",
            "decision_type": "EXPLOITATION",
            "v75_candidate": "large_model",
            "v76_best_model": "medium_model",
            "final_score": 0.91,
        },
    )
    monkeypatch.setattr(agent, "_bedrock_client", lambda model: bedrock)
    monkeypatch.setattr(agent, "verify_changes", lambda paths: "tests passed")
    result = agent.run(44, "Issue title", "Detailed issue description")
    assert result.success
    assert route_args == [(44, "Issue title", "Detailed issue description")]
    assert result.selected_model == "large_model"
    assert result.decision_type == "EXPLOITATION"
    assert len(bedrock.calls) == 3
    assert all(model == "large_model" for model, _ in bedrock.calls)
    initial_prompt = bedrock.calls[0][1]
    assert "Issue title" in initial_prompt
    assert "Detailed issue description" in initial_prompt
    assert '"decision_type": "EXPLOITATION"' in initial_prompt


def test_forbidden_test_command_is_rejected(agent):
    with pytest.raises(ValueError, match="configured approved"):
        agent.execute_action({
            "action": "run_tests",
            "command": "python -m pytest -q tests; touch /tmp/not-allowed",
        })
    with pytest.raises(ValueError, match="Shell operators"):
        agent._safe_test_command(
            "python -m pytest -q tests; echo nope",
            "python -m pytest -q tests; echo nope",
        )


def test_failed_verification_prevents_pr_creation(agent, monkeypatch):
    bedrock = FakeBedrock([
        action(action="read_file", path="src/module.py"),
        action(action="write_file", path="src/module.py", content="VALUE = 'new'\n"),
        action(action="finish"),
    ])
    monkeypatch.setattr(
        agent,
        "_route_issue",
        lambda *_: {"selected_model": "medium_model", "decision_type": "EXPLORATION"},
    )
    monkeypatch.setattr(agent, "_bedrock_client", lambda _: bedrock)
    monkeypatch.setattr(agent, "verify_changes", lambda _: (_ for _ in ()).throw(
        RuntimeError("Tests failed: assertion error")
    ))
    published = []
    monkeypatch.setattr(agent, "publish_changes", lambda *args: published.append(args))
    agent.dry_run = False
    result = agent.run(9, "Fail", "Issue")
    assert not result.success
    assert "Tests failed: assertion error" in result.message
    assert not published


def test_repair_uses_same_selected_model_and_refreshed_contents(agent, monkeypatch):
    agent.max_repairs = 1
    bedrock = FakeBedrock([
        action(action="read_file", path="src/module.py"),
        action(action="write_file", path="src/module.py", content="VALUE = 'first'\n"),
        action(action="finish"),
        action(action="read_file", path="src/module.py"),
        action(action="write_file", path="src/module.py", content="VALUE = 'repaired'\n"),
        action(action="finish"),
    ])
    monkeypatch.setattr(
        agent,
        "_route_issue",
        lambda *_: {"selected_model": "small_model", "decision_type": "EXPLORATION"},
    )
    monkeypatch.setattr(agent, "_bedrock_client", lambda _: bedrock)
    checks = iter([
        RuntimeError("assertion failed with actual error"),
        "repaired tests passed",
    ])

    def verify(_):
        result = next(checks)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(agent, "verify_changes", verify)
    result = agent.run(21, "Repair this", "Keep the constant")
    assert result.success
    assert result.test_output == "repaired tests passed"
    assert all(model == "small_model" for model, _ in bedrock.calls)
    assert "assertion failed with actual error" in bedrock.calls[3][1]
    assert "VALUE = 'first'" in bedrock.calls[3][1]


def test_mocked_success_reaches_pr_creation_without_external_mutations(
    agent, monkeypatch
):
    bedrock = FakeBedrock([
        action(action="read_file", path="src/module.py"),
        action(action="write_file", path="src/module.py", content="VALUE = 'new'\n"),
        action(action="finish", summary="change complete"),
    ])
    monkeypatch.setattr(
        agent,
        "_route_issue",
        lambda *_: {"selected_model": "medium_model", "decision_type": "EXPLOITATION"},
    )
    monkeypatch.setattr(agent, "_bedrock_client", lambda _: bedrock)
    monkeypatch.setattr(agent, "verify_changes", lambda _: "1 passed")
    calls = []

    def fake_publish(number, title, paths):
        calls.append((number, title, paths))
        return "agent/issue-5", "https://example.invalid/pr/5"

    monkeypatch.setattr(agent, "publish_changes", fake_publish)
    agent.dry_run = False
    result = agent.run(5, "Safe change", "Update the constant")
    assert result.success
    assert result.pull_request_url == "https://example.invalid/pr/5"
    assert calls == [(5, "Safe change", ["src/module.py"])]
    assert all(model == "medium_model" for model, _ in bedrock.calls)
    # Route, Bedrock, and GitHub operations are mocked in all local agent tests.
