from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from app.agents.v8_issue_to_pr import (
    ActionResponseError,
    V84IssueToPRAgent,
)
from app.models.v7_bedrock import V7BedrockClient


def action(**data):
    return json.dumps(data)


class FakeBedrock:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, model, prompt, **kwargs):
        self.calls.append((model, prompt))
        if not self.responses:
            raise AssertionError("unexpected additional Bedrock invocation")
        return {
            "success": True,
            "response": self.responses.pop(0),
            "response_type": "dict",
            "stop_reason": "end_turn",
            "content_block_count": 1,
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


def test_valid_json_action_and_fenced_json_action_are_accepted(agent):
    plain = agent._parse_action(
        '{"action":"read_file","path":"src/module.py"}'
    )
    named_tool = agent._parse_action(
        '{"read_file":{"path":"README.md"}}'
    )
    wrapped_action = agent._parse_action(
        '{"action":{"action":"write_file","path":"README.md",'
        '"content":"HELLO WORLD"}}'
    )
    fenced = agent._parse_action(
        '```json\n{"action":"finish","summary":"done"}\n```'
    )
    assert plain == {"action": "read_file", "path": "src/module.py"}
    assert named_tool == {"action": "read_file", "path": "README.md"}
    assert wrapped_action == {
        "action": "write_file",
        "path": "README.md",
        "content": "HELLO WORLD",
    }
    assert fenced == {"action": "finish", "summary": "done"}


def test_named_tool_call_executes_through_normal_validation(agent):
    response = '{"read_file":{"path":"src/module.py"}}'
    parsed = agent._parse_action(response)
    assert agent.execute_action(parsed) == "VALUE = 'old'\n"

    forbidden = agent._parse_action(
        '{"read_file":{"path":"../outside.py"}}'
    )
    with pytest.raises(ValueError, match="traversal paths"):
        agent.execute_action(forbidden)


def test_named_tool_call_rejects_multiple_or_unknown_tools(agent):
    with pytest.raises(ActionResponseError, match="one action"):
        agent._parse_action(
            '{"read_file":{"path":"README.md"},"write_file":{}}'
        )
    with pytest.raises(ActionResponseError, match="supported action"):
        agent._parse_action('{"shell":{"command":"anything"}}')


def test_nested_action_wrapper_executes_as_a_normal_validated_action(agent):
    parsed = agent._parse_action(
        '{"action":{"action":"read_file","path":"src/module.py"}}'
    )
    assert parsed == {"action": "read_file", "path": "src/module.py"}
    assert agent.execute_action(parsed) == "VALUE = 'old'\n"

    with pytest.raises(ActionResponseError, match="missing fields: content"):
        agent._parse_action(
            '{"action":{"action":"write_file","path":"README.md"}}'
        )


@pytest.mark.parametrize(
    ("response", "category"),
    [
        ("", "empty"),
        ("I made the change.", "plain_text"),
        ('{"action":', "malformed_json"),
    ],
)
def test_empty_plain_text_and_malformed_json_have_distinct_categories(
    agent, response, category
):
    with pytest.raises(ActionResponseError) as error:
        agent._parse_action(response)
    assert error.value.category == category


def test_action_schema_requires_all_fields_and_rejects_unknown_fields(agent):
    with pytest.raises(ActionResponseError, match="missing fields: path"):
        agent._parse_action('{"action":"read_file"}')
    with pytest.raises(ActionResponseError, match="unexpected fields"):
        agent._parse_action(
            '{"action":"read_file","path":"src/module.py","extra":true}'
        )


def test_bedrock_converse_shape_is_extracted_by_existing_client():
    bedrock = V7BedrockClient.__new__(V7BedrockClient)
    bedrock.model_ids = {"small_model": "eu.amazon.nova-micro-v1:0"}
    bedrock.client = SimpleNamespace(
        converse=lambda **_: {
            "output": {
                "message": {
                    "content": [
                        {"text": '{"action":"finish"}'},
                    ],
                },
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 4, "outputTokens": 3, "totalTokens": 7},
            "metrics": {"latencyMs": 12},
        }
    )
    result = bedrock.invoke("small_model", "return json")
    assert result["success"] is True
    assert result["response"] == '{"action":"finish"}'
    assert result["response_type"] == "dict"
    assert result["stop_reason"] == "end_turn"
    assert result["content_block_count"] == 1


def test_empty_bedrock_converse_content_preserves_diagnostics():
    bedrock = V7BedrockClient.__new__(V7BedrockClient)
    bedrock.model_ids = {"small_model": "eu.amazon.nova-micro-v1:0"}
    bedrock.client = SimpleNamespace(
        converse=lambda **_: {
            "output": {"message": {"content": []}},
            "stopReason": "max_tokens",
        }
    )
    result = bedrock.invoke("small_model", "return json")
    assert result["success"] is False
    assert result["response"] == ""
    assert result["response_error"] == "empty_response"
    assert result["stop_reason"] == "max_tokens"
    assert result["content_block_count"] == 0


def test_unexpected_bedrock_response_shape_is_not_executed(agent):
    class UnexpectedShape:
        def invoke(self, *_args, **_kwargs):
            return ["not", "a", "response"]

    with pytest.raises(RuntimeError, match="Unexpected Bedrock client response shape"):
        agent._invoke_model(UnexpectedShape(), "small_model", "prompt")


def test_invocation_errors_are_not_treated_as_model_text(agent):
    class InvocationFailure:
        def invoke(self, *_args, **_kwargs):
            return {
                "success": False,
                "error": "Authorization: Bearer sensitive-token-value",
            }

    with pytest.raises(RuntimeError, match="Bedrock invocation failed") as error:
        agent._invoke_model(InvocationFailure(), "small_model", "prompt")
    assert "sensitive-token-value" not in str(error.value)
    assert "[REDACTED]" in str(error.value)
    assert "json-secret" not in agent._redact_diagnostic(
        '{"token":"json-secret","authorization":"Bearer json-bearer-secret"}'
    )


def test_invalid_response_logs_safe_bedrock_diagnostics(agent, caplog):
    bedrock = FakeBedrock([
        "plain text token=super-secret-value",
        '{"action":"finish","summary":"done"}',
    ])
    history = []
    agent._tool_loop(
        bedrock,
        "medium_model",
        77,
        "Safe diagnostics",
        "Do not log secrets.",
        {
            "selected_model": "medium_model",
            "decision_type": "EXPLOITATION",
        },
        history,
    )
    assert "category=plain_text" in caplog.text
    assert "response_type=dict" in caplog.text
    assert "stop_reason=end_turn" in caplog.text
    assert "content_block_count=1" in caplog.text
    assert "super-secret-value" not in caplog.text


@pytest.mark.parametrize(
    ("response", "category"),
    [
        ("", "empty"),
        ("This is plain text, not an action.", "plain_text"),
        ('{"action":', "malformed_json"),
    ],
)
def test_invalid_responses_retry_twice_then_stop_without_changes(
    agent, response, category
):
    bedrock = FakeBedrock([response, response, response])
    history = []
    with pytest.raises(RuntimeError, match=f"retries exhausted.*{category}"):
        agent._tool_loop(
            bedrock,
            "large_model",
            78,
            "Test response handling",
            "No repository edit should occur.",
            {"selected_model": "large_model", "decision_type": "EXPLORATION"},
            history,
        )
    assert len(bedrock.calls) == 3
    assert all(model == "large_model" for model, _ in bedrock.calls)
    assert "previous_response_error" in bedrock.calls[1][1]
    assert "previous_response_error" in bedrock.calls[2][1]
    assert not agent.changed_paths()


def test_invalid_path_action_is_rejected_without_modification(agent):
    bedrock = FakeBedrock([
        '{"action":"read_file","path":"../outside.py"}',
        '{"action":"finish","summary":"Stopped safely."}',
    ])
    history = []
    summary = agent._tool_loop(
        bedrock,
        "medium_model",
        79,
        "Reject traversal",
        "Do not access outside the workspace.",
        {"selected_model": "medium_model", "decision_type": "EXPLOITATION"},
        history,
    )
    assert summary == "Stopped safely."
    assert json.loads(history[0]["result"])["ok"] is False
    assert "traversal paths" in history[0]["result"]
    assert not agent.changed_paths()


def test_retry_exhaustion_prevents_changes_and_pr_creation(agent, monkeypatch):
    monkeypatch.setattr(
        agent,
        "_route_issue",
        lambda *_: {
            "selected_model": "small_model",
            "decision_type": "EXPLORATION",
        },
    )
    bedrock = FakeBedrock(["not-json"] * 3)
    monkeypatch.setattr(agent, "_bedrock_client", lambda _model: bedrock)
    published = []
    monkeypatch.setattr(
        agent,
        "publish_changes",
        lambda *args: published.append(args),
    )
    agent.dry_run = False
    result = agent.run(80, "Malformed model", "Do not edit files.")
    assert result.success is False
    assert "retries exhausted" in result.message
    assert result.selected_model == "small_model"
    assert len(bedrock.calls) == 3
    assert all(model == "small_model" for model, _ in bedrock.calls)
    assert not published
    assert not agent.changed_paths()


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
