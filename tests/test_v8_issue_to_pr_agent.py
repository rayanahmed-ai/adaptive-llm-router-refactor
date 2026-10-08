from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app.agents import v8_issue_to_pr as agent_module
from app.agents.v8_issue_to_pr import V84IssueToPRAgent


API_SOURCE = '''from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "adaptive-llm-routing",
        "version": "V8.3",
        "routing_version": "V7.7",
        "aws_region": "eu-north-1",
    }
'''

TEST_SOURCE = '''from fastapi.testclient import TestClient

from app.api_server import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200
'''

SOURCE_SOURCE = '''def helper():
    return 1
'''


def make_edits(path, operation, old=None, new=None):
    old_block = f"<old><![CDATA[{old}]]></old>" if old is not None else ""
    new_block = f"<new><![CDATA[{new}]]></new>" if new is not None else ""
    return (
        "<edits><edit>"
        f"<path>{path}</path>"
        f"<operation>{operation}</operation>"
        f"{old_block}{new_block}"
        "</edit></edits>"
    )


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "app" / "api").mkdir(parents=True)
    (tmp_path / "app" / "agents").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)

    (tmp_path / "app" / "api" / "v8_routes.py").write_text(
        API_SOURCE, encoding="utf-8"
    )
    (tmp_path / "app" / "helpers.py").write_text(
        SOURCE_SOURCE, encoding="utf-8"
    )
    (tmp_path / "tests" / "test_v8_api.py").write_text(
        TEST_SOURCE, encoding="utf-8"
    )

    return tmp_path


@pytest.fixture()
def agent(repo, monkeypatch):
    # Deterministic validation independent of git availability.
    monkeypatch.setattr(
        agent_module.shutil, "which", lambda _name: None
    )
    return V84IssueToPRAgent(repo_root=str(repo), dry_run=True)


def route(payload):
    return {
        "router": {
            "selected_model": "medium_model",
            "decision_type": "EXPLORATION",
        },
        "response": payload,
    }


# ----------------------------------------------------------------------
# Patch validation
# ----------------------------------------------------------------------

def test_valid_replacement_produces_applicable_patch(agent):
    edits = make_edits(
        "app/api/v8_routes.py",
        "replace",
        '"status": "ok",',
        '"status": "ok",\n        "status_message": "healthy",',
    )

    patch = agent.edits_to_patch(edits)

    assert "diff --git a/app/api/v8_routes.py" in patch
    assert "status_message" in patch

    paths = agent.validate_patch(patch)

    assert paths == ["app/api/v8_routes.py"]


def test_missing_old_text_is_rejected(agent):
    edits = make_edits(
        "app/api/v8_routes.py",
        "replace",
        '"version": "V9.9",',
        '"version": "V8.3",',
    )

    with pytest.raises(ValueError, match="Exact old text was not found"):
        agent.edits_to_patch(edits)


def test_ambiguous_old_text_is_rejected(agent):
    # '": "' occurs once per dict entry, so it is ambiguous here.
    edits = make_edits(
        "app/api/v8_routes.py",
        "replace",
        '": "',
        '": "ok",\n        "extra": "',
    )

    with pytest.raises(ValueError, match="unambiguous"):
        agent.edits_to_patch(edits)


def test_insert_after_requires_unique_anchor(agent):
    ambiguous = make_edits(
        "app/api/v8_routes.py",
        "insert_after",
        '": "',
        "\n# note",
    )

    with pytest.raises(ValueError, match="unambiguous"):
        agent.edits_to_patch(ambiguous)

    ok = make_edits(
        "app/api/v8_routes.py",
        "insert_after",
        '"routing_version": "V7.7",',
        "\n        \"kept\": True,",
    )

    assert "kept" in agent.edits_to_patch(ok)


def test_delete_and_create_operations(agent):
    delete_edits = make_edits(
        "app/helpers.py", "delete", new=""
    )

    patch = agent.edits_to_patch(delete_edits)

    assert "-def helper():" in patch
    assert not [
        line for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]

    create_edits = make_edits(
        "app/new_module.py",
        "create",
        new="VALUE = 2\n",
    )

    patch = agent.edits_to_patch(create_edits)

    assert "new_module.py" in patch

    duplicate = make_edits(
        "app/helpers.py",
        "create",
        new="VALUE = 2\n",
    )

    with pytest.raises(ValueError, match="Cannot create existing file"):
        agent.edits_to_patch(duplicate)


def test_noop_patch_is_rejected(agent):
    edits = make_edits(
        "app/api/v8_routes.py",
        "replace",
        '"status": "ok",',
        '"status": "ok",',
    )

    with pytest.raises(ValueError, match="no change"):
        agent.edits_to_patch(edits)


def test_malformed_xml_is_rejected(agent):
    with pytest.raises(ValueError):
        agent.extract_edits("no xml here")

    with pytest.raises(ValueError, match="Invalid XML edit response"):
        agent.extract_edits("<edits><edit><path>x</edit></edits>")

    with pytest.raises(ValueError, match="missing <path>"):
        agent.extract_edits(
            "<edits><edit><operation>replace</operation></edit></edits>"
        )

    with pytest.raises(ValueError, match="no XML edits"):
        agent.extract_edits("<edits></edits>")


# ----------------------------------------------------------------------
# API version protection
# ----------------------------------------------------------------------

def test_api_version_change_rejected_when_not_requested(agent):
    edits = make_edits(
        "app/api/v8_routes.py",
        "replace",
        '"version": "V8.3",',
        '"version": "V8.4",',
    )

    patch = agent.edits_to_patch(edits)

    with pytest.raises(ValueError, match="does not request a version change"):
        agent.validate_patch(patch, issue_body="Add a status message.")


def test_api_version_change_allowed_when_explicitly_requested(agent):
    edits = make_edits(
        "app/api/v8_routes.py",
        "replace",
        '"version": "V8.3",',
        '"version": "V8.4",',
    )

    patch = agent.edits_to_patch(edits)

    paths = agent.validate_patch(
        patch,
        issue_body="Bump the API version to V8.4.",
    )

    assert paths == ["app/api/v8_routes.py"]


def test_routing_version_change_rejected(agent):
    edits = make_edits(
        "app/api/v8_routes.py",
        "replace",
        '"routing_version": "V7.7",',
        '"routing_version": "V8.0",',
    )

    with pytest.raises(ValueError, match="version change"):
        agent.validate_patch(
            agent.edits_to_patch(edits),
            issue_body="Add a status message.",
        )


def test_adding_a_field_without_touching_version_is_allowed(agent):
    edits = make_edits(
        "app/api/v8_routes.py",
        "replace",
        '"status": "ok",',
        '"status": "ok",\n        "status_message": "Adaptive LLM Router is healthy",',
    )

    patch = agent.edits_to_patch(edits)

    assert '"version": "V8.3"' in (agent.repo_root / "app/api/v8_routes.py").read_text()
    assert agent.validate_patch(patch, issue_body="Add status_message.")


# ----------------------------------------------------------------------
# Test architecture protection
# ----------------------------------------------------------------------

def test_module_level_test_client_is_respected(agent):
    good = make_edits(
        "tests/test_v8_api.py",
        "replace",
        "    assert response.status_code == 200",
        "    assert response.status_code == 200\n    assert response.json()[\"status\"] == \"ok\"",
    )

    patch = agent.edits_to_patch(good)

    assert agent.validate_patch(patch, issue_body="Update the test.")

    invented = make_edits(
        "tests/test_v8_api.py",
        "append",
        new=(
            "\n\ndef test_new_health_status(test_client):\n"
            "    assert test_client is not None\n"
        ),
    )

    with pytest.raises(ValueError, match="module-level TestClient"):
        agent.validate_patch(
            agent.edits_to_patch(invented),
            issue_body="Update the test.",
        )


def test_unknown_pytest_fixture_is_rejected(agent):
    invented = make_edits(
        "tests/test_v8_api.py",
        "append",
        new=(
            "\n\ndef test_something(arbitrary_fixture):\n"
            "    assert arbitrary_fixture is not None\n"
        ),
    )

    with pytest.raises(ValueError, match="does not exist in the repository"):
        agent.validate_patch(
            agent.edits_to_patch(invented),
            issue_body="Update the test.",
        )


@pytest.mark.parametrize(
    "text",
    [
        "Add a test",
        "Add tests",
        "Update a test",
        "Update tests",
        "Update the test",
        "Update the relevant test",
        "Update the relevant pytest test",
        "pytest test",
        "pytest tests",
        "Write a test",
        "Write tests",
        "Include a test",
        "Include tests",
    ],
)
def test_issue_requires_tests_detects_normal_wording(text):
    assert V84IssueToPRAgent.issue_requires_tests(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "No change needed.",
        "Refactor the router internals.",
        "No tests required.",
    ],
)
def test_issue_requires_tests_ignores_non_test_issues(text):
    assert V84IssueToPRAgent.issue_requires_tests(text) is False


# ----------------------------------------------------------------------
# Repair loop: STATE A (validation failed, patch NOT applied)
# ----------------------------------------------------------------------

def test_validation_failure_repair_uses_current_repository_state(
    repo, agent, monkeypatch
):
    source_file = repo / "app" / "api" / "v8_routes.py"

    bad_edits = make_edits(
        "app/api/v8_routes.py",
        "replace",
        '"status": "STALE-MARKER",',
        '"status": "ok",',
    )

    good_edits = make_edits(
        "app/api/v8_routes.py",
        "replace",
        '"status": "ok",',
        '"status": "ok",\n        "status_message": "healthy",',
    )

    prompts = []
    responses = [bad_edits, good_edits]

    def fake_generate(prompt):
        prompts.append(prompt)
        return route(responses.pop(0))

    monkeypatch.setattr(agent, "generate_model_response", fake_generate)

    result = agent.run(
        issue_number=101,
        issue_title="Add status message",
        issue_body="Add a status_message field.",
    )

    assert result.success is True
    assert len(prompts) == 2
    assert "The previous patch was NOT applied." in prompts[1]
    assert "Exact old text was not found" in prompts[1]
    assert "status_message" in prompts[1]
    assert source_file.read_text(encoding="utf-8").count(
        "status_message"
    ) == 0


def test_repair_prompt_never_reuses_stale_source(repo, agent, monkeypatch):
    source_file = repo / "app" / "api" / "v8_routes.py"

    stale_edits = make_edits(
        "app/api/v8_routes.py",
        "replace",
        '"status": "STALE-MARKER",',
        '"status": "ok",',
    )

    prompts = []
    state = {"first": True}

    def fake_generate(prompt):
        prompts.append(prompt)
        if state["first"]:
            state["first"] = False
            # Simulate the repository changing underneath the agent.
            source_file.write_text(
                API_SOURCE + '\n\nMARKER = "CURRENT-STATE"\n',
                encoding="utf-8",
            )
            return route(stale_edits)
        return route(
            make_edits(
                "app/api/v8_routes.py",
                "replace",
                "MARKER = \"CURRENT-STATE\"",
                "MARKER = \"UPDATED-STATE\"",
            )
        )

    monkeypatch.setattr(agent, "generate_model_response", fake_generate)

    result = agent.run(
        issue_number=102,
        issue_title="Refresh state",
        issue_body="Update the helper constant.",
    )

    assert result.success is True
    repository_section = prompts[1].split(
        "REPOSITORY CONTEXT (CURRENT STATE):"
    )[1].split("EXISTING TEST FILES:")[0]
    assert "CURRENT-STATE" in repository_section
    assert "STALE-MARKER" not in repository_section
    assert 'MARKER = "UPDATED-STATE"' in result.patch


# ----------------------------------------------------------------------
# Repair loop: STATE B (patch applied, syntax/tests failed)
# ----------------------------------------------------------------------

def test_syntax_failure_is_rejected_before_application(
    repo, agent, monkeypatch
):
    module_file = repo / "app" / "helpers.py"

    broken_edits = make_edits(
        "app/helpers.py",
        "replace",
        "def helper():\n    return 1\n",
        "def helper(:\n    return 1\n",
    )

    prompts = []
    apply_called = []

    monkeypatch.setattr(
        agent,
        "generate_model_response",
        lambda prompt: prompts.append(prompt)
        or route(broken_edits),
    )
    monkeypatch.setattr(
        agent,
        "apply_patch",
        lambda patch: apply_called.append(patch),
    )

    result = agent.run(
        issue_number=103,
        issue_title="Break helper",
        issue_body="Refactor the helper.",
    )

    assert result.success is False
    assert apply_called == []
    assert "not valid Python" in result.message
    assert "The previous patch was NOT applied." in prompts[1]
    assert module_file.read_text(encoding="utf-8") == SOURCE_SOURCE


def test_post_apply_syntax_validation_enters_applied_state_repair(
    repo, agent, monkeypatch
):
    """
    validate_python_syntax() remains a real post-application guard. When
    it fires, the repair prompt must describe an APPLIED patch and carry
    the current working tree.
    """

    module_file = repo / "app" / "helpers.py"

    # Turn 1 produces a patch that applies, but the resulting file on
    # disk is not valid Python (simulating a partially applied state).
    turn1 = make_edits(
        "app/helpers.py",
        "replace",
        "def helper():\n    return 1\n",
        "def helper():\n    return 2\n",
    )

    # Turn 2 targets the CURRENT (broken) file content.
    turn2 = make_edits(
        "app/helpers.py",
        "replace",
        "def helper(:\n    return 2\n",
        "def helper():\n    return 2\n",
    )

    prompts = []
    responses = [turn1, turn2]

    def fake_generate(prompt):
        prompts.append(prompt)
        return route(responses.pop(0))

    monkeypatch.setattr(agent, "generate_model_response", fake_generate)
    monkeypatch.setattr(agent, "dry_run", False)
    monkeypatch.setattr(
        agent, "validate_patch", lambda patch, issue_body="": ["app/helpers.py"]
    )
    monkeypatch.setattr(
        agent, "create_branch", lambda issue_number: "agent/issue-103"
    )
    monkeypatch.setattr(agent, "git_status", lambda: " M app/helpers.py")
    monkeypatch.setattr(
        agent, "git_diff", lambda: "diff --git a/app/helpers.py"
    )
    def fake_apply(patch):
        module_file.write_text(
            "def helper():\n    return 2\n"
            if "-def helper(:" in patch
            else "def helper(:\n    return 2\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(agent, "apply_patch", fake_apply)
    monkeypatch.setattr(
        agent,
        "run_tests",
        lambda changed_paths=None: (_ for _ in ()).throw(
            RuntimeError("Tests failed:\nstill broken")
        ),
    )

    result = agent.run(
        issue_number=103,
        issue_title="Break helper",
        issue_body="Refactor the helper.",
    )

    assert result.success is False
    assert len(prompts) == 2
    assert "The previous patch HAS already been applied." in prompts[1]
    assert "CURRENT FILE app/helpers.py" in prompts[1]
    assert "syntax validation failed" in prompts[1]
    assert "def helper(:\n    return 2\n" in prompts[1]
    assert "still broken" in result.message
    assert result.patch


def test_applied_state_repair_can_complete_the_workflow(
    repo, agent, monkeypatch
):
    module_file = repo / "app" / "helpers.py"

    # Turn 1: patch applies, tests then fail.
    turn1 = make_edits(
        "app/helpers.py",
        "replace",
        "def helper():\n    return 1\n",
        "def helper():\n    return 2\n",
    )

    # Turn 2: repair against the CURRENT working tree (return 2).
    turn2 = make_edits(
        "app/helpers.py",
        "replace",
        "def helper():\n    return 2\n",
        "def helper():\n    return 3\n",
    )

    prompts = []
    responses = [turn1, turn2]

    def fake_generate(prompt):
        prompts.append(prompt)
        return route(responses.pop(0))

    class GitResult:
        returncode = 0
        stdout = ""
        stderr = ""

    calls = {"tests": 0}

    def fake_tests(changed_paths=None):
        calls["tests"] += 1
        if calls["tests"] == 1:
            raise RuntimeError("Tests failed:\nassert 2 == 1")
        return "1 passed"

    monkeypatch.setattr(agent, "generate_model_response", fake_generate)
    monkeypatch.setattr(agent, "dry_run", False)
    monkeypatch.setattr(
        agent, "validate_patch", lambda patch, issue_body="": ["app/helpers.py"]
    )
    monkeypatch.setattr(
        agent, "create_branch", lambda issue_number: "agent/issue-104"
    )
    monkeypatch.setattr(agent, "git_status", lambda: " M app/helpers.py")
    monkeypatch.setattr(
        agent, "git_diff", lambda: "diff --git a/app/helpers.py"
    )
    written = []

    def fake_apply(patch):
        content = (
            "def helper():\n    return 3\n"
            if "+    return 3" in patch
            else "def helper():\n    return 2\n"
        )
        written.append(content)
        module_file.write_text(content, encoding="utf-8")

    monkeypatch.setattr(agent, "apply_patch", fake_apply)
    monkeypatch.setattr(agent, "run_tests", fake_tests)
    monkeypatch.setattr(
        agent, "run_git", lambda args, check=True: GitResult()
    )
    monkeypatch.setattr(
        agent,
        "create_pull_request",
        lambda **kwargs: "https://example.invalid/pr/1",
    )

    result = agent.run(
        issue_number=104,
        issue_title="Change helper",
        issue_body="Refactor the helper.",
    )

    assert result.success is True
    assert len(prompts) == 2
    assert "The previous patch HAS already been applied." in prompts[1]
    assert result.pull_request_url == "https://example.invalid/pr/1"
    assert result.test_output == "1 passed"
    assert result.branch == "agent/issue-104"
    assert written[-1] == "def helper():\n    return 3\n"


def test_test_failure_repair_prompt_contains_pytest_output(
    repo, agent, monkeypatch
):
    source_file = repo / "app" / "helpers.py"

    edits = make_edits(
        "app/helpers.py",
        "replace",
        "return 1",
        "return 2",
    )

    prompts = []

    def fake_generate(prompt):
        prompts.append(prompt)
        return route(edits)

    def fake_apply(patch):
        source_file.write_text(
            SOURCE_SOURCE.replace("return 1", "return 2"),
            encoding="utf-8",
        )

    monkeypatch.setattr(agent, "generate_model_response", fake_generate)
    monkeypatch.setattr(agent, "dry_run", False)
    monkeypatch.setattr(
        agent, "validate_patch", lambda patch, issue_body="": ["app/helpers.py"]
    )
    monkeypatch.setattr(
        agent, "create_branch", lambda issue_number: "agent/issue-1"
    )
    monkeypatch.setattr(agent, "git_status", lambda: " M app/helpers.py")
    monkeypatch.setattr(
        agent, "git_diff", lambda: "CURRENT-DIFF-MARKER"
    )
    monkeypatch.setattr(agent, "apply_patch", fake_apply)
    monkeypatch.setattr(
        agent,
        "run_tests",
        lambda changed_paths=None: (_ for _ in ()).throw(
            RuntimeError("Tests failed:\nassert 2 == 1")
        ),
    )

    result = agent.run(
        issue_number=104,
        issue_title="Failing change",
        issue_body="Change the helper return value.",
    )

    assert result.success is False
    assert len(prompts) == 2
    assert "The previous patch HAS already been applied." in prompts[1]
    assert "assert 2 == 1" in prompts[1]
    assert "CURRENT-DIFF-MARKER" in prompts[1]


def test_dry_run_never_touches_working_tree(repo, agent, monkeypatch):
    edits = make_edits(
        "app/api/v8_routes.py",
        "replace",
        '"status": "ok",',
        '"status": "ok",\n        "status_message": "healthy",',
    )

    monkeypatch.setattr(
        agent,
        "generate_model_response",
        lambda prompt: route(edits),
    )

    result = agent.run(
        issue_number=105,
        issue_title="Add status message",
        issue_body="Add a status_message field.",
    )

    assert result.success is True
    assert "status_message" not in (
        repo / "app" / "api" / "v8_routes.py"
    ).read_text(encoding="utf-8")
    assert not list(repo.glob(".v84_*.patch"))


def test_max_turns_is_respected_and_reports_failure(repo, agent, monkeypatch):
    bad = make_edits(
        "app/api/v8_routes.py",
        "replace",
        '"not-present": True,',
        '"ok": True,',
    )

    calls = []

    def fake_generate(prompt):
        calls.append(prompt)
        return route(bad)

    monkeypatch.setattr(agent, "generate_model_response", fake_generate)

    agent.max_turns = 3
    result = agent.run(
        issue_number=106,
        issue_title="Never matches",
        issue_body="No test needed.",
    )

    assert len(calls) == 3
    assert result.success is False
    assert result.issue_number == 106
    assert result.selected_model == "medium_model"
    assert result.decision_type == "EXPLORATION"
    assert "Exact old text was not found" in result.message
    assert result.patch


# ----------------------------------------------------------------------
# Git diagnostics
# ----------------------------------------------------------------------

def test_git_errors_expose_stderr(repo, agent, monkeypatch):
    class Result:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a git repository"

    monkeypatch.setattr(
        agent_module.subprocess, "run", lambda *a, **k: Result()
    )

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        agent.run_git(["status"])

    assert excinfo.value.returncode == 128
    assert "fatal: not a git repository" in str(excinfo.value.stderr)
    assert excinfo.value.cmd == ["git", "status"]


def test_git_result_exposes_stdout_stderr_and_code(repo, agent, monkeypatch):
    class Result:
        returncode = 0
        stdout = "clean"
        stderr = ""

    monkeypatch.setattr(
        agent_module.subprocess, "run", lambda *a, **k: Result()
    )

    result = agent.run_git(["status"], check=False)

    assert result.returncode == 0
    assert result.stdout == "clean"
    assert result.stderr == ""


# ----------------------------------------------------------------------
# Adaptive routing is preserved
# ----------------------------------------------------------------------

def _agent_source() -> str:
    return Path(agent_module.__file__).read_text(encoding="utf-8")


def test_agent_still_uses_the_adaptive_routing_path():
    source = _agent_source()

    assert '"/api/route"' in source
    assert "V77FinalAdaptiveRouter" not in source
    assert "selected_model" in source
    assert "decision_type" in source


def test_agent_does_not_hardcode_a_bedrock_model():
    source = _agent_source().lower()

    for forbidden in ("nova-micro", "nova-lite", "nova-pro", "anthropic.", "us.anthropic"):
        assert forbidden not in source


def test_agent_routes_through_v83_api_and_router_context(agent, monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "version": "V8.3",
                "selected_model": "small_model",
                "decision_type": "EXPLORATION",
            }

    class FakeClient:
        def __init__(self, _app):
            captured["constructed"] = True

        def post(self, url, json=None):
            captured["url"] = url
            captured["prompt"] = json["prompt"]
            return FakeResponse()

    monkeypatch.setattr(agent_module, "TestClient", FakeClient)

    context = agent.get_router_context()

    assert captured["url"] == "/api/route"
    assert context["selected_model"] == "small_model"
    assert context["version"] == "V8.3"


def test_existing_agent_contract_is_preserved(repo):
    agent = V84IssueToPRAgent(repo_root=str(repo), dry_run=True)
    assert agent.max_turns >= 1

    result_fields = {
        "branch",
        "decision_type",
        "issue_number",
        "message",
        "patch",
        "pull_request_url",
        "selected_model",
        "success",
        "test_output",
    }

    assert result_fields.issubset(
        set(agent_module.AgentResult.__dataclass_fields__)
    )
