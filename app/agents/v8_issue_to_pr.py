from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from fastapi.testclient import TestClient


logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 40
MAX_MODEL_CALLS = 42
MAX_TOOL_OUTPUT_CHARS = 12_000
MAX_TOTAL_TOOL_OUTPUT_CHARS = 100_000
MAX_FILE_CHARS = 50_000
MAX_MODEL_RESPONSE_CHARS = 50_000
MAX_ACTION_RESPONSE_RETRIES = 2
MAX_TASK_SECONDS = 1_800
MAX_REPAIR_ATTEMPTS = 1


class ActionResponseError(ValueError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


@dataclass
class AgentResult:
    success: bool
    issue_number: int
    message: str
    selected_model: str | None = None
    decision_type: str | None = None
    routing_metadata: dict[str, Any] = field(default_factory=dict)
    branch: str | None = None
    pull_request_url: str | None = None
    test_output: str | None = None
    tool_actions: list[dict[str, Any]] = field(default_factory=list)


class V84IssueToPRAgent:
    """Model-driven issue-to-PR agent using the V8.3 router and V7 Bedrock client."""

    allowed_extensions = {
        ".cfg", ".ini", ".ipynb", ".json", ".md", ".py", ".sh",
        ".toml", ".txt", ".yaml", ".yml",
    }
    protected_paths = {
        ".github",
        ".env",
        "config",
        "dockerfile",
        "requirements.txt",
        "pyproject.toml",
        "setup.cfg",
        "tox.ini",
        "app/agents/v8_issue_to_pr.py",
        "tests/test_v8_issue_to_pr_agent.py",
        "app/config",
        "app/agents",
    }

    def __init__(
        self,
        repo_root: str,
        dry_run: bool = True,
        max_turns: int = 2,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.workspace_root = self.repo_root
        self.dry_run = dry_run
        self.max_repairs = min(max(0, max_turns - 1), MAX_REPAIR_ATTEMPTS)
        self.github_token = os.getenv("GITHUB_TOKEN", "")
        self.github_repository = os.getenv("GITHUB_REPOSITORY", "")
        self.test_command = os.getenv(
            "V84_TEST_COMMAND",
            f"{sys.executable} -m pytest -q",
        )
        self.max_tool_calls = MAX_TOOL_CALLS
        self.max_model_calls = MAX_MODEL_CALLS
        self.max_task_seconds = MAX_TASK_SECONDS
        self.started_at = 0.0
        self.tool_actions: list[dict[str, Any]] = []
        self._tool_calls = 0
        self._model_calls = 0
        self._read_paths: set[str] = set()
        self._tool_output_chars = 0

    def _check_deadline(self) -> None:
        if (
            self.started_at
            and time.monotonic() - self.started_at > self.max_task_seconds
        ):
            raise TimeoutError("V8.4 task exceeded its execution time limit.")

    def _git(
        self,
        args: list[str],
        *,
        cwd: Path | None = None,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd or self.workspace_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"git {' '.join(args)} failed ({result.returncode}): {detail}"
            )
        return result

    @contextmanager
    def isolated_workspace(self) -> Iterator[Path]:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if status.returncode:
            raise RuntimeError(
                f"Cannot inspect repository state: {status.stderr.strip()}"
            )
        if status.stdout.strip():
            raise RuntimeError(
                "Refusing to start because the source worktree has "
                "uncommitted changes."
            )

        with tempfile.TemporaryDirectory(prefix="v84-agent-") as temporary:
            workspace = Path(temporary) / "repository"
            added = False
            try:
                self._git(
                    ["worktree", "add", "--detach", str(workspace), "HEAD"],
                    cwd=self.repo_root,
                )
                added = True
                self.workspace_root = workspace
                yield workspace
            finally:
                self.workspace_root = self.repo_root
                if added:
                    self._git(
                        ["worktree", "remove", "--force", str(workspace)],
                        cwd=self.repo_root,
                    )

    @staticmethod
    def _parse_action(text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ActionResponseError("empty", "Model returned empty text.")

        candidate = text.strip()
        if candidate.startswith("```"):
            match = re.fullmatch(
                r"```(?:json)?\s*\n?(.*?)\n?```",
                candidate,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not match:
                raise ActionResponseError(
                    "malformed_json",
                    "Model returned an incomplete or invalid JSON code fence.",
                )
            candidate = match.group(1).strip()

        try:
            action = json.loads(candidate)
        except (json.JSONDecodeError, TypeError) as exc:
            category = (
                "plain_text"
                if candidate[:1] not in ("{", "[")
                else "malformed_json"
            )
            message = (
                "Model returned plain text instead of a JSON action."
                if category == "plain_text"
                else f"Model response is not valid JSON (malformed JSON): {exc}"
            )
            raise ActionResponseError(category, message) from exc

        if not isinstance(action, dict):
            raise ActionResponseError(
                "invalid_action",
                "Decoded model response must be a JSON object.",
            )
        schemas: dict[str, tuple[set[str], dict[str, type]]] = {
            "read_file": ({"action", "path"}, {"path": str}),
            "list_files": ({"action", "path"}, {"path": str}),
            "write_file": (
                {"action", "path", "content"},
                {"path": str, "content": str},
            ),
            "run_tests": ({"action", "command"}, {"command": str}),
            "finish": ({"action", "summary"}, {"summary": str}),
        }
        # Accept the equivalent function-call envelope used by some text-only
        # model responses, e.g. {"read_file": {"path": "README.md"}}.
        if "action" not in action:
            if len(action) != 1:
                raise ActionResponseError(
                    "invalid_action",
                    "Model JSON must contain one action or one named tool call.",
                )
            action_name, arguments = next(iter(action.items()))
            if action_name not in schemas or not isinstance(arguments, dict):
                raise ActionResponseError(
                    "invalid_action",
                    "Named tool call must use a supported action with object arguments.",
                )
            action = {"action": action_name, **arguments}

        name = action.get("action")
        if not isinstance(name, str) or name not in schemas:
            raise ActionResponseError(
                "invalid_action",
                "Model JSON must specify a supported action.",
            )
        required, field_types = schemas[name]
        if name == "list_files" and "path" not in action:
            action["path"] = "."
        if name == "finish" and "summary" not in action:
            action["summary"] = "Model completed the requested edits."
        if set(action) != required:
            missing = sorted(required - set(action))
            unexpected = sorted(set(action) - required)
            details = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected fields: {', '.join(unexpected)}")
            raise ActionResponseError(
                "invalid_action",
                "Invalid action fields (" + "; ".join(details) + ").",
            )
        for field_name, expected_type in field_types.items():
            if not isinstance(action[field_name], expected_type):
                raise ActionResponseError(
                    "invalid_action",
                    f"Action field '{field_name}' must be "
                    f"{expected_type.__name__}.",
                )
        if name in ("read_file", "list_files", "write_file"):
            if not action["path"]:
                raise ActionResponseError(
                    "invalid_action",
                    "Action field 'path' must be a non-empty string.",
                )
        if name == "run_tests" and not action["command"]:
            raise ActionResponseError(
                "invalid_action",
                "Action field 'command' must be a non-empty string.",
            )
        return action

    @staticmethod
    def _redact_diagnostic(value: Any) -> str:
        text = str(value)
        text = re.sub(
            r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)\S+",
            r"\1[REDACTED]",
            text,
        )
        text = re.sub(
            r"(?i)([\"']?(?:authorization|aws_access_key_id|"
            r"aws_secret_access_key|aws_session_token|access[_ -]?key|"
            r"secret[_ -]?key|api[_ -]?key|password|token)[\"']?"
            r"\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;}]+)",
            r"\1[REDACTED]",
            text,
        )
        text = re.sub(
            r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*",
            "Bearer [REDACTED]",
            text,
        )
        text = re.sub(r"\bAKIA[0-9A-Z]{16}\b", "[REDACTED_AWS_KEY]", text)
        text = re.sub(
            r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
            r"(?:\.[A-Za-z0-9_-]{10,})?\b",
            "[REDACTED_TOKEN]",
            text,
        )
        return text[:500]

    def _resolve_path(self, raw_path: Any, *, allow_root: bool = False) -> Path:
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("Path must be a non-empty relative string.")
        if "\\" in raw_path or "\x00" in raw_path:
            raise ValueError("Backslashes and NUL bytes are not allowed in paths.")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or any(part in (".", "..") for part in path.parts):
            if allow_root and raw_path == ".":
                return self.workspace_root
            raise ValueError(f"Absolute and traversal paths are forbidden: {raw_path}")
        target = (self.workspace_root / Path(*path.parts)).resolve()
        try:
            target.relative_to(self.workspace_root.resolve())
        except ValueError as exc:
            raise ValueError(f"Path escapes the workspace: {raw_path}") from exc
        return target

    @classmethod
    def _is_protected(cls, relative: str) -> bool:
        normalized = relative.casefold().strip("/")
        parts = normalized.split("/")
        if any(part in (".git", ".aws", ".ssh", "__pycache__") for part in parts):
            return True
        if any(part.startswith(".env") for part in parts):
            return True
        if any(
            word in part
            for part in parts
            for word in ("secret", "credential", "private_key")
        ):
            return True
        return any(
            normalized == protected or normalized.startswith(protected + "/")
            for protected in cls.protected_paths
        )

    def _validate_file_path(self, raw_path: Any, *, write: bool) -> tuple[Path, str]:
        target = self._resolve_path(raw_path)
        relative = target.relative_to(self.workspace_root.resolve()).as_posix()
        if self._is_protected(relative):
            raise ValueError(f"Protected path is not accessible: {relative}")
        if target.suffix.lower() not in self.allowed_extensions:
            raise ValueError(f"Unsupported text file type: {relative}")
        if write and target.suffix.lower() not in self.allowed_extensions:
            raise ValueError(f"Unsupported file type for writing: {relative}")
        return target, relative

    def read_file(self, path: str) -> str:
        target, relative = self._validate_file_path(path, write=False)
        self._read_paths.add(relative)
        if not target.is_file():
            raise FileNotFoundError(f"Repository file does not exist: {relative}")
        content = target.read_text(encoding="utf-8")
        read_limit = MAX_TOOL_OUTPUT_CHARS // 2
        if len(content) > read_limit:
            raise ValueError(
                f"{relative} exceeds the {read_limit}-character read limit."
            )
        return content

    def list_files(self, path: str = ".") -> list[str]:
        directory = self._resolve_path(path, allow_root=True)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a repository directory: {path}")
        results: list[str] = []
        for candidate in sorted(directory.rglob("*")):
            relative = candidate.relative_to(self.workspace_root).as_posix()
            if self._is_protected(relative):
                continue
            if candidate.is_file() and candidate.suffix.lower() in self.allowed_extensions:
                results.append(relative)
                if len(results) >= 1_000:
                    break
        return results

    def write_file(self, path: str, content: str) -> str:
        target, relative = self._validate_file_path(path, write=True)
        if relative not in self._read_paths:
            raise ValueError(
                f"Read {relative} with read_file before replacing its contents."
            )
        if not isinstance(content, str):
            raise ValueError("File content must be a string.")
        if len(content) > MAX_FILE_CHARS:
            raise ValueError(
                f"File content exceeds the {MAX_FILE_CHARS}-character limit."
            )
        if target.exists() and not target.is_file():
            raise ValueError(f"Not a regular file: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._read_paths.discard(relative)
        return f"Wrote {relative} ({len(content)} characters)."

    @staticmethod
    def _safe_test_command(command: Any, approved: str) -> list[str]:
        if not isinstance(command, str) or command != approved:
            raise ValueError("Only the configured approved test command is allowed.")
        if any(character in command for character in (";", "|", "&", "$", "`", ">", "<", "\n")):
            raise ValueError("Shell operators are not allowed in test commands.")
        args = shlex.split(command)
        allowed_executables = {sys.executable, "python", "python3"}
        if (
            len(args) < 4
            or args[0] not in allowed_executables
            or args[1:3] != ["-m", "pytest"]
        ):
            raise ValueError("Approved test command must invoke Python pytest.")
        return args

    def run_tests(self, command: str) -> dict[str, Any]:
        args = self._safe_test_command(command, self.test_command)
        result = subprocess.run(
            args,
            cwd=self.workspace_root,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        return {
            "returncode": result.returncode,
            "output": output[-MAX_TOOL_OUTPUT_CHARS:],
        }

    def execute_action(self, action: dict[str, Any]) -> Any:
        name = action["action"]
        if name == "read_file":
            if set(action) != {"action", "path"}:
                raise ValueError("read_file requires exactly action and path.")
            return self.read_file(action["path"])
        if name == "list_files":
            if set(action) - {"action", "path"}:
                raise ValueError("list_files accepts only action and optional path.")
            return self.list_files(action.get("path", "."))
        if name == "write_file":
            if set(action) != {"action", "path", "content"}:
                raise ValueError(
                    "write_file requires exactly action, path, and complete content."
                )
            return self.write_file(action["path"], action["content"])
        if name == "run_tests":
            if set(action) != {"action", "command"}:
                raise ValueError("run_tests requires exactly action and command.")
            return self.run_tests(action["command"])
        if name == "finish":
            if set(action) - {"action", "summary"}:
                raise ValueError("finish accepts only action and optional summary.")
            summary = action.get("summary", "Model completed the requested edits.")
            if not isinstance(summary, str):
                raise ValueError("finish summary must be a string.")
            return {"finished": True, "summary": summary}
        raise ValueError(f"Unsupported model action: {name}")

    def _route_issue(
        self, issue_number: int, issue_title: str, issue_body: str
    ) -> dict[str, Any]:
        from app.api_server import app

        prompt = (
            f"V8.4 repository coding task for issue #{issue_number}\n"
            f"Title: {issue_title}\n"
            f"Description:\n{issue_body}"
        )
        response = TestClient(app).post("/api/route", json={"prompt": prompt})
        if response.status_code != 200:
            raise RuntimeError(
                f"V8.3 /api/route failed: {response.status_code} {response.text}"
            )
        route = response.json()
        required = ("selected_model", "decision_type")
        if any(not route.get(key) for key in required):
            raise RuntimeError(
                "V8.3 /api/route response omitted selected_model or decision_type."
            )
        return route

    @staticmethod
    def _bedrock_client(selected_model: str) -> Any:
        from app.config.settings import Settings
        from app.models.registry import registry
        from app.models.v7_bedrock import V7BedrockClient

        client = V7BedrockClient(region=Settings.AWS_REGION)
        client_model_id = client.get_model_id(selected_model)
        registered_model_id = registry.resolve(selected_model)
        if client_model_id != registered_model_id:
            raise RuntimeError(
                f"Model registry/client mapping mismatch for {selected_model}."
            )
        return client

    def _model_prompt(
        self,
        issue_number: int,
        issue_title: str,
        issue_body: str,
        route: dict[str, Any],
        history: list[dict[str, Any]],
        repair: str | None = None,
        response_error: str | None = None,
    ) -> str:
        payload = {
            "issue_number": issue_number,
            "issue_title": issue_title,
            "issue_description": issue_body,
            "routing_decision": route,
            "tool_history": history,
            "verification_failure": repair,
            "previous_response_error": response_error,
            "available_actions": {
                "read_file": {"path": "relative repository path"},
                "list_files": {"path": "optional relative directory, default ."},
                "write_file": {
                    "path": "relative repository path",
                    "content": "complete replacement file contents",
                },
                "run_tests": {"command": self.test_command},
                "finish": {"summary": "brief completion summary"},
            },
        }
        return (
            "You are a repository coding agent. Use only the available JSON actions. "
            "Return exactly one JSON object per response, with no Markdown or prose. "
            "Each action must contain exactly the fields shown by its action schema. "
            "Do not return explanations, status text, or multiple actions. "
            "Read a file before replacing it, and write complete file contents. "
            "Never use shell commands except run_tests with the exact approved "
            "command. After editing, return a finish action.\n\n"
            + json.dumps(payload, ensure_ascii=False)
        )

    def _invoke_model(
        self, client: Any, selected_model: str, prompt: str
    ) -> tuple[str, dict[str, Any]]:
        self._check_deadline()
        if self._model_calls >= self.max_model_calls:
            raise RuntimeError("Model invocation limit reached.")
        self._model_calls += 1
        result = client.invoke(
            selected_model,
            prompt,
            max_tokens=4096,
            temperature=0.1,
        )
        if not isinstance(result, dict):
            raise RuntimeError(
                "Unexpected Bedrock client response shape: "
                f"expected dict, got {type(result).__name__}."
            )

        diagnostics = {
            "response_type": result.get("response_type", "str"),
            "stop_reason": result.get("stop_reason", "unknown"),
            "content_block_count": result.get("content_block_count", 0),
        }
        if not isinstance(result.get("success"), bool):
            raise RuntimeError(
                "Unexpected Bedrock client response shape: "
                "success must be bool; "
                f"response_type={diagnostics['response_type']}, "
                f"stop_reason={diagnostics['stop_reason']}, "
                f"content_block_count={diagnostics['content_block_count']}."
            )
        if result.get("response_error") == "empty_response":
            return "", diagnostics
        if result.get("success") is not True:
            raise RuntimeError(
                f"Bedrock invocation failed for {selected_model}: "
                f"{self._redact_diagnostic(result.get('error') or 'unknown error')}"
            )
        response = result.get("response")
        if not isinstance(response, str):
            raise RuntimeError(
                "Unexpected Bedrock client response shape: response must be "
                f"str, got {type(response).__name__}; "
                f"stop_reason={diagnostics['stop_reason']}, "
                f"content_block_count={diagnostics['content_block_count']}."
            )
        if len(response) > MAX_MODEL_RESPONSE_CHARS:
            raise RuntimeError(
                f"Bedrock response exceeds the {MAX_MODEL_RESPONSE_CHARS}-character limit."
            )
        self._check_deadline()
        return response, diagnostics

    def _tool_loop(
        self,
        client: Any,
        selected_model: str,
        issue_number: int,
        issue_title: str,
        issue_body: str,
        route: dict[str, Any],
        history: list[dict[str, Any]],
        repair: str | None = None,
    ) -> str:
        response_retries = 0
        response_error = None
        while True:
            prompt = self._model_prompt(
                issue_number,
                issue_title,
                issue_body,
                route,
                history,
                repair,
                response_error,
            )
            text, diagnostics = self._invoke_model(
                client, selected_model, prompt
            )
            try:
                action = self._parse_action(text)
            except ActionResponseError as exc:
                preview = self._redact_diagnostic(text) if text else "<empty>"
                logger.warning(
                    "V8.4 model action response rejected category=%s "
                    "response_type=%s stop_reason=%s content_block_count=%s "
                    "preview=%r",
                    exc.category,
                    diagnostics.get("response_type", "unknown"),
                    diagnostics.get("stop_reason", "unknown"),
                    diagnostics.get("content_block_count", 0),
                    preview,
                )
                if response_retries >= MAX_ACTION_RESPONSE_RETRIES:
                    raise RuntimeError(
                        "Model action response retries exhausted after "
                        f"{response_retries} retries: {exc.category}: {exc}; "
                        f"response_type={diagnostics.get('response_type', 'unknown')}, "
                        f"stop_reason={diagnostics.get('stop_reason', 'unknown')}, "
                        "content_block_count="
                        f"{diagnostics.get('content_block_count', 0)}, "
                        f"preview={preview!r}"
                    ) from exc
                response_retries += 1
                response_error = f"{exc.category}: {exc}"
                history.append({"model_error": response_error})
                continue

            if action["action"] == "finish":
                result = self.execute_action(action)
                return result["summary"]

            if self._tool_calls >= self.max_tool_calls:
                raise RuntimeError("Model tool-call limit reached.")
            self._tool_calls += 1
            started = time.monotonic()
            output = None
            try:
                output = self.execute_action(action)
                tool_result = {"ok": True, "result": output}
            except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
                tool_result = {"ok": False, "error": str(exc)}
            elapsed_ms = (time.monotonic() - started) * 1000
            action_record = {
                "action": action.get("action"),
                "path": action.get("path"),
                "returncode": (
                    output.get("returncode")
                    if isinstance(output, dict) and "returncode" in output
                    else None
                ),
                "duration_ms": round(elapsed_ms, 2),
                "ok": tool_result["ok"],
            }
            self.tool_actions.append(action_record)
            logger.info(
                "V8.4 issue=%s model=%s decision=%s tool=%s path=%s "
                "duration_ms=%.2f ok=%s",
                issue_number,
                selected_model,
                route.get("decision_type"),
                action.get("action"),
                action.get("path", ""),
                elapsed_ms,
                tool_result["ok"],
            )
            output_text = json.dumps(tool_result, ensure_ascii=False, default=str)
            output_text = output_text[:MAX_TOOL_OUTPUT_CHARS]
            remaining_output = (
                MAX_TOTAL_TOOL_OUTPUT_CHARS - self._tool_output_chars
            )
            if remaining_output <= 0:
                raise RuntimeError("Total model tool-output limit reached.")
            if len(output_text) > remaining_output:
                output_text = output_text[:remaining_output]
                output_text += " [total tool-output limit reached]"
            self._tool_output_chars += len(output_text)
            history.append({"action": action, "result": output_text})
            self._check_deadline()

    def changed_paths(self) -> list[str]:
        tracked = self._git(["diff", "--name-only"]).stdout.splitlines()
        untracked = self._git(
            ["ls-files", "--others", "--exclude-standard"]
        ).stdout.splitlines()
        paths = sorted(set(tracked + untracked))
        for relative in paths:
            self._validate_file_path(relative, write=True)
        return paths

    def verify_changes(self, paths: list[str]) -> str:
        if not paths:
            raise RuntimeError("Model produced no repository changes.")
        diff_check = self._git(["diff", "--check"])
        if diff_check.stdout.strip() or diff_check.stderr.strip():
            raise RuntimeError(
                "Changed files failed git diff --check:\n"
                f"{diff_check.stdout}{diff_check.stderr}"
            )
        for relative in paths:
            path = self.workspace_root / relative
            if path.exists():
                if not path.is_file():
                    raise RuntimeError(f"Changed path is not a regular file: {relative}")
                raw_content = path.read_bytes()
                if len(raw_content) > MAX_FILE_CHARS:
                    raise RuntimeError(
                        f"Changed file exceeds the {MAX_FILE_CHARS}-byte limit: "
                        f"{relative}"
                    )
                try:
                    raw_content.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise RuntimeError(
                        f"Changed file is not UTF-8 text: {relative}"
                    ) from exc
            if path.suffix.lower() == ".py":
                if not path.exists():
                    continue
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(path)],
                    cwd=self.workspace_root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                if result.returncode:
                    raise RuntimeError(
                        f"Python syntax validation failed for {relative}:\n"
                        f"{result.stderr.strip() or result.stdout.strip()}"
                    )
        test_result = self.run_tests(self.test_command)
        if test_result["returncode"]:
            raise RuntimeError(
                f"Tests failed (exit {test_result['returncode']}):\n"
                f"{test_result['output']}"
            )
        return test_result["output"]

    def _current_file_contents(self, paths: list[str]) -> str:
        sections = []
        for relative in paths:
            path = self.workspace_root / relative
            if path.is_file():
                content = path.read_text(encoding="utf-8")
                sections.append(f"FILE {relative}:\n{content[:MAX_FILE_CHARS]}")
            else:
                sections.append(f"FILE {relative}: <deleted>")
        return "\n\n".join(sections)

    def create_branch(self, issue_number: int) -> str:
        branch = f"agent/issue-{issue_number}"
        self._git(["checkout", "-b", branch])
        return branch

    def publish_changes(
        self,
        issue_number: int,
        issue_title: str,
        paths: list[str],
    ) -> tuple[str, str]:
        branch = self.create_branch(issue_number)
        self._git(["add", "--", *paths])
        self._git(["commit", "-m", f"fix: resolve issue #{issue_number}"])
        self._git(["push", "-u", "origin", branch])
        url = self.create_pull_request(
            issue_number=issue_number,
            branch=branch,
            title=f"Fix #{issue_number}: {issue_title}",
            body=(
                f"Automated fix for #{issue_number}.\n\n"
                "Generated and verified by the V8.4 Issue-to-PR Agent."
            ),
        )
        return branch, url

    def create_pull_request(
        self,
        issue_number: int,
        branch: str,
        title: str,
        body: str,
    ) -> str:
        if not self.github_token:
            raise RuntimeError("GITHUB_TOKEN is required to create a pull request.")
        if not self.github_repository:
            raise RuntimeError("GITHUB_REPOSITORY is required to create a PR.")
        url = f"https://api.github.com/repos/{self.github_repository}/pulls"
        payload = {
            "title": title,
            "head": branch,
            "base": "main",
            "body": body,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data["html_url"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"GitHub PR creation failed: {exc.code} {detail}"
            ) from exc

    def run(
        self,
        issue_number: int,
        issue_title: str,
        issue_body: str,
        repository: str | None = None,
    ) -> AgentResult:
        self.started_at = time.monotonic()
        self.tool_actions = []
        self._tool_calls = 0
        self._model_calls = 0
        self._tool_output_chars = 0
        selected_model = None
        decision_type = None
        test_output = None
        outcome = False
        try:
            route = self._route_issue(issue_number, issue_title, issue_body)
            selected_model = route["selected_model"]
            decision_type = route["decision_type"]
            bedrock = self._bedrock_client(selected_model)
            logger.info(
                "V8.4 issue=%s selected_model=%s decision=%s",
                issue_number,
                selected_model,
                decision_type,
            )

            with self.isolated_workspace():
                history: list[dict[str, Any]] = []
                summary = self._tool_loop(
                    bedrock,
                    selected_model,
                    issue_number,
                    issue_title,
                    issue_body,
                    route,
                    history,
                )
                paths = self.changed_paths()
                repair_count = 0
                while True:
                    try:
                        test_output = self.verify_changes(paths)
                        break
                    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                        if repair_count >= self.max_repairs:
                            raise
                        repair_count += 1
                        repair = (
                            f"Verification failed: {exc}\n\n"
                            "Current complete contents of changed files:\n"
                            f"{self._current_file_contents(paths)}\n\n"
                            "Make a bounded repair using the same selected model, "
                            "then finish."
                        )
                        self._tool_loop(
                            bedrock,
                            selected_model,
                            issue_number,
                            issue_title,
                            issue_body,
                            route,
                            history,
                            repair=repair,
                        )
                        paths = self.changed_paths()

                if self.dry_run:
                    outcome = True
                    return AgentResult(
                        success=True,
                        issue_number=issue_number,
                        message=f"Dry run verified: {summary}",
                        selected_model=selected_model,
                        decision_type=decision_type,
                        routing_metadata=dict(route),
                        test_output=test_output,
                        tool_actions=list(self.tool_actions),
                    )

                branch, pull_request_url = self.publish_changes(
                    issue_number, issue_title, paths
                )
                result = AgentResult(
                    success=True,
                    issue_number=issue_number,
                    message="Issue-to-PR workflow completed.",
                    selected_model=selected_model,
                    decision_type=decision_type,
                    routing_metadata=dict(route),
                    branch=branch,
                    pull_request_url=pull_request_url,
                    test_output=test_output,
                    tool_actions=list(self.tool_actions),
                )
                outcome = True
            return result
        except Exception as exc:
            logger.exception(
                "V8.4 issue=%s selected_model=%s decision=%s outcome=failed",
                issue_number,
                selected_model,
                decision_type,
            )
            return AgentResult(
                success=False,
                issue_number=issue_number,
                message=str(exc),
                selected_model=selected_model,
                decision_type=decision_type,
                routing_metadata=(dict(route) if "route" in locals() else {}),
                test_output=test_output,
                tool_actions=list(self.tool_actions),
            )
        finally:
            logger.info(
                "V8.4 issue=%s selected_model=%s decision=%s duration_ms=%.2f "
                "outcome=%s tool_actions=%s",
                issue_number,
                selected_model,
                decision_type,
                (time.monotonic() - self.started_at) * 1000,
                "success" if outcome else "failed",
                len(self.tool_actions),
            )


def main() -> None:
    repo_root = os.getenv("V84_REPO_ROOT", os.getcwd())
    issue_number = int(os.getenv("V84_ISSUE_NUMBER", "999"))
    issue_title = os.getenv("V84_ISSUE_TITLE", "Automated Issue")
    issue_body = os.getenv("V84_ISSUE_BODY", "Make the requested change.")
    event_path = os.getenv("GITHUB_EVENT_PATH")
    if event_path and Path(event_path).is_file():
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
        issue = event.get("issue", {})
        issue_number = int(issue.get("number", issue_number))
        issue_title = issue.get("title", issue_title)
        issue_body = issue.get("body") or issue_body

    agent = V84IssueToPRAgent(
        repo_root=repo_root,
        dry_run=os.getenv("V84_DRY_RUN", "true").lower() == "true",
        max_turns=int(os.getenv("V84_MAX_TURNS", "2")),
    )
    result = agent.run(issue_number, issue_title, issue_body)
    print(json.dumps(result.__dict__, indent=2, default=str))
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
