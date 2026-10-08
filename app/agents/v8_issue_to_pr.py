from __future__ import annotations

import json
import difflib
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient


@dataclass
class AgentResult:
    success: bool
    issue_number: int
    message: str
    selected_model: Optional[str] = None
    decision_type: Optional[str] = None
    patch: Optional[str] = None
    branch: Optional[str] = None
    pull_request_url: Optional[str] = None
    test_output: Optional[str] = None


class V84IssueToPRAgent:
    """
    V8.4 GitHub Issue -> PR agent.

    The LLM proposes structured XML edits. Python controls everything
    else: repository state, routing, patch validation/application,
    syntax checks, tests, git, and PR creation.

    Model selection always flows through:
        V8.4 agent -> V8.3 /api/route -> V7.7 adaptive router
    """

    def __init__(
        self,
        repo_root: str,
        dry_run: bool = True,
        max_turns: int = 2,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.dry_run = dry_run
        self.max_turns = max(1, max_turns)

        self.github_token = os.getenv("GITHUB_TOKEN", "")
        self.github_repository = os.getenv("GITHUB_REPOSITORY", "")

        self.test_command = os.getenv(
            "V84_TEST_COMMAND",
            "python -m pytest -q",
        )

        self.max_patch_chars = 50000

        self.allowed_extensions = {
            ".py",
            ".yaml",
            ".yml",
            ".json",
            ".md",
            ".txt",
            ".toml",
            ".ini",
            ".cfg",
            ".sh",
            ".ipynb",
        }

        self.blocked_prefixes = (
            ".git/",
            ".github/workflows/",
        )

        self.blocked_names = {
            ".env",
            ".env.local",
            ".env.production",
        }

        # Set when a repair turn is needed. Every model turn rebuilds
        # repository state from disk, never from this cache.
        self._repair: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Repository state (source of truth, rebuilt every turn)
    # ------------------------------------------------------------------

    def git_status(self) -> str:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout
            return result.stderr
        except Exception as exc:
            return f"git status unavailable: {exc}"

    def git_diff(self) -> str:
        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout
            return result.stderr
        except Exception as exc:
            return f"git diff unavailable: {exc}"

    def collect_repository_context(self) -> str:
        """Collect CURRENT repository structure and file contents."""

        files: List[str] = []
        contents: List[str] = []

        total_chars = 0
        max_total_chars = 60000
        max_file_chars = 10000

        for path in self.repo_root.rglob("*"):
            if not path.is_file():
                continue

            relative = path.relative_to(self.repo_root).as_posix()

            if relative.startswith(self.blocked_prefixes):
                continue

            if relative in self.blocked_names:
                continue

            if any(
                part in (".venv", "venv", "__pycache__", ".git")
                for part in Path(relative).parts
            ):
                continue

            if path.suffix.lower() not in self.allowed_extensions:
                continue

            files.append(relative)

        files.sort()

        for relative in files:
            if not (
                relative.startswith("app/")
                or relative.startswith("tests/")
                or relative.startswith("config/")
                or relative == "README.md"
                or relative == "requirements.txt"
            ):
                continue

            if total_chars >= max_total_chars:
                break

            file_path = self.repo_root / relative

            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue

            if len(content) > max_file_chars:
                content = content[:max_file_chars]
                content += "\n... [TRUNCATED] ..."

            block = (
                "\n" + "=" * 80
                + f"\nFILE: {relative}\n"
                + "=" * 80 + "\n"
                + content + "\n"
            )

            if total_chars + len(block) > max_total_chars:
                break

            contents.append(block)
            total_chars += len(block)

        return (
            "Repository root:\n"
            f"{self.repo_root}\n\n"
            "Repository file tree:\n"
            + "\n".join(files[:500])
            + "\n\nCURRENT git status:\n"
            + self.git_status()
            + "\n\nCURRENT git diff:\n"
            + self.git_diff()[:20000]
            + "\n\nRelevant source and test file contents:\n"
            + "".join(contents)
        )

    # ------------------------------------------------------------------
    # GitHub event
    # ------------------------------------------------------------------

    def load_github_event(self) -> Dict[str, Any]:
        event_path = os.getenv("GITHUB_EVENT_PATH")

        if not event_path:
            return {}

        path = Path(event_path)

        if not path.exists():
            return {}

        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # V8.3 routing API context
    # ------------------------------------------------------------------

    def get_router_context(self) -> Dict[str, Any]:
        """Probe the existing V8.3 routing API (POST /api/route)."""

        try:
            from app.api_server import app

            client = TestClient(app)

            response = client.post(
                "/api/route",
                json={"prompt": "V8.4 repository context probe"},
            )

            if response.status_code == 200:
                return response.json()

            return {"router_error": f"HTTP {response.status_code}"}

        except Exception as exc:
            return {"router_error": str(exc)}

    # ------------------------------------------------------------------
    # Adaptive routing + Bedrock
    # ------------------------------------------------------------------

    def generate_model_response(
        self,
        prompt: str,
    ) -> Dict[str, Any]:
        """Route via V8.3 /api/route, then invoke Bedrock unchanged."""

        from app.api_server import app
        from app.models.v7_bedrock import V7BedrockClient

        client = TestClient(app)

        route_response = client.post(
            "/api/route",
            json={"prompt": prompt},
        )

        if route_response.status_code != 200:
            raise RuntimeError(
                "V8.3 /api/route failed: "
                f"{route_response.status_code} {route_response.text}"
            )

        route_data = route_response.json()

        selected_model = route_data.get("selected_model")
        decision_type = route_data.get("decision_type")

        if not selected_model:
            raise RuntimeError(
                "V8.3 /api/route did not return selected_model."
            )

        bedrock = V7BedrockClient(region="eu-north-1")

        response = bedrock.invoke(
            selected_model,
            prompt,
            max_tokens=4096,
            temperature=0.1,
        )

        return {
            "router": {
                "selected_model": selected_model,
                "decision_type": decision_type,
                "v75_candidate": route_data.get("v75_candidate"),
                "v76_best_model": route_data.get("v76_best_model"),
                "final_score": route_data.get("final_score"),
            },
            "response": response,
        }

    # ------------------------------------------------------------------
    # Structured edit parsing
    # ------------------------------------------------------------------

    @staticmethod
    def extract_edits(text: str) -> List[Dict[str, Any]]:
        import xml.etree.ElementTree as ET

        text = (text or "").strip()

        start = text.find("<edits>")
        end = text.rfind("</edits>")

        if start == -1 or end == -1:
            raise ValueError(
                "Model response did not contain XML edits."
            )

        xml_text = text[start:end + len("</edits>")]

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid XML edit response: {exc}")

        if root.tag != "edits":
            raise ValueError("Root XML element must be <edits>.")

        edits = []

        for node in root.findall("edit"):
            path_node = node.find("path")
            operation_node = node.find("operation")
            old_node = node.find("old")
            new_node = node.find("new")

            if path_node is None or not (path_node.text or "").strip():
                raise ValueError("Edit is missing <path>.")

            if operation_node is None or not (
                operation_node.text or ""
            ).strip():
                raise ValueError("Edit is missing <operation>.")

            edit = {
                "path": path_node.text.strip(),
                "operation": operation_node.text.strip(),
                "old": old_node.text if old_node is not None else "",
                "new": new_node.text if new_node is not None else "",
            }

            edits.append(edit)

        if not edits:
            raise ValueError("Model returned no XML edits.")

        return edits

    # ------------------------------------------------------------------
    # Issue requirement detection
    # ------------------------------------------------------------------

    @staticmethod
    def issue_allows_new_test_file(issue_body: str) -> bool:
        body = (issue_body or "").lower()

        phrases = (
            "new test file",
            "create a test file",
            "create new test",
            "add a new test file",
            "add new test file",
        )

        return any(phrase in body for phrase in phrases)

    @staticmethod
    def issue_requires_tests(issue_body: str) -> bool:
        """Detect explicit test requirements in arbitrary issue text."""

        body = (issue_body or "").lower()

        if not body:
            return False

        negative_phrases = (
            "no test required",
            "no tests required",
            "no need to add a test",
            "no need to update a test",
            "do not add a test",
            "do not update tests",
            "tests are not required",
        )

        if any(phrase in body for phrase in negative_phrases):
            return False

        positive_patterns = (
            r"\badd\b.{0,80}\b(?:or\s+update\s+)?(?:the\s+)?(?:relevant\s+)?(?:pytest\s+)?tests?\b",
            r"\bupdate\b.{0,80}\b(?:the\s+)?(?:relevant\s+)?(?:pytest\s+)?tests?\b",
            r"\bwrite\b.{0,80}\b(?:the\s+)?(?:relevant\s+)?(?:pytest\s+)?tests?\b",
            r"\bcreate\b.{0,80}\b(?:the\s+)?(?:relevant\s+)?(?:pytest\s+)?tests?\b",
            r"\binclude\b.{0,80}\b(?:the\s+)?(?:relevant\s+)?(?:pytest\s+)?tests?\b",
            r"\bmodify\b.{0,80}\b(?:the\s+)?(?:relevant\s+)?(?:pytest\s+)?tests?\b",
            r"\b(?:the\s+)?test\s+(?:must|should|need|needs|required|to)\b",
            r"\b(?:pytest\s+)?tests?\s+(?:must|should|need|needs|required|to)\b",
            r"\bpytest\s+tests?\b",
            r"\brelevant\s+pytest\s+tests?\b",
            r"\brelevant\s+tests?\b",
            r"\btests?\s+for\s+the\b",
        )

        return any(
            re.search(pattern, body)
            for pattern in positive_patterns
        )

    # ------------------------------------------------------------------
    # Test architecture inspection
    # ------------------------------------------------------------------

    def existing_test_files(self) -> List[str]:
        test_files = []

        tests_root = self.repo_root / "tests"

        if not tests_root.exists():
            return test_files

        for file_path in tests_root.rglob("*"):
            if not file_path.is_file():
                continue

            test_files.append(
                file_path.relative_to(self.repo_root).as_posix()
            )

        return sorted(test_files)

    def known_test_fixtures(self) -> set:
        """Names of functions defined anywhere under tests/ (fixtures)."""

        fixtures = set()

        for relative in self.existing_test_files():
            try:
                content = (self.repo_root / relative).read_text(
                    encoding="utf-8"
                )
            except Exception:
                continue

            for match in re.finditer(
                r"^\s*def (\w+)\(", content, flags=re.MULTILINE
            ):
                fixtures.add(match.group(1))

        return fixtures

    # ------------------------------------------------------------------
    # Patch building (in memory, working tree untouched)
    # ------------------------------------------------------------------

    def edits_to_patch(self, text: str) -> str:
        edits = self.extract_edits(text)

        modified_files: Dict[str, tuple[str, str]] = {}

        for edit in edits:
            if not isinstance(edit, dict):
                raise ValueError("Each edit must be an object.")

            relative = edit.get("path")
            operation = edit.get("operation", "replace")

            if not isinstance(relative, str) or not relative.strip():
                raise ValueError("Edit path must be a string.")

            relative = relative.replace("\\", "/").strip()

            if relative.startswith("/"):
                raise ValueError(
                    f"Absolute path is not allowed: {relative}"
                )

            if ".." in Path(relative).parts:
                raise ValueError(
                    f"Parent traversal is not allowed: {relative}"
                )

            if relative.startswith(self.blocked_prefixes):
                raise ValueError(
                    f"Protected path is not allowed: {relative}"
                )

            if relative in self.blocked_names:
                raise ValueError(
                    f"Protected file is not allowed: {relative}"
                )

            if (
                Path(relative).suffix.lower()
                not in self.allowed_extensions
            ):
                raise ValueError(
                    f"File extension is not allowed: {relative}"
                )

            file_path = self.repo_root / relative

            if relative not in modified_files:
                if file_path.exists():
                    original = file_path.read_text(encoding="utf-8")
                else:
                    original = ""
                modified_files[relative] = (original, original)

            original, current = modified_files[relative]

            if operation == "replace":
                old = edit.get("old")
                new = edit.get("new")

                if not isinstance(old, str) or not old:
                    raise ValueError(
                        f"Missing non-empty 'old' for {relative}"
                    )

                if not isinstance(new, str):
                    raise ValueError(
                        f"Missing string 'new' for {relative}"
                    )

                occurrences = current.count(old)

                if occurrences == 0:
                    raise ValueError(
                        f"Exact old text was not found in {relative}"
                    )

                if occurrences > 1:
                    raise ValueError(
                        f"Old text occurs {occurrences} times in "
                        f"{relative}; edit must be unambiguous."
                    )

                current = current.replace(old, new, 1)

            elif operation == "create":
                if file_path.exists():
                    raise ValueError(
                        f"Cannot create existing file: {relative}"
                    )

                new = edit.get("new")

                if not isinstance(new, str):
                    raise ValueError(
                        f"Missing string 'new' content for {relative}"
                    )

                current = new

            elif operation == "delete":
                if not file_path.exists():
                    raise ValueError(
                        f"Cannot delete missing file: {relative}"
                    )

                current = ""

            elif operation == "insert_after":
                anchor = edit.get("old")
                new_text = edit.get("new")

                if not isinstance(anchor, str) or not anchor:
                    raise ValueError(
                        f"Missing non-empty anchor for {relative}"
                    )

                if not isinstance(new_text, str):
                    raise ValueError(
                        f"Missing string 'new' for {relative}"
                    )

                occurrences = current.count(anchor)

                if occurrences == 0:
                    raise ValueError(
                        f"Insert anchor was not found in {relative}"
                    )

                if occurrences > 1:
                    raise ValueError(
                        f"Insert anchor occurs {occurrences} times in "
                        f"{relative}; edit must be unambiguous."
                    )

                current = current.replace(
                    anchor, anchor + new_text, 1
                )

            elif operation == "append":
                if not file_path.exists():
                    raise ValueError(
                        f"Cannot append to nonexistent file: {relative}"
                    )

                new_text = edit.get("new")

                if not isinstance(new_text, str):
                    raise ValueError(
                        f"Missing string 'new' for {relative}"
                    )

                if current and not current.endswith("\n"):
                    current += "\n"

                current += new_text

            else:
                raise ValueError(
                    f"Unsupported edit operation: {operation}"
                )

            modified_files[relative] = (original, current)

        patch_parts = []

        for relative, (original, modified) in modified_files.items():
            if original == modified:
                raise ValueError(
                    f"Edit produced no change in {relative}"
                )

            diff_lines = list(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    modified.splitlines(keepends=True),
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                    n=3,
                )
            )

            if not diff_lines:
                raise ValueError(
                    f"Unable to generate diff for {relative}"
                )

            # In-memory Python syntax check. The working tree is never
            # touched, but syntactically broken output is rejected here
            # instead of after application.
            if relative.endswith(".py") and modified:
                try:
                    compile(
                        modified, relative, "exec"
                    )
                except SyntaxError as exc:
                    raise ValueError(
                        f"Generated content for {relative} is not "
                        f"valid Python: line {exc.lineno}: {exc.msg}"
                    ) from exc

            patch_parts.append(
                f"diff --git a/{relative} b/{relative}\n"
            )
            patch_parts.extend(diff_lines)

        return "".join(patch_parts).rstrip("\n") + "\n"

    # ------------------------------------------------------------------
    # Patch path extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_patch_paths(patch: str) -> List[str]:
        paths = []

        for line in patch.splitlines():
            if line.startswith("+++ b/"):
                paths.append(line[6:].strip())
            elif line.startswith("--- a/"):
                paths.append(line[6:].strip())

        unique = []

        for path in paths:
            if path != "/dev/null" and path not in unique:
                unique.append(path)

        return unique

    # ------------------------------------------------------------------
    # Deterministic patch validation
    # ------------------------------------------------------------------

    def validate_patch(
        self,
        patch: str,
        issue_body: str = "",
    ) -> List[str]:
        if not patch.strip():
            raise ValueError("Patch is empty.")

        if len(patch) > self.max_patch_chars:
            raise ValueError(
                f"Patch exceeds {self.max_patch_chars} characters."
            )

        if not patch.startswith("diff --git "):
            raise ValueError(
                "Patch must start with 'diff --git'."
            )

        paths = self.extract_patch_paths(patch)

        if not paths:
            raise ValueError("Patch contains no valid file paths.")

        for path in paths:
            normalized = path.replace("\\", "/")

            if normalized.startswith("/"):
                raise ValueError(
                    f"Absolute path is not allowed: {path}"
                )

            if ".." in Path(normalized).parts:
                raise ValueError(
                    f"Parent traversal is not allowed: {path}"
                )

            if normalized.startswith(self.blocked_prefixes):
                raise ValueError(
                    f"Protected path is not allowed: {path}"
                )

            if normalized in self.blocked_names:
                raise ValueError(
                    f"Protected file is not allowed: {path}"
                )

            if (
                Path(normalized).suffix.lower()
                not in self.allowed_extensions
            ):
                raise ValueError(
                    f"File extension is not allowed: {path}"
                )

        self.validate_api_version_guard(patch, issue_body)
        self.validate_test_fixture_guard(patch)

        if shutil.which("git") is None:
            # Deterministic fallback when git is unavailable: the diff
            # was already derived from exact in-memory matching, so a
            # structural check is the best available validation.
            if "@@" not in patch:
                raise ValueError(
                    "Patch contains no hunks; cannot validate."
                )
            return paths

        patch_file = self.repo_root / ".v84_validation.patch"

        try:
            # newline="" keeps LF line endings so git sees exactly the
            # bytes the diff was generated from.
            patch_file.write_text(
                patch, encoding="utf-8", newline=""
            )

            result = subprocess.run(
                [
                    "git", "apply", "--recount", "--check",
                    "--whitespace=nowarn", str(patch_file),
                ],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                error = (
                    result.stderr.strip()
                    or result.stdout.strip()
                    or "git apply --check failed."
                )
                raise RuntimeError(
                    f"git apply --check failed: {error}"
                )

        finally:
            try:
                patch_file.unlink()
            except FileNotFoundError:
                pass

        return paths

    # ------------------------------------------------------------------
    # API version protection
    # ------------------------------------------------------------------

    _VERSION_CHANGE_PATTERNS = (
        r"api version",
        r"version change",
        r"change.{0,30}version",
        r"update.{0,30}version",
        r"bump.{0,30}version",
        r"version.{0,30}(change|update|bump|new)",
        r"new version",
        r"routing version",
    )

    @classmethod
    def _issue_allows_version_change(cls, issue_body: str) -> bool:
        body = (issue_body or "").lower()
        return any(
            re.search(pattern, body)
            for pattern in cls._VERSION_CHANGE_PATTERNS
        )

    def validate_api_version_guard(
        self,
        patch: str,
        issue_body: str = "",
    ) -> None:
        """
        Reject patches that change an existing API version ("V8.3") or
        routing version ("V7.7") unless the issue explicitly asks for a
        version change.
        """

        if self._issue_allows_version_change(issue_body):
            return

        removed_versions = set()
        added_versions = set()

        for line in patch.splitlines():
            match = re.search(r'"(?:routing_)?version"\s*:\s*"([^"]+)"', line)

            if not match:
                continue

            if line.startswith("-"):
                removed_versions.add(match.group(1))
            elif line.startswith("+"):
                added_versions.add(match.group(1))

        if removed_versions and added_versions and (
            removed_versions != added_versions
        ):
            raise ValueError(
                "Patch changes an existing API/routing version "
                f"({sorted(removed_versions)} -> "
                f"{sorted(added_versions)}) but the issue does not "
                "request a version change. Preserve the current version."
            )

        # A patch may also remove a version line outright.
        if removed_versions and not added_versions:
            raise ValueError(
                "Patch removes an existing API/routing version "
                f"{sorted(removed_versions)} but the issue does not "
                "request a version change."
            )

    # ------------------------------------------------------------------
    # Test fixture protection
    # ------------------------------------------------------------------

    def validate_test_fixture_guard(self, patch: str) -> None:
        """
        Generated tests must use the existing test module's patterns:
        - no invented pytest fixtures (e.g. `test_client`),
        - when the repo uses a module-level TestClient, tests must use
          it directly instead of taking it as a parameter.
        """

        known_fixtures = self.known_test_fixtures()

        for path in self.extract_patch_paths(patch):
            if not path.startswith("tests/"):
                continue

            file_path = self.repo_root / path
            existing = ""
            if file_path.exists():
                try:
                    existing = file_path.read_text(encoding="utf-8")
                except Exception:
                    existing = ""

            added_test_defs = re.findall(
                r"^\+\s*def (test_\w*)\(([^)]*)\)",
                patch,
                flags=re.MULTILINE,
            )

            uses_module_client = (
                "TestClient(" in existing and "client =" in existing
            )

            for _name, params in added_test_defs:
                for param in [
                    p.strip() for p in params.split(",") if p.strip()
                ]:
                    if uses_module_client and param in (
                        "client",
                        "test_client",
                    ):
                        raise ValueError(
                            "Generated test takes a 'client'/'test_client' "
                            "parameter, but the repository uses a "
                            f"module-level TestClient in {path}. Use the "
                            "existing module-level client instead."
                        )

                    if param not in known_fixtures and param not in (
                        "self",
                    ):
                        raise ValueError(
                            f"Generated test uses pytest fixture "
                            f"'{param}' which does not exist in the "
                            "repository. Use the existing test patterns."
                        )

    # ------------------------------------------------------------------
    # Syntax validation
    # ------------------------------------------------------------------

    def validate_python_syntax(self, paths: List[str]) -> List[str]:
        python_files = [p for p in paths if p.endswith(".py")]

        for relative in python_files:
            file_path = self.repo_root / relative

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Changed Python file is missing: {relative}"
                )

            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(file_path)],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                detail = (
                    result.stderr.strip()
                    or result.stdout.strip()
                    or "py_compile failed."
                )
                raise RuntimeError(
                    f"Python syntax validation failed for "
                    f"{relative}:\n{detail}"
                )

        return python_files

    # ------------------------------------------------------------------
    # Git
    # ------------------------------------------------------------------

    def run_git(
        self,
        args: List[str],
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if check and result.returncode != 0:
            error = (
                result.stderr.strip()
                or result.stdout.strip()
                or "git command failed without output."
            )
            raise subprocess.CalledProcessError(
                result.returncode,
                ["git", *args],
                output=result.stdout,
                stderr=error,
            )

        return result

    def create_branch(self, issue_number: int) -> str:
        branch = f"agent/issue-{issue_number}"
        self.run_git(["checkout", "-B", branch])
        return branch

    def apply_patch(self, patch: str) -> None:
        patch_file = self.repo_root / ".v84_apply.patch"

        try:
            patch_file.write_text(
                patch.rstrip("\n") + "\n",
                encoding="utf-8",
                newline="",
            )

            self.run_git(
                ["apply", "--whitespace=nowarn", str(patch_file)]
            )

        finally:
            try:
                patch_file.unlink()
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def run_tests(
        self,
        changed_paths: Optional[List[str]] = None,
    ) -> str:
        outputs = []
        returncode = 0

        targeted = [
            p for p in (changed_paths or [])
            if p.startswith("tests/") and p.endswith(".py")
        ]

        if targeted:
            targeted_cmd = (
                f"{sys.executable} -m pytest -q "
                + " ".join(targeted)
            )
            result = subprocess.run(
                targeted_cmd,
                cwd=self.repo_root,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            outputs.append(
                "Targeted tests: " + targeted_cmd + "\n"
                + result.stdout[-8000:] + "\n"
                + result.stderr[-8000:]
            )
            returncode = result.returncode

        if returncode == 0:
            result = subprocess.run(
                self.test_command,
                cwd=self.repo_root,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
            outputs.append(
                "Full suite: " + self.test_command + "\n"
                + result.stdout[-8000:] + "\n"
                + result.stderr[-8000:]
            )
            returncode = result.returncode

        output = "\n".join(outputs)

        if returncode != 0:
            raise RuntimeError("Tests failed:\n" + output)

        return output

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        issue_number: int,
        issue_title: str,
        issue_body: str,
        repository_context: str,
        router_context: Dict[str, Any],
        repair: Optional[Dict[str, Any]] = None,
    ) -> str:
        existing_tests = self.existing_test_files()

        existing_tests_text = (
            "\n".join(existing_tests)
            if existing_tests
            else "(No existing test files found.)"
        )

        repair_section = ""

        if repair:
            state = repair.get("state")

            if state == "validation_failed":
                repair_section = (
                    "REPAIR CONTEXT (validation failure):\n"
                    "The previous patch was NOT applied.\n\n"
                    "Exact validation error:\n"
                    + str(repair.get("error"))
                    + "\n\nPrevious candidate patch/response:\n"
                    + str(repair.get("patch_text") or "(unavailable)")
                    + "\n\nThe repository files above show the CURRENT "
                    "state. Generate a NEW patch against the CURRENT "
                    "repository state.\n\n"
                )

            elif state == "tests_failed":
                changed = repair.get("changed_files") or []
                contents = []

                for rel in changed:
                    try:
                        content = (
                            self.repo_root / rel
                        ).read_text(encoding="utf-8")
                        contents.append(
                            f"CURRENT FILE {rel}:\n{content[:8000]}\n"
                        )
                    except Exception:
                        pass

                repair_section = (
                    "REPAIR CONTEXT (test/syntax failure):\n"
                    "The previous patch HAS already been applied.\n\n"
                    "Exact error / test output:\n"
                    + str(repair.get("error") or repair.get("test_output") or "(none)")
                    + "\n\nCURRENT git diff:\n"
                    + str(repair.get("git_diff") or "(empty)")
                    + "\n\n"
                    + "\n".join(contents)
                    + "\nGenerate a repair patch against the CURRENT "
                    "working tree shown above. Do NOT reuse stale "
                    "source text.\n\n"
                )

        requires_tests = self.issue_requires_tests(issue_body)

        prompt = (
            "You are the code-editing component of the V8.4 "
            "autonomous GitHub Issue -> PR agent.\n\n"

            f"Issue #{issue_number}\n\n"

            "TITLE:\n" + issue_title + "\n\n"

            "ISSUE:\n" + issue_body + "\n\n"

            "REPOSITORY CONTEXT (CURRENT STATE):\n"
            + repository_context
            + "\n\n"

            "EXISTING TEST FILES:\n"
            + existing_tests_text
            + "\n\n"

            "ROUTER:\n"
            + json.dumps(router_context, indent=2, default=str)
            + "\n\n"

            "Your task is to describe the smallest safe change "
            "needed to satisfy the issue.\n\n"

            "CRITICAL OUTPUT REQUIREMENT:\n"
            "Return ONLY a valid <edits> XML document. "
            "Begin with <edits> and end with </edits>. "
            "No Markdown, code fences, prose, or unified diff.\n\n"

            "RULES:\n"
            "1. Use only files that exist in the repository context.\n"
            "2. <old> must match exactly one location in the real file.\n"
            "3. Do not invent surrounding source or line numbers.\n"
            "4. Preserve the existing architecture and public API contract.\n"
            "5. Never change the existing API version or routing version "
            "unless the issue explicitly asks for a version change.\n"
            "6. Make the smallest possible change.\n"
            "7. If the issue requires a test change, modify an EXISTING "
            "test file. Do NOT invent fixture parameters like "
            "`test_client`. If the repository defines a module-level "
            "TestClient (e.g. `client = TestClient(app)`), use that "
            "existing client in tests.\n"
            "8. If the issue does not require tests, DO NOT modify "
            "anything under tests/.\n"
            "9. Do not touch secrets, Docker, GitHub Actions, or "
            "infrastructure files.\n"
            "10. Each edit's <old> must be unique within its file.\n\n"

            "REQUIRED OUTPUT FORMAT:\n\n"
            "<edits>\n"
            "  <edit>\n"
            "    <path>app/api/v8_routes.py</path>\n"
            "    <operation>replace</operation>\n"
            "    <old><![CDATA[\nEXACT EXISTING SOURCE\n]]></old>\n"
            "    <new><![CDATA[\nREPLACEMENT SOURCE\n]]></new>\n"
            "  </edit>\n"
            "</edits>\n\n"

            "ALLOWED OPERATIONS: replace, create, delete, "
            "insert_after, append\n\n"

            "The Python agent generates the git diff itself and "
            "controls validation, testing, git, and PR creation.\n\n"

            + ("This issue EXPLICITLY requires test changes. "
               "Your patch MUST modify an existing test file under "
               "tests/.\n\n" if requires_tests else "")
            + repair_section
        )

        return prompt

    # ------------------------------------------------------------------
    # Pull request
    # ------------------------------------------------------------------

    def create_pull_request(
        self,
        issue_number: int,
        branch: str,
        title: str,
        body: str,
    ) -> str:
        if not self.github_token:
            raise RuntimeError(
                "GITHUB_TOKEN is required to create a pull request."
            )

        if not self.github_repository:
            raise RuntimeError(
                "GITHUB_REPOSITORY is required to create a PR."
            )

        url = (
            "https://api.github.com/repos/"
            f"{self.github_repository}/pulls"
        )

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
            with urllib.request.urlopen(request, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["html_url"]

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"GitHub PR creation failed: {exc.code} {detail}"
            )

    # ------------------------------------------------------------------
    # Main agent loop
    # ------------------------------------------------------------------

    def run(
        self,
        issue_number: int,
        issue_title: str,
        issue_body: str,
        repository: Optional[str] = None,
    ) -> AgentResult:
        print("=" * 100)
        print("V8.4 - GITHUB ISSUE -> PR AGENT")
        print("=" * 100)
        print(f"Issue: #{issue_number}")
        print(f"Title: {issue_title}")
        print(f"Dry run: {self.dry_run}")

        router_context = self.get_router_context()
        selected_model = None
        decision_type = None
        last_error = None
        last_patch = None
        last_test_output = None
        last_branch = None

        for turn in range(1, self.max_turns + 1):
            print(f"\nTurn {turn}/{self.max_turns}")

            # Every turn uses the CURRENT repository state.
            repository_context = self.collect_repository_context()

            prompt = self.build_prompt(
                issue_number=issue_number,
                issue_title=issue_title,
                issue_body=issue_body,
                repository_context=repository_context,
                router_context=router_context,
                repair=self._repair,
            )

            try:
                generated = self.generate_model_response(prompt)
            except Exception as exc:
                last_error = str(exc)
                print(f"Model routing/invocation failed: {last_error}")
                break

            decision = generated["router"]
            response = generated["response"]

            selected_model = decision.get("selected_model")
            decision_type = decision.get("decision_type")

            print(f"Model: {selected_model} ({decision_type})")

            if isinstance(response, dict):
                text = (
                    response.get("text")
                    or response.get("response")
                    or response.get("output")
                    or ""
                )
            else:
                text = str(response)

            patch = None
            applied = False

            try:
                # ------------------------------------------------
                # STATE A: everything before application fails here
                # ------------------------------------------------
                patch = self.edits_to_patch(text)
                paths = self.validate_patch(patch, issue_body=issue_body)

                tests_required = self.issue_requires_tests(issue_body)
                test_paths = [
                    p for p in paths if p.startswith("tests/")
                ]

                if tests_required and not test_paths:
                    raise ValueError(
                        "Issue explicitly requires a test change, but "
                        "the generated patch does not modify anything "
                        "under tests/."
                    )

                if tests_required and not self.issue_allows_new_test_file(
                    issue_body
                ):
                    existing = set(self.existing_test_files())
                    invented = [
                        p for p in test_paths if p not in existing
                    ]
                    if invented:
                        raise ValueError(
                            "Issue requires a test change, but the patch "
                            "targets nonexistent test file(s): "
                            + ", ".join(invented)
                            + ". Modify an existing test file instead."
                        )

                if test_paths and not tests_required:
                    raise ValueError(
                        "Patch modifies test files, but the issue does "
                        "not explicitly require test changes: "
                        + ", ".join(test_paths)
                    )

                print("Patch validation passed. Files:", paths)

                if self.dry_run:
                    return AgentResult(
                        success=True,
                        issue_number=issue_number,
                        message=(
                            "Dry-run passed. Patch is valid and "
                            "applicable. Working tree unchanged."
                        ),
                        selected_model=selected_model,
                        decision_type=decision_type,
                        patch=patch,
                    )

                # ------------------------------------------------
                # Apply patch (STATE B territory begins here)
                # ------------------------------------------------
                last_branch = self.create_branch(issue_number)
                applied = True
                self.apply_patch(patch)

                # No-op detection: the applied patch must change
                # something in the working tree.
                git_state = self.git_status()
                if not git_state.strip():
                    raise RuntimeError(
                        "Patch produced no actual change in the "
                        "working tree."
                    )

                self.validate_python_syntax(paths)

                last_test_output = self.run_tests(
                    changed_paths=paths
                )

                self.run_git(["add", "--", *paths])
                self.run_git(
                    ["commit", "-m", f"fix: resolve issue #{issue_number}"]
                )
                self.run_git(["push", "-u", "origin", last_branch])

                pr_url = self.create_pull_request(
                    issue_number=issue_number,
                    branch=last_branch,
                    title=f"Fix #{issue_number}: {issue_title}",
                    body=(
                        f"Automated fix for #{issue_number}.\n\n"
                        "Generated and validated by V8.4 "
                        "Issue-to-PR Agent."
                    ),
                )

                return AgentResult(
                    success=True,
                    issue_number=issue_number,
                    message="Issue-to-PR workflow completed.",
                    selected_model=selected_model,
                    decision_type=decision_type,
                    patch=patch,
                    branch=last_branch,
                    pull_request_url=pr_url,
                    test_output=last_test_output,
                )

            except Exception as exc:
                last_error = str(exc)
                last_patch = patch if patch else text

                # A no-op check failure ("Patch produced no actual
                # change") also means the tree was not modified.
                no_op = "no actual change" in last_error

                if applied and not no_op:
                    self._repair = {
                        "state": "tests_failed",
                        "error": last_error,
                        "test_output": last_test_output,
                        "git_diff": self.git_diff(),
                        "changed_files": (
                            locals().get("paths") or []
                        ),
                    }
                    print(
                        "Patch was applied but validation/tests "
                        f"failed: {last_error}"
                    )
                else:
                    self._repair = {
                        "state": "validation_failed",
                        "error": last_error,
                        "patch_text": last_patch,
                    }
                    print(f"Patch validation failed: {last_error}")

                if turn >= self.max_turns:
                    break

        return AgentResult(
            success=False,
            issue_number=issue_number,
            message=(
                "V8.4 agent exhausted its bounded repair attempts: "
                + str(last_error)
            ),
            selected_model=selected_model,
            decision_type=decision_type,
            patch=last_patch,
            test_output=last_test_output,
        )


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main() -> None:
    repo_root = os.getenv("V84_REPO_ROOT", os.getcwd())

    event_path = os.getenv("GITHUB_EVENT_PATH")

    issue_number = int(os.getenv("V84_ISSUE_NUMBER", "999"))
    issue_title = os.getenv("V84_ISSUE_TITLE", "Automated Issue")
    issue_body = os.getenv(
        "V84_ISSUE_BODY", "Make the requested change."
    )

    if event_path and Path(event_path).exists():
        try:
            event = json.loads(Path(event_path).read_text())
            issue = event.get("issue", {})
            issue_number = int(issue.get("number", issue_number))
            issue_title = issue.get("title", issue_title)
            issue_body = issue.get("body", issue_body)
        except Exception:
            pass

    dry_run = (
        os.getenv("V84_DRY_RUN", "true").lower() == "true"
    )
    max_turns = int(os.getenv("V84_MAX_TURNS", "2"))

    agent = V84IssueToPRAgent(
        repo_root=repo_root,
        dry_run=dry_run,
        max_turns=max_turns,
    )

    result = agent.run(
        issue_number=issue_number,
        issue_title=issue_title,
        issue_body=issue_body,
    )

    print("\n" + "=" * 100)
    print("V8.4 RESULT")
    print("=" * 100)
    print(json.dumps(result.__dict__, indent=2, default=str))

    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
