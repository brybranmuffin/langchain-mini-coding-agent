# LangChain Mini Coding Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `mini_coding_agent.py` (rasbt/mini-coding-agent) to LangChain + LangGraph as a small package split by LangChain component, preserving six capabilities: Ollama inference, short/long-term memory, context management, tools + validation, repo reading, and prompt-prefix caching.

**Architecture:** A LangGraph `StateGraph` replaces the hand-written `while` loop in `MiniAgent.ask`. Each capability lives in its own module built on the matching LangChain/LangGraph primitive (`ChatOllama`, `StructuredTool` + Pydantic, `trim_messages`, `ChatPromptTemplate`, checkpointer, `BaseStore`, `interrupt`). A thin `Agent` facade hides thread config and the approval-interrupt loop from the CLI and tests.

**Tech Stack:** Python ≥3.10, uv, langchain-core, langgraph, langchain-ollama, langgraph-checkpoint-sqlite, pydantic v2, pytest.

**Spec:** No separate spec. Behavior reference is `/Users/bryantbettencourt/my-venv/mini-coding-agent/mini_coding_agent.py` (and its `tests/test_mini_coding_agent.py`). The component map below is the design.

## Component Map

| # | Capability (your list) | Original code | LangChain / LangGraph component | New module |
|---|---|---|---|---|
| 1 | Inference model on Ollama | `OllamaModelClient` (`/api/generate`, raw text) | `langchain_ollama.ChatOllama` + `.bind_tools()` (native tool calling replaces `<tool>` XML parsing) | `model.py` |
| 2 | Long- and short-term memory | `SessionStore` JSON, `session["memory"]`, `note_tool`, `remember` | **Short-term:** LangGraph checkpointer (`SqliteSaver`, one thread per session) holding messages + working memory. **Long-term:** LangGraph `BaseStore` (`SqliteStore`) holding cross-session notes per repo + "latest session" pointer | `memory.py` |
| 3 | Context management | `clip`, `history_text` | `langchain_core.messages.trim_messages` with a char counter, plus a per-message compression pass (clip, dedupe old reads) applied to a *copy* of state before each model call | `context.py` |
| 4 | Tools and verification | `build_tools`, `validate_tool`, `approve`, `run_tool`, `tool_*` | `StructuredTool` with Pydantic `args_schema` (validation), `ToolException`, workspace sandbox; approvals via LangGraph `interrupt()` / `Command(resume=...)` | `tools.py`, `graph.py` |
| 5 | Read the repo | `WorkspaceContext` | Plain dataclass (git + doc snapshot) rendered into the system prompt; live reads via the tools | `workspace.py` |
| 6 | Prompt prefix caching | `build_prefix` + `prompt()` | `ChatPromptTemplate` with a byte-stable `SystemMessage` built once per graph; memory/history follow it. Ollama's KV cache reuses the shared prefix; `keep_alive` + adequate `num_ctx` keep it valid | `prompts.py`, `model.py` |

Graph (in `graph.py`):

```
START → start_turn → agent ─┬─ tool_calls ──────────→ tools ─┬─ budget left → agent
                            ├─ empty / invalid call ─→ retry ─┤
                            └─ text answer ──→ finalize → END └─ budget spent → stop → END
```

**Deliberate changes from the original** (review these before approving the plan):
- **Delegation / subagents (original component 6) is out of scope** — it isn't in your list. It can be added later as a subgraph tool.
- **XML `<tool>`/`<final>` protocol is replaced by native tool calling.** Final answer = an AI message with no tool calls. Malformed calls surface as `AIMessage.invalid_tool_calls` and go through the same retry path.
- **Long-term memory is new**: final-answer notes persist across sessions per repo (max 10) and are shown in the memory block of new sessions.
- **`/reset` starts a new thread id** instead of wiping the old session file; the old thread stays in the checkpoint DB.
- Sessions live in `.mini-coding-agent/state.sqlite` instead of one JSON file per session.

## Global Constraints

- Python `>=3.10`; package at `src/langchain_mini_agent/`; CLI entry point `lc-mini-agent`.
- Dependencies added with `uv add` (no hand-pinned versions): `langchain-core langgraph langchain-ollama langgraph-checkpoint-sqlite`; dev: `pytest ruff`.
- Defaults copied from the original: model `qwen3.5:4b`, host `http://127.0.0.1:11434`, timeout `300`, approval `ask`, max steps `6`, max new tokens `512`, temperature `0.2`, top-p `0.9`. New: `num_ctx` `16384`, `keep_alive` `"30m"`.
- Limits copied verbatim: `MAX_TOOL_OUTPUT = 4000`, `MAX_HISTORY = 12000` chars, recent window `6` messages, recent clip `900`, old tool clip `180`, old other clip `220`, doc snippet `1200`, git status `1500`, task `300`, working-memory files `8`, notes `5`, note clip `220`, shell timeout `[1, 120]`, list/search cap `200`.
- `max_attempts = max(max_steps * 3, max_steps + 4)`.
- State directory: `<repo_root>/.mini-coding-agent/`, DB file `state.sqlite`.
- Tests never hit the network; the only live-Ollama test is skipped unless `LIVE_OLLAMA=1`.
- Never wrap a call to `interrupt()` in a broad `try/except` (it raises to pause the graph).

## Review Focus

1. **Model emits several tool calls in one message** (qwen does this): only the first runs; every other call still gets a `ToolMessage` error reply so the transcript stays valid for Ollama. Test: `test_extra_tool_calls_get_error_replies` (Task 8).
2. **Long retry/tool loops hit LangGraph's default `recursion_limit` (25)** and crash with `GraphRecursionError` instead of the friendly stop message. Test: `test_stops_after_too_many_malformed_responses` runs 18 attempts (Task 8).
3. **Ollama's small default context silently truncates the prompt**, dropping the system prefix and wrecking the cache. Test: `test_build_chat_model_sets_context_and_cache_options` (Task 7).
4. **Approval interrupt resumes by re-running the tools node** — anything with side effects before `interrupt()` would run twice. Test: `test_ask_policy_approval_runs_tool_once` (Task 8).
5. **`--resume latest` in a repo with no sessions** should start a new session, not crash. Test: `test_build_agent_from_args_resume_latest_without_sessions` (Task 9).

---

## File Structure

```
pyproject.toml                       # Task 1
.gitignore                           # Task 1
src/langchain_mini_agent/
  __init__.py                        # Task 1
  __main__.py                        # Task 9
  context.py      # (3) clip, middle, compress_history          Task 2
  workspace.py    # (5) WorkspaceContext                        Task 3
  tools.py        # (4) WorkspaceTools, arg schemas, sandbox    Task 4
  memory.py       # (2) working memory, LongTermMemory, sqlite  Task 5
  prompts.py      # (6) build_prefix, PROMPT, render_prompt     Task 6
  model.py        # (1) build_chat_model                        Task 7
  graph.py        # StateGraph wiring, approvals, limits        Task 8
  agent.py        # Agent facade + create_agent                 Task 8
  cli.py          # argparse, REPL, welcome banner              Task 9
tests/
  fakes.py        # ScriptedChatModel, call(), final()          Task 7
  test_imports.py / test_context.py / test_workspace.py / test_tools.py /
  test_memory.py / test_prompts.py / test_model.py / test_graph.py /
  test_cli.py / test_live_ollama.py
```

---

### Task 1: Project scaffold and dependencies

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/langchain_mini_agent/__init__.py`
- Test: `tests/test_imports.py`

**Interfaces:**
- Consumes: nothing
- Produces: importable package `langchain_mini_agent`; installed deps.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "langchain-mini-coding-agent"
version = "0.1.0"
description = "LangChain/LangGraph port of mini-coding-agent, backed by Ollama"
requires-python = ">=3.10"
dependencies = []

[project.scripts]
lc-mini-agent = "langchain_mini_agent.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["tests"]
```

- [ ] **Step 2: Write `.gitignore` and the package init**

```gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
.mini-coding-agent/
*.egg-info/
```

`src/langchain_mini_agent/__init__.py`:

```python
"""LangChain/LangGraph port of mini-coding-agent."""
```

- [ ] **Step 3: Add dependencies**

Run:
```bash
uv add langchain-core langgraph langchain-ollama langgraph-checkpoint-sqlite
uv add --dev pytest ruff
```
Expected: `uv.lock` created, `dependencies` in `pyproject.toml` filled in.

- [ ] **Step 4: Write the import smoke test**

`tests/test_imports.py`:

```python
def test_dependencies_import():
    from langchain_core.messages import trim_messages  # noqa: F401
    from langchain_ollama import ChatOllama  # noqa: F401
    from langgraph.checkpoint.memory import InMemorySaver  # noqa: F401
    from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
    from langgraph.store.memory import InMemoryStore  # noqa: F401
    from langgraph.store.sqlite import SqliteStore  # noqa: F401
    from langgraph.types import Command, interrupt  # noqa: F401

    import langchain_mini_agent  # noqa: F401
```

- [ ] **Step 5: Run it**

Run: `uv run pytest tests/test_imports.py -v`
Expected: PASS. If `langgraph.store.sqlite` fails to import, stop and report — Task 5 depends on it.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .gitignore src tests
git commit -m "chore: scaffold langchain_mini_agent package and deps"
```

---

### Task 2: Context management (`context.py`)

**Files:**
- Create: `src/langchain_mini_agent/context.py`
- Test: `tests/test_context.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `MAX_TOOL_OUTPUT: int = 4000`, `MAX_HISTORY: int = 12000`, `RECENT_WINDOW: int = 6`
  - `clip(text, limit: int = MAX_TOOL_OUTPUT) -> str`
  - `middle(text, limit: int) -> str`
  - `char_count(messages: list[BaseMessage]) -> int`
  - `compress_history(messages: list[BaseMessage], recent_window: int = RECENT_WINDOW, budget: int = MAX_HISTORY) -> list[BaseMessage]` — returns new message objects, never mutates input.

- [ ] **Step 1: Write the failing tests**

`tests/test_context.py`:

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langchain_mini_agent.context import char_count, clip, compress_history, middle


def tool_pair(name, args, content, call_id):
    return [
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}]),
        ToolMessage(content=content, tool_call_id=call_id, name=name),
    ]


def filler(n=6):
    return [HumanMessage(f"m{i}") for i in range(n)]


def test_clip_short_text_unchanged():
    assert clip("abc", 10) == "abc"


def test_clip_marks_truncation():
    assert clip("a" * 20, 5) == "aaaaa\n...[truncated 15 chars]"


def test_middle_keeps_both_ends():
    assert middle("abcdefghij", 7) == "ab...ij"


def test_old_tool_output_clipped_to_180():
    history = [HumanMessage("task"), *tool_pair("read_file", {"path": "a.py"}, "x" * 1000, "1"), *filler()]
    out = compress_history(history)
    assert out[2].content == "x" * 180 + "\n...[truncated 820 chars]"


def test_recent_tool_output_clipped_to_900():
    history = [HumanMessage("task"), *tool_pair("read_file", {"path": "a.py"}, "x" * 1000, "1")]
    out = compress_history(history)
    assert out[-1].content.endswith("[truncated 100 chars]")


def test_duplicate_old_reads_are_replaced_until_file_changes():
    history = [
        HumanMessage("task"),
        *tool_pair("read_file", {"path": "a.py"}, "v1", "1"),
        *tool_pair("read_file", {"path": "a.py"}, "v1 again", "2"),
        *tool_pair("write_file", {"path": "a.py", "content": "v2"}, "wrote a.py", "3"),
        *tool_pair("read_file", {"path": "a.py"}, "v2", "4"),
        *filler(),
    ]
    out = compress_history(history)
    contents = [m.content for m in out if isinstance(m, ToolMessage)]
    assert contents == ["v1", "[duplicate read of a.py omitted]", "wrote a.py", "v2"]


def test_old_tool_call_args_are_clipped():
    history = [HumanMessage("t"), *tool_pair("write_file", {"path": "a.py", "content": "c" * 1000}, "wrote", "1"), *filler()]
    out = compress_history(history)
    assert len(out[1].tool_calls[0]["args"]["content"]) < 300


def test_does_not_mutate_input():
    history = [HumanMessage("t"), *tool_pair("run_shell", {"command": "ls"}, "x" * 1000, "1"), *filler()]
    compress_history(history)
    assert len(history[2].content) == 1000


def test_budget_drops_oldest_and_starts_on_human():
    history = []
    for i in range(40):
        history += [HumanMessage(f"q{i} " + "y" * 300), *tool_pair("run_shell", {"command": "ls"}, "z" * 300, str(i))]
    out = compress_history(history, budget=2000)
    assert isinstance(out[0], HumanMessage)
    assert out[-1].tool_call_id == "39"
    assert char_count(out) <= 2000


def test_keeps_latest_request_when_it_alone_exceeds_budget():
    out = compress_history([HumanMessage("old"), HumanMessage("q" * 5000)], budget=500)
    assert out[-1].content.startswith("q")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langchain_mini_agent.context'`.

- [ ] **Step 3: Implement**

`src/langchain_mini_agent/context.py`:

```python
import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage, trim_messages

MAX_TOOL_OUTPUT = 4000
MAX_HISTORY = 12000
RECENT_WINDOW = 6


def clip(text, limit=MAX_TOOL_OUTPUT):
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def middle(text, limit):
    text = str(text).replace("\n", " ")
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    left = (limit - 3) // 2
    right = limit - 3 - left
    return text[:left] + "..." + text[-right:]


def char_count(messages):
    # Character budget stands in for tokens, as in the original MAX_HISTORY.
    total = 0
    for message in messages:
        total += len(str(message.content))
        if isinstance(message, AIMessage) and message.tool_calls:
            total += len(json.dumps([call["args"] for call in message.tool_calls], default=str))
    return total


def _clip_args(tool_calls, limit):
    return [
        {**call, "args": {k: clip(v, limit) if isinstance(v, str) else v for k, v in call["args"].items()}}
        for call in tool_calls
    ]


def compress_history(messages, recent_window=RECENT_WINDOW, budget=MAX_HISTORY):
    call_args = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                call_args[call["id"]] = call["args"]

    recent_start = max(0, len(messages) - recent_window)
    seen_reads = set()
    out = []
    for index, message in enumerate(messages):
        recent = index >= recent_start
        update = {}
        if isinstance(message, ToolMessage):
            path = str(call_args.get(message.tool_call_id, {}).get("path", ""))
            if message.name in ("write_file", "patch_file"):
                seen_reads.discard(path)
            if message.name == "read_file" and not recent:
                if path in seen_reads:
                    # Replace rather than drop, so every tool call keeps its reply.
                    out.append(message.model_copy(update={"content": f"[duplicate read of {path} omitted]"}))
                    continue
                seen_reads.add(path)
            update["content"] = clip(message.content, 900 if recent else 180)
        else:
            update["content"] = clip(message.content, 900 if recent else 220)
            if isinstance(message, AIMessage) and message.tool_calls and not recent:
                update["tool_calls"] = _clip_args(message.tool_calls, 220)
        out.append(message.model_copy(update=update))

    trimmed = trim_messages(
        out,
        max_tokens=budget,
        token_counter=char_count,
        strategy="last",
        start_on="human",
        include_system=False,
        allow_partial=False,
    )
    if trimmed:
        return trimmed
    humans = [i for i, m in enumerate(out) if isinstance(m, HumanMessage)]
    return out[humans[-1]:] if humans else out[-1:]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_context.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/langchain_mini_agent/context.py tests/test_context.py
git commit -m "feat(context): clip, dedupe, and budget-trim message history"
```

---

### Task 3: Repo reading (`workspace.py`)

**Files:**
- Create: `src/langchain_mini_agent/workspace.py`
- Test: `tests/test_workspace.py`

**Interfaces:**
- Consumes: `clip` from `context.py`
- Produces:
  - `DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")`
  - `@dataclass class WorkspaceContext(cwd: str, repo_root: str, branch: str, default_branch: str, status: str, recent_commits: list[str], project_docs: dict[str, str])`
  - `WorkspaceContext.build(cwd) -> WorkspaceContext`
  - `WorkspaceContext.text() -> str`

- [ ] **Step 1: Write the failing tests**

`tests/test_workspace.py`:

```python
import subprocess

from langchain_mini_agent.workspace import WorkspaceContext


def git(tmp_path, *args):
    subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)


def test_build_outside_git_uses_fallbacks(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    ws = WorkspaceContext.build(tmp_path)
    assert ws.repo_root == str(tmp_path.resolve())
    assert ws.branch == "-"
    assert ws.status == "clean"
    assert ws.recent_commits == []
    assert ws.project_docs == {"README.md": "demo\n"}


def test_long_docs_are_clipped(tmp_path):
    (tmp_path / "README.md").write_text("x" * 2000, encoding="utf-8")
    ws = WorkspaceContext.build(tmp_path)
    assert ws.project_docs["README.md"].endswith("[truncated 800 chars]")


def test_build_reads_git_state(tmp_path):
    git(tmp_path, "init", "-b", "main")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "first")
    (tmp_path / "new.txt").write_text("x", encoding="utf-8")

    ws = WorkspaceContext.build(tmp_path)

    assert ws.branch == "main"
    assert "?? new.txt" in ws.status
    assert ws.recent_commits[0].endswith("first")


def test_subdirectory_docs_are_keyed_relative_to_repo_root(tmp_path):
    git(tmp_path, "init", "-b", "main")
    (tmp_path / "README.md").write_text("root\n", encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "README.md").write_text("pkg\n", encoding="utf-8")
    ws = WorkspaceContext.build(tmp_path / "pkg")
    assert ws.project_docs == {"README.md": "root\n", "pkg/README.md": "pkg\n"}


def test_text_lists_all_sections(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    text = WorkspaceContext.build(tmp_path).text()
    for label in ("Workspace:", "- cwd:", "- repo_root:", "- branch:", "- status:", "- recent_commits:", "- project_docs:", "- README.md"):
        assert label in text
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_workspace.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/langchain_mini_agent/workspace.py`:

```python
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .context import clip

DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")


@dataclass
class WorkspaceContext:
    cwd: str
    repo_root: str
    branch: str
    default_branch: str
    status: str
    recent_commits: list[str]
    project_docs: dict[str, str]

    @classmethod
    def build(cls, cwd):
        cwd = Path(cwd).resolve()

        def git(args, fallback=""):
            try:
                result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=5)
                return result.stdout.strip() or fallback
            except Exception:
                return fallback

        repo_root = Path(git(["rev-parse", "--show-toplevel"], str(cwd))).resolve()
        docs = {}
        for base in (repo_root, cwd):
            for name in DOC_NAMES:
                path = base / name
                if not path.exists():
                    continue
                key = str(path.relative_to(repo_root))
                if key not in docs:
                    docs[key] = clip(path.read_text(encoding="utf-8", errors="replace"), 1200)

        default_branch = git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], "origin/main") or "origin/main"
        return cls(
            cwd=str(cwd),
            repo_root=str(repo_root),
            branch=git(["branch", "--show-current"], "-") or "-",
            default_branch=default_branch.removeprefix("origin/"),
            status=clip(git(["status", "--short"], "clean") or "clean", 1500),
            recent_commits=[line for line in git(["log", "--oneline", "-5"]).splitlines() if line],
            project_docs=docs,
        )

    def text(self):
        commits = "\n".join(f"- {line}" for line in self.recent_commits) or "- none"
        docs = "\n".join(f"- {path}\n{snippet}" for path, snippet in self.project_docs.items()) or "- none"
        return "\n".join([
            "Workspace:",
            f"- cwd: {self.cwd}",
            f"- repo_root: {self.repo_root}",
            f"- branch: {self.branch}",
            f"- default_branch: {self.default_branch}",
            "- status:",
            self.status,
            "- recent_commits:",
            commits,
            "- project_docs:",
            docs,
        ])
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_workspace.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/langchain_mini_agent/workspace.py tests/test_workspace.py
git commit -m "feat(workspace): snapshot git state and project docs"
```

---

### Task 4: Tools and validation (`tools.py`)

**Files:**
- Create: `src/langchain_mini_agent/tools.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `IGNORED_PATH_NAMES: set[str]`, `RISKY_TOOLS: frozenset[str]`, `TOOL_EXAMPLES: dict[str, dict]`
  - Pydantic schemas `ListFilesArgs, ReadFileArgs, SearchArgs, RunShellArgs, WriteFileArgs, PatchFileArgs`
  - `class WorkspaceTools(root)`:
    - `.root: Path`, `.tools: dict[str, StructuredTool]` (order: list_files, read_file, search, run_shell, write_file, patch_file)
    - `.langchain_tools() -> list[StructuredTool]`
    - `.is_risky(name: str) -> bool`
    - `.path(raw_path) -> Path` (raises `ToolException` on escape)
    - `.validate(name: str, args: dict) -> dict` (raises `pydantic.ValidationError` or `ToolException`; returns normalized args)
    - `.run(name: str, args: dict) -> str` (raises on tool failure)

- [ ] **Step 1: Write the failing tests**

`tests/test_tools.py`:

```python
import pytest
from langchain_core.tools import ToolException
from pydantic import ValidationError

from langchain_mini_agent import tools as tools_module
from langchain_mini_agent.tools import WorkspaceTools


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / ".git").mkdir()
    return WorkspaceTools(tmp_path)


def test_tool_names_and_order(ws):
    assert [t.name for t in ws.langchain_tools()] == [
        "list_files", "read_file", "search", "run_shell", "write_file", "patch_file",
    ]


def test_risky_tools(ws):
    assert [n for n in ws.tools if ws.is_risky(n)] == ["run_shell", "write_file", "patch_file"]


def test_path_rejects_parent_escape(ws):
    with pytest.raises(ToolException, match="escapes workspace"):
        ws.path("../outside.txt")


def test_path_rejects_symlink_escape(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("s", encoding="utf-8")
    (root / "link.txt").symlink_to(tmp_path / "secret.txt")
    with pytest.raises(ToolException, match="escapes workspace"):
        WorkspaceTools(root).validate("read_file", {"path": "link.txt"})


def test_list_files_hides_ignored_and_marks_kinds(ws):
    assert ws.run("list_files", {}) == "[D] src\n[F] hello.txt"


def test_read_file_numbers_lines(ws):
    assert ws.run("read_file", {"path": "hello.txt", "start": 2, "end": 2}) == "# hello.txt\n   2: beta"


def test_validate_fills_defaults(ws):
    assert ws.validate("read_file", {"path": "hello.txt"}) == {"path": "hello.txt", "start": 1, "end": 200}


def test_validate_rejects_bad_range(ws):
    with pytest.raises(ValidationError, match="invalid line range"):
        ws.validate("read_file", {"path": "hello.txt", "start": 5, "end": 1})


def test_validate_rejects_missing_file(ws):
    with pytest.raises(ToolException, match="not a file"):
        ws.validate("read_file", {"path": "nope.txt"})


def test_validate_rejects_blank_command_and_bad_timeout(ws):
    with pytest.raises(ValidationError):
        ws.validate("run_shell", {"command": "   "})
    with pytest.raises(ValidationError):
        ws.validate("run_shell", {"command": "ls", "timeout": 121})


def test_validate_patch_requires_single_occurrence(ws):
    with pytest.raises(ToolException, match="exactly once, found 0"):
        ws.validate("patch_file", {"path": "hello.txt", "old_text": "gamma", "new_text": "x"})


def test_validate_write_rejects_directory(ws):
    with pytest.raises(ToolException, match="is a directory"):
        ws.validate("write_file", {"path": "src", "content": "x"})


def test_write_file_creates_parents(ws, tmp_path):
    assert ws.run("write_file", {"path": "a/b/c.py", "content": "x = 1\n"}) == "wrote a/b/c.py (6 chars)"
    assert (tmp_path / "a/b/c.py").read_text(encoding="utf-8") == "x = 1\n"


def test_patch_file_replaces_once(ws, tmp_path):
    assert ws.run("patch_file", {"path": "hello.txt", "old_text": "beta", "new_text": "gamma"}) == "patched hello.txt"
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "alpha\ngamma\n"


def test_run_shell_formats_output(ws):
    assert ws.run("run_shell", {"command": "echo hi"}) == "exit_code: 0\nstdout:\nhi\nstderr:\n(empty)"


def test_search_finds_matches(ws):
    assert "hello.txt:1:alpha" in ws.run("search", {"pattern": "alpha"})


def test_search_fallback_without_rg(ws, monkeypatch):
    monkeypatch.setattr(tools_module.shutil, "which", lambda name: None)
    assert ws.run("search", {"pattern": "BETA"}) == "hello.txt:2:beta"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/langchain_mini_agent/tools.py`:

```python
import shutil
import subprocess
from pathlib import Path
from typing import Annotated

from langchain_core.tools import StructuredTool, ToolException
from pydantic import AfterValidator, BaseModel, Field, model_validator

IGNORED_PATH_NAMES = {".git", ".mini-coding-agent", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv"}
RISKY_TOOLS = frozenset({"run_shell", "write_file", "patch_file"})
TOOL_EXAMPLES = {
    "list_files": {"path": "."},
    "read_file": {"path": "README.md", "start": 1, "end": 80},
    "search": {"pattern": "binary_search", "path": "."},
    "run_shell": {"command": "uv run --with pytest python -m pytest -q", "timeout": 20},
    "write_file": {"path": "binary_search.py", "content": "def binary_search(nums, target):\n    return -1\n"},
    "patch_file": {"path": "binary_search.py", "old_text": "return -1", "new_text": "return mid"},
}


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be empty")
    return value


NonBlank = Annotated[str, AfterValidator(_non_blank)]


class ListFilesArgs(BaseModel):
    path: str = Field(".", description="Directory relative to the repo root.")


class ReadFileArgs(BaseModel):
    path: NonBlank = Field(description="File relative to the repo root.")
    start: int = Field(1, ge=1, description="First line, 1-based.")
    end: int = Field(200, ge=1, description="Last line, inclusive.")

    @model_validator(mode="after")
    def _check_range(self):
        if self.end < self.start:
            raise ValueError("invalid line range")
        return self


class SearchArgs(BaseModel):
    pattern: NonBlank = Field(description="Text or regex to search for.")
    path: str = Field(".", description="File or directory to search.")


class RunShellArgs(BaseModel):
    command: NonBlank = Field(description="Shell command, run in the repo root.")
    timeout: int = Field(20, ge=1, le=120, description="Seconds before the command is killed.")


class WriteFileArgs(BaseModel):
    path: NonBlank = Field(description="File to create or overwrite.")
    content: str = Field(description="Full file contents.")


class PatchFileArgs(BaseModel):
    path: NonBlank = Field(description="File to patch.")
    old_text: NonBlank = Field(description="Exact text that occurs exactly once in the file.")
    new_text: str = Field(description="Replacement text.")


SPECS = (
    ("list_files", ListFilesArgs, "List files in the workspace."),
    ("read_file", ReadFileArgs, "Read a UTF-8 file by line range."),
    ("search", SearchArgs, "Search the workspace with rg or a simple fallback."),
    ("run_shell", RunShellArgs, "Run a shell command in the repo root."),
    ("write_file", WriteFileArgs, "Write a text file."),
    ("patch_file", PatchFileArgs, "Replace one exact text block in a file."),
)


class WorkspaceTools:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.tools = {
            name: StructuredTool.from_function(func=getattr(self, name), name=name, description=description, args_schema=schema)
            for name, schema, description in SPECS
        }

    def langchain_tools(self):
        return list(self.tools.values())

    def is_risky(self, name):
        return name in RISKY_TOOLS

    def validate(self, name, args):
        values = self.tools[name].args_schema.model_validate(args or {}).model_dump()
        getattr(self, f"_check_{name}")(**values)
        return values

    def run(self, name, args):
        return str(self.tools[name].invoke(args))

    # --- sandbox ---
    def path(self, raw_path):
        path = Path(raw_path)
        resolved = (path if path.is_absolute() else self.root / path).resolve()
        if not self._within_root(resolved):
            raise ToolException(f"path escapes workspace: {raw_path}")
        return resolved

    def _within_root(self, resolved):
        probe = resolved
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        for candidate in (probe, *probe.parents):
            try:
                if candidate.samefile(self.root):
                    return True
            except OSError:
                continue
        return False

    def _dir(self, raw_path):
        path = self.path(raw_path)
        if not path.is_dir():
            raise ToolException("path is not a directory")
        return path

    def _file(self, raw_path):
        path = self.path(raw_path)
        if not path.is_file():
            raise ToolException("path is not a file")
        return path

    # --- pre-approval checks (no side effects) ---
    def _check_list_files(self, path):
        self._dir(path)

    def _check_read_file(self, path, start, end):
        self._file(path)

    def _check_search(self, pattern, path):
        self.path(path)

    def _check_run_shell(self, command, timeout):
        pass

    def _check_write_file(self, path, content):
        if self.path(path).is_dir():
            raise ToolException("path is a directory")

    def _check_patch_file(self, path, old_text, new_text):
        count = self._file(path).read_text(encoding="utf-8").count(old_text)
        if count != 1:
            raise ToolException(f"old_text must occur exactly once, found {count}")

    # --- tool bodies ---
    def list_files(self, path="."):
        directory = self._dir(path)
        entries = [
            item for item in sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name.lower()))
            if item.name not in IGNORED_PATH_NAMES
        ]
        lines = [f"{'[D]' if e.is_dir() else '[F]'} {e.relative_to(self.root)}" for e in entries[:200]]
        return "\n".join(lines) or "(empty)"

    def read_file(self, path, start=1, end=200):
        file_path = self._file(path)
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        body = "\n".join(f"{n:>4}: {line}" for n, line in enumerate(lines[start - 1:end], start=start))
        return f"# {file_path.relative_to(self.root)}\n{body}"

    def search(self, pattern, path="."):
        target = self.path(path)
        if shutil.which("rg"):
            result = subprocess.run(
                ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(target.relative_to(self.root))],
                cwd=self.root, capture_output=True, text=True,
            )
            return result.stdout.strip() or result.stderr.strip() or "(no matches)"
        matches = []
        files = [target] if target.is_file() else [
            item for item in target.rglob("*")
            if item.is_file() and not any(part in IGNORED_PATH_NAMES for part in item.relative_to(self.root).parts)
        ]
        for file_path in files:
            for number, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if pattern.lower() in line.lower():
                    matches.append(f"{file_path.relative_to(self.root)}:{number}:{line}")
                    if len(matches) >= 200:
                        return "\n".join(matches)
        return "\n".join(matches) or "(no matches)"

    def run_shell(self, command, timeout=20):
        result = subprocess.run(command.strip(), cwd=self.root, shell=True, capture_output=True, text=True, timeout=timeout)
        return "\n".join([
            f"exit_code: {result.returncode}",
            "stdout:",
            result.stdout.strip() or "(empty)",
            "stderr:",
            result.stderr.strip() or "(empty)",
        ])

    def write_file(self, path, content):
        file_path = self.path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"wrote {file_path.relative_to(self.root)} ({len(content)} chars)"

    def patch_file(self, path, old_text, new_text):
        self._check_patch_file(path, old_text, new_text)
        file_path = self.path(path)
        text = file_path.read_text(encoding="utf-8")
        file_path.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"patched {file_path.relative_to(self.root)}"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_tools.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/langchain_mini_agent/tools.py tests/test_tools.py
git commit -m "feat(tools): StructuredTools with pydantic validation and sandbox"
```

---

### Task 5: Short- and long-term memory (`memory.py`)

**Files:**
- Create: `src/langchain_mini_agent/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `clip` from `context.py`
- Produces:
  - `class WorkingMemory(TypedDict): task: str; files: list[str]; notes: list[str]`
  - `empty_memory() -> WorkingMemory`
  - `remember(bucket: list[str], item: str, limit: int) -> list[str]` (pure)
  - `with_task(memory, request: str) -> WorkingMemory`
  - `note_tool(memory, name: str, args: dict, result: str) -> WorkingMemory`
  - `note_final(memory, final: str) -> WorkingMemory`
  - `memory_text(memory, long_term_notes: list[str] = ()) -> str`
  - `class LongTermMemory(store: BaseStore, repo_root: str)`: `.notes() -> list[str]`, `.add_note(note: str)`, `.touch_session(thread_id: str)`, `.latest_session() -> str | None`
  - `STATE_DIR = ".mini-coding-agent"`
  - `open_persistence(repo_root) -> tuple[SqliteSaver, SqliteStore, Path]`

- [ ] **Step 1: Write the failing tests**

`tests/test_memory.py`:

```python
from langgraph.store.memory import InMemoryStore

from langchain_mini_agent.memory import (
    LongTermMemory, empty_memory, memory_text, note_final, note_tool, open_persistence, remember, with_task,
)


def test_remember_moves_duplicates_to_end_and_caps():
    assert remember(["a", "b", "c"], "a", 3) == ["b", "c", "a"]
    assert remember(["a", "b", "c"], "d", 3) == ["b", "c", "d"]
    assert remember(["a"], "", 3) == ["a"]


def test_with_task_sets_once_and_clips():
    memory = with_task(empty_memory(), "  " + "t" * 400)
    assert memory["task"].startswith("t" * 300)
    assert with_task(memory, "other")["task"] == memory["task"]


def test_note_tool_tracks_files_and_notes():
    memory = note_tool(empty_memory(), "read_file", {"path": "a.py"}, "line1\nline2")
    assert memory["files"] == ["a.py"]
    assert memory["notes"] == ["read_file: line1 line2"]
    memory = note_tool(memory, "run_shell", {"command": "ls"}, "ok")
    assert memory["files"] == ["a.py"]


def test_note_final_caps_notes_at_five():
    memory = empty_memory()
    for i in range(7):
        memory = note_final(memory, f"answer {i}")
    assert memory["notes"] == [f"answer {i}" for i in range(2, 7)]


def test_memory_text_includes_long_term_notes():
    text = memory_text(with_task(empty_memory(), "fix bug"), ["prefers pytest"])
    assert "- task: fix bug" in text
    assert "- files: -" in text
    assert "- long_term_notes:\n- prefers pytest" in text


def test_long_term_notes_are_capped_and_scoped_per_repo():
    store = InMemoryStore()
    repo_a = LongTermMemory(store, "/repo/a")
    for i in range(12):
        repo_a.add_note(f"n{i}")
    assert repo_a.notes() == [f"n{i}" for i in range(2, 12)]
    assert LongTermMemory(store, "/repo/b").notes() == []


def test_latest_session_round_trip():
    lt = LongTermMemory(InMemoryStore(), "/repo/a")
    assert lt.latest_session() is None
    lt.touch_session("s1")
    lt.touch_session("s2")
    assert lt.latest_session() == "s2"


def test_open_persistence_survives_reopen(tmp_path):
    _, store, db_path = open_persistence(tmp_path)
    LongTermMemory(store, str(tmp_path)).add_note("kept")
    assert db_path == tmp_path / ".mini-coding-agent" / "state.sqlite"
    _, reopened, _ = open_persistence(tmp_path)
    assert LongTermMemory(reopened, str(tmp_path)).notes() == ["kept"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_memory.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/langchain_mini_agent/memory.py`:

```python
import hashlib
import sqlite3
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from .context import clip

STATE_DIR = ".mini-coding-agent"


###########################################
#### Short-term: checkpointed per thread ###
###########################################
class WorkingMemory(TypedDict):
    task: str
    files: list[str]
    notes: list[str]


def empty_memory():
    return {"task": "", "files": [], "notes": []}


def remember(bucket, item, limit):
    if not item:
        return list(bucket)
    kept = [existing for existing in bucket if existing != item] + [item]
    return kept[-limit:]


def with_task(memory, request):
    if memory["task"]:
        return memory
    return {**memory, "task": clip(request.strip(), 300)}


def note_tool(memory, name, args, result):
    files = memory["files"]
    path = args.get("path") if isinstance(args, dict) else None
    if name in {"read_file", "write_file", "patch_file"} and path:
        files = remember(files, str(path), 8)
    note = f"{name}: {clip(str(result).replace(chr(10), ' '), 220)}"
    return {**memory, "files": files, "notes": remember(memory["notes"], note, 5)}


def note_final(memory, final):
    return {**memory, "notes": remember(memory["notes"], clip(final, 220), 5)}


def memory_text(memory, long_term_notes=()):
    notes = "\n".join(f"- {note}" for note in memory["notes"]) or "- none"
    long_term = "\n".join(f"- {note}" for note in long_term_notes) or "- none"
    return "\n".join([
        "Memory:",
        f"- task: {memory['task'] or '-'}",
        f"- files: {', '.join(memory['files']) or '-'}",
        "- notes:",
        notes,
        "- long_term_notes:",
        long_term,
    ])


####################################################
#### Long-term: LangGraph store, shared per repo ###
####################################################
class LongTermMemory:
    NOTE_LIMIT = 10

    def __init__(self, store, repo_root):
        self.store = store
        # Store namespace labels may not contain "."; hash the path.
        self.namespace = ("repos", hashlib.sha1(str(repo_root).encode("utf-8")).hexdigest()[:12])

    def notes(self):
        item = self.store.get(self.namespace, "notes")
        return list(item.value["notes"]) if item else []

    def add_note(self, note):
        self.store.put(self.namespace, "notes", {"notes": remember(self.notes(), note, self.NOTE_LIMIT)})

    def touch_session(self, thread_id):
        self.store.put(self.namespace, "latest_session", {"thread_id": thread_id})

    def latest_session(self):
        item = self.store.get(self.namespace, "latest_session")
        return item.value["thread_id"] if item else None


def open_persistence(repo_root):
    db_path = Path(repo_root) / STATE_DIR / "state.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))
    store = SqliteStore(sqlite3.connect(db_path, check_same_thread=False, isolation_level=None))
    store.setup()
    return checkpointer, store, db_path
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_memory.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/langchain_mini_agent/memory.py tests/test_memory.py
git commit -m "feat(memory): working memory, long-term store, sqlite persistence"
```

---

### Task 6: Stable prompt prefix (`prompts.py`)

**Files:**
- Create: `src/langchain_mini_agent/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `WorkspaceTools` (Task 4), `WorkspaceContext` (Task 3)
- Produces:
  - `RULES: tuple[str, ...]`
  - `PROMPT: ChatPromptTemplate` with variables `prefix`, `memory`, `history`
  - `build_prefix(tools: WorkspaceTools, workspace: WorkspaceContext) -> str`
  - `render_prompt(prefix: str, memory: str, history: list[BaseMessage]) -> list[BaseMessage]` → `[SystemMessage(prefix), HumanMessage(memory), *history]`

Why this ordering: Ollama reuses its KV cache for the longest identical token prefix. The chat template renders the system message and the bound tool schemas first, so keeping `prefix` byte-identical for the whole session means that block is only evaluated once. Memory changes every step, so it goes *after* the prefix (same order as the original `prompt()`).

- [ ] **Step 1: Write the failing tests**

`tests/test_prompts.py`:

```python
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_mini_agent.prompts import build_prefix, render_prompt
from langchain_mini_agent.tools import WorkspaceTools
from langchain_mini_agent.workspace import WorkspaceContext


def make(tmp_path, readme="demo\n"):
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    return WorkspaceTools(tmp_path), WorkspaceContext.build(tmp_path)


def test_prefix_lists_rules_tools_and_workspace(tmp_path):
    prefix = build_prefix(*make(tmp_path))
    assert "Rules:\n- Use tools instead of guessing" in prefix
    assert "- read_file [safe] Read a UTF-8 file by line range." in prefix
    assert "- write_file [approval required] Write a text file." in prefix
    assert "Workspace:" in prefix and "demo" in prefix


def test_prefix_is_deterministic(tmp_path):
    tools, workspace = make(tmp_path)
    assert build_prefix(tools, workspace) == build_prefix(tools, workspace)


def test_render_prompt_orders_prefix_memory_history(tmp_path):
    messages = render_prompt("PREFIX", "Memory: x", [HumanMessage("hi")])
    assert [type(m) for m in messages] == [SystemMessage, HumanMessage, HumanMessage]
    assert [m.content for m in messages] == ["PREFIX", "Memory: x", "hi"]


def test_braces_in_repo_docs_do_not_break_templating(tmp_path):
    prefix = build_prefix(*make(tmp_path, readme='{"name": "{project}"}\n'))
    messages = render_prompt(prefix, "Memory: {x}", [])
    assert messages[0].content == prefix
    assert messages[1].content == "Memory: {x}"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`src/langchain_mini_agent/prompts.py`:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

RULES = (
    "Use tools instead of guessing about the workspace.",
    "Call at most one tool per reply. When you are done, reply with plain text and no tool call; that text is your final answer.",
    "Never invent tool results.",
    "Keep answers concise and concrete.",
    "If the user asks you to create or update a specific file and the path is clear, use write_file or patch_file instead of repeatedly listing files.",
    "Before writing tests for existing code, read the implementation first.",
    "When writing tests, match the current implementation unless the user explicitly asked you to change the code.",
    "New files should be complete and runnable, including obvious imports.",
    "Do not repeat the same tool call with the same arguments if it did not help. Choose a different tool or give a final answer.",
    "Required tool arguments must not be empty.",
)

# prefix and memory are passed as variable *values*, so braces inside them are never re-parsed.
PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{prefix}"),
    ("human", "{memory}"),
    MessagesPlaceholder("history"),
])


def build_prefix(tools, workspace):
    tool_lines = "\n".join(
        f"- {tool.name} [{'approval required' if tools.is_risky(tool.name) else 'safe'}] {tool.description}"
        for tool in tools.langchain_tools()
    )
    return "\n\n".join([
        "You are Mini-Coding-Agent, a small local coding agent running through Ollama and LangGraph.",
        "Rules:\n" + "\n".join(f"- {rule}" for rule in RULES),
        "Tools:\n" + tool_lines,
        workspace.text(),
    ])


def render_prompt(prefix, memory, history):
    return PROMPT.invoke({"prefix": prefix, "memory": memory, "history": history}).to_messages()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/langchain_mini_agent/prompts.py tests/test_prompts.py
git commit -m "feat(prompts): cache-stable system prefix via ChatPromptTemplate"
```

---

### Task 7: Ollama chat model and test double (`model.py`, `tests/fakes.py`)

**Files:**
- Create: `src/langchain_mini_agent/model.py`, `tests/fakes.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `DEFAULT_MODEL = "qwen3.5:4b"`, `DEFAULT_HOST = "http://127.0.0.1:11434"`
  - `build_chat_model(model=DEFAULT_MODEL, host=DEFAULT_HOST, temperature=0.2, top_p=0.9, max_new_tokens=512, num_ctx=16384, timeout=300, keep_alive="30m") -> ChatOllama`
  - `describe_model_error(exc: Exception, host: str, model: str) -> str`
  - tests only: `ScriptedChatModel(responses: list[AIMessage])` with `.received: list[list[BaseMessage]]`, `.bound_tool_names: list[list[str]]`; helpers `call(name, **args) -> AIMessage`, `final(text) -> AIMessage`

- [ ] **Step 1: Write the test double**

`tests/fakes.py`:

```python
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedChatModel(BaseChatModel):
    """Returns canned AIMessages in order and records every prompt it receives."""

    responses: list = Field(default_factory=list)
    received: list = Field(default_factory=list)
    bound_tool_names: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names.append([tool.name for tool in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(list(messages))
        if not self.responses:
            raise RuntimeError("scripted model ran out of responses")
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid4().hex[:8]}"}])


def final(text):
    return AIMessage(content=text)
```

- [ ] **Step 2: Write the failing tests**

`tests/test_model.py`:

```python
from fakes import ScriptedChatModel, call, final
from langchain_core.messages import HumanMessage

from langchain_mini_agent.model import build_chat_model, describe_model_error


def test_build_chat_model_sets_context_and_cache_options():
    model = build_chat_model(model="m", host="http://h:1", max_new_tokens=64, num_ctx=8192)
    assert model.model == "m"
    assert model.base_url == "http://h:1"
    assert model.num_predict == 64
    assert model.num_ctx == 8192
    assert model.keep_alive == "30m"
    assert model.reasoning is False
    assert model.temperature == 0.2 and model.top_p == 0.9


def test_describe_model_error_mentions_host_and_model():
    text = describe_model_error(ConnectionError("refused"), "http://h:1", "m")
    assert "ollama serve" in text and "http://h:1" in text and "Model: m" in text and "refused" in text


def test_scripted_model_replays_and_records():
    model = ScriptedChatModel(responses=[call("list_files", path="."), final("done")])
    bound = model.bind_tools([])
    assert bound.invoke([HumanMessage("a")]).tool_calls[0]["name"] == "list_files"
    assert bound.invoke([HumanMessage("b")]).content == "done"
    assert [m[0].content for m in model.received] == ["a", "b"]
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langchain_mini_agent.model'`.

- [ ] **Step 4: Implement**

`src/langchain_mini_agent/model.py`:

```python
from langchain_ollama import ChatOllama

DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_HOST = "http://127.0.0.1:11434"


def build_chat_model(
    model=DEFAULT_MODEL,
    host=DEFAULT_HOST,
    temperature=0.2,
    top_p=0.9,
    max_new_tokens=512,
    num_ctx=16384,
    timeout=300,
    keep_alive="30m",
):
    # num_ctx: Ollama's default window is small; a truncated prompt loses the
    # system prefix and invalidates the prefix cache.
    # keep_alive: keeps the model (and its KV cache) loaded between steps.
    return ChatOllama(
        model=model,
        base_url=host,
        temperature=temperature,
        top_p=top_p,
        num_predict=max_new_tokens,
        num_ctx=num_ctx,
        keep_alive=keep_alive,
        reasoning=False,
        client_kwargs={"timeout": timeout},
    )


def describe_model_error(exc, host, model):
    return "\n".join([
        "Could not get a response from Ollama.",
        "Make sure `ollama serve` is running and the model is available.",
        f"Host: {host}",
        f"Model: {model}",
        f"Error: {exc}",
    ])
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_model.py -v`
Expected: all PASS. If `reasoning` is not a `ChatOllama` field in the installed version, check `uv run python -c "from langchain_ollama import ChatOllama; print(ChatOllama.model_fields.keys())"` and use the field that maps to Ollama's `think` option.

- [ ] **Step 6: Commit**

```bash
git add src/langchain_mini_agent/model.py tests/fakes.py tests/test_model.py
git commit -m "feat(model): ChatOllama factory and scripted test model"
```

---

### Task 8: Agent graph and facade (`graph.py`, `agent.py`)

**Files:**
- Create: `src/langchain_mini_agent/graph.py`, `src/langchain_mini_agent/agent.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `clip`, `compress_history` (T2); `WorkspaceContext` (T3); `WorkspaceTools`, `TOOL_EXAMPLES` (T4); `WorkingMemory`, `LongTermMemory`, `empty_memory`, `with_task`, `note_tool`, `note_final`, `memory_text` (T5); `build_prefix`, `render_prompt` (T6)
- Produces:
  - `graph.py`: `AgentSettings(approval_policy="ask", max_steps=6, read_only=False)` with `.max_attempts`, `.recursion_limit`; `AgentState`; `STEP_LIMIT_MESSAGE`, `MALFORMED_MESSAGE`, `EXTRA_CALL_ERROR`; `retry_notice(problem) -> str`; `repeated_tool_call(history, name, args) -> bool`; `build_graph(model, workspace, tools, long_term, settings, checkpointer) -> CompiledStateGraph`
  - `agent.py`: `Approver = Callable[[str, dict], bool]`; `deny_all`; `new_thread_id() -> str`; `class Agent` with `.thread_id`, `.workspace`, `.settings`, `.config`, `.ask(text) -> str`, `.state() -> dict`, `.memory() -> WorkingMemory`, `.memory_text() -> str`, `.reset()`; `create_agent(model, workspace, checkpointer, store, settings=AgentSettings(), thread_id=None, approver=deny_all) -> Agent`

- [ ] **Step 1: Write the failing tests**

`tests/test_graph.py`:

```python
from fakes import ScriptedChatModel, call, final
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import invalid_tool_call
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from langchain_mini_agent.agent import create_agent
from langchain_mini_agent.graph import MALFORMED_MESSAGE, STEP_LIMIT_MESSAGE, AgentSettings
from langchain_mini_agent.workspace import WorkspaceContext

TOOL_NAMES = ["list_files", "read_file", "search", "run_shell", "write_file", "patch_file"]


def make_agent(tmp_path, responses, store=None, checkpointer=None, thread_id=None, approver=None, **settings):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    model = ScriptedChatModel(responses=list(responses))
    kwargs = {"approver": approver} if approver else {}
    agent = create_agent(
        model=model,
        workspace=WorkspaceContext.build(tmp_path),
        checkpointer=checkpointer or InMemorySaver(),
        store=store or InMemoryStore(),
        settings=AgentSettings(**{"approval_policy": "auto", **settings}),
        thread_id=thread_id,
        **kwargs,
    )
    return agent, model


def tool_messages(agent):
    return [m for m in agent.state()["messages"] if isinstance(m, ToolMessage)]


def test_runs_tool_then_final(tmp_path):
    (tmp_path / "hello.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    agent, _ = make_agent(tmp_path, [call("read_file", path="hello.txt", start=1, end=2), final("Read it.")])
    assert agent.ask("Inspect hello.txt") == "Read it."
    assert "1: alpha" in tool_messages(agent)[0].content
    assert agent.memory()["files"] == ["hello.txt"]
    assert agent.memory()["task"] == "Inspect hello.txt"


def test_retries_after_empty_output(tmp_path):
    agent, _ = make_agent(tmp_path, [AIMessage(""), final("Recovered.")])
    assert agent.ask("Do it") == "Recovered."
    notices = [m.content for m in agent.state()["messages"] if isinstance(m, HumanMessage)]
    assert any("empty response" in n for n in notices)


def test_retries_after_invalid_tool_call(tmp_path):
    bad = AIMessage("", invalid_tool_calls=[invalid_tool_call(name="read_file", args="bad", id="x", error="bad json")])
    agent, _ = make_agent(tmp_path, [bad, final("Recovered.")])
    assert agent.ask("Do it") == "Recovered."
    assert any("malformed tool call" in m.content for m in agent.state()["messages"] if isinstance(m, HumanMessage))


def test_unknown_tool_reports_error(tmp_path):
    agent, _ = make_agent(tmp_path, [call("delete_everything"), final("ok")])
    agent.ask("x")
    assert tool_messages(agent)[0].content == "error: unknown tool 'delete_everything'"


def test_invalid_args_include_example(tmp_path):
    (tmp_path / "hello.txt").write_text("a\n", encoding="utf-8")
    agent, _ = make_agent(tmp_path, [call("read_file", path="hello.txt", start=5, end=1), final("ok")])
    agent.ask("x")
    content = tool_messages(agent)[0].content
    assert content.startswith("error: invalid arguments for read_file")
    assert "example:" in content


def test_third_identical_call_is_blocked(tmp_path):
    agent, _ = make_agent(tmp_path, [call("list_files", path="."), call("list_files", path="."), call("list_files", path="."), final("ok")])
    agent.ask("x")
    assert "repeated identical tool call" in tool_messages(agent)[2].content


def test_extra_tool_calls_get_error_replies(tmp_path):
    double = AIMessage("", tool_calls=[
        {"name": "list_files", "args": {}, "id": "a"},
        {"name": "list_files", "args": {"path": "."}, "id": "b"},
    ])
    agent, _ = make_agent(tmp_path, [double, final("ok")])
    agent.ask("x")
    replies = {m.tool_call_id: m.content for m in tool_messages(agent)}
    assert set(replies) == {"a", "b"}
    assert "only one tool call per step" in replies["b"]
    assert agent.state()["tool_steps"] == 1


def test_never_policy_denies_without_prompt(tmp_path):
    agent, _ = make_agent(tmp_path, [call("write_file", path="out.txt", content="hi"), final("ok")], approval_policy="never")
    agent.ask("x")
    assert tool_messages(agent)[0].content == "error: approval denied for write_file"
    assert not (tmp_path / "out.txt").exists()


def test_ask_policy_denial_blocks_write(tmp_path):
    seen = []
    agent, _ = make_agent(
        tmp_path, [call("write_file", path="out.txt", content="hi"), final("done")],
        approval_policy="ask", approver=lambda name, args: seen.append((name, args)) or False,
    )
    assert agent.ask("write it") == "done"
    assert seen == [("write_file", {"path": "out.txt", "content": "hi"})]
    assert not (tmp_path / "out.txt").exists()


def test_ask_policy_approval_runs_tool_once(tmp_path):
    agent, model = make_agent(
        tmp_path, [call("write_file", path="out.txt", content="hi"), final("done")],
        approval_policy="ask", approver=lambda name, args: True,
    )
    assert agent.ask("write it") == "done"
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "hi"
    assert agent.state()["tool_steps"] == 1
    assert len(model.received) == 2


def test_stops_at_step_limit(tmp_path):
    agent, model = make_agent(tmp_path, [call("list_files", path=".") for _ in range(6)])
    assert agent.ask("loop") == STEP_LIMIT_MESSAGE
    assert len(model.received) == 6


def test_stops_after_too_many_malformed_responses(tmp_path):
    agent, model = make_agent(tmp_path, [AIMessage("") for _ in range(18)])
    assert agent.ask("x") == MALFORMED_MESSAGE
    assert len(model.received) == 18


def test_resume_same_thread_keeps_history_and_memory(tmp_path):
    saver, store = InMemorySaver(), InMemoryStore()
    first, _ = make_agent(tmp_path, [final("one")], store=store, checkpointer=saver)
    first.ask("first task")
    second, model = make_agent(tmp_path, [final("two")], store=store, checkpointer=saver, thread_id=first.thread_id)
    second.ask("follow up")
    assert second.memory()["task"] == "first task"
    assert any(m.content == "first task" for m in model.received[0])


def test_long_term_notes_reach_new_sessions(tmp_path):
    store = InMemoryStore()
    first, _ = make_agent(tmp_path, [final("remember zebra")], store=store)
    first.ask("x")
    second, model = make_agent(tmp_path, [final("ok")], store=store)
    assert second.thread_id != first.thread_id
    second.ask("y")
    assert "remember zebra" in model.received[0][1].content


def test_system_prefix_is_identical_across_steps_and_turns(tmp_path):
    (tmp_path / "hello.txt").write_text("a\n", encoding="utf-8")
    agent, model = make_agent(tmp_path, [call("read_file", path="hello.txt"), final("one"), final("two")])
    agent.ask("zebra-task")
    agent.ask("second")
    prefixes = [messages[0] for messages in model.received]
    assert len(prefixes) == 3
    assert isinstance(prefixes[0], SystemMessage)
    assert all(p.content == prefixes[0].content for p in prefixes)
    assert "zebra-task" not in prefixes[0].content
    assert model.bound_tool_names == [TOOL_NAMES]


def test_reset_starts_new_thread_with_empty_memory(tmp_path):
    agent, _ = make_agent(tmp_path, [final("one")])
    agent.ask("x")
    old = agent.thread_id
    agent.reset()
    assert agent.thread_id != old
    assert agent.memory()["task"] == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langchain_mini_agent.agent'`.

- [ ] **Step 3: Implement `graph.py`**

`src/langchain_mini_agent/graph.py`:

```python
import json
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from .context import clip, compress_history
from .memory import WorkingMemory, empty_memory, memory_text, note_final, note_tool, with_task
from .prompts import build_prefix, render_prompt
from .tools import TOOL_EXAMPLES

STEP_LIMIT_MESSAGE = "Stopped after reaching the step limit without a final answer."
MALFORMED_MESSAGE = "Stopped after too many malformed model responses without a valid tool call or final answer."
EXTRA_CALL_ERROR = "error: only one tool call per step; this call was skipped"


@dataclass(frozen=True)
class AgentSettings:
    approval_policy: Literal["ask", "auto", "never"] = "ask"
    max_steps: int = 6
    read_only: bool = False

    @property
    def max_attempts(self):
        return max(self.max_steps * 3, self.max_steps + 4)

    @property
    def recursion_limit(self):
        # Each attempt costs up to two super-steps (agent + tools/retry), plus start and finish.
        return 2 * self.max_attempts + 5


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    memory: WorkingMemory
    tool_steps: int
    attempts: int
    final: str


def retry_notice(problem):
    return f"Runtime notice: {problem}. Reply with a valid tool call or a non-empty final answer."


def repeated_tool_call(history, name, args):
    previous = [m.tool_calls[0] for m in history if isinstance(m, AIMessage) and m.tool_calls]
    if len(previous) < 2:
        return False
    return all(c["name"] == name and c["args"] == args for c in previous[-2:])


def build_graph(model, workspace, tools, long_term, settings, checkpointer):
    bound_model = model.bind_tools(tools.langchain_tools())
    prefix = build_prefix(tools, workspace)  # built once per graph: the cacheable prefix

    def start_turn(state):
        request = str(state["messages"][-1].content)
        memory = with_task(state.get("memory") or empty_memory(), request)
        return {"memory": memory, "tool_steps": 0, "attempts": 0, "final": ""}

    def call_model(state):
        messages = render_prompt(
            prefix,
            memory_text(state["memory"], long_term.notes()),
            compress_history(state["messages"]),
        )
        return {"messages": [bound_model.invoke(messages)], "attempts": state["attempts"] + 1}

    def route_after_model(state):
        last = state["messages"][-1]
        if last.tool_calls:
            return "tools"
        if last.invalid_tool_calls or not str(last.content).strip():
            return "retry"
        return "finalize"

    def retry(state):
        last = state["messages"][-1]
        problem = "model returned a malformed tool call" if last.invalid_tool_calls else "model returned an empty response"
        return {"messages": [HumanMessage(retry_notice(problem))]}

    def approve(name, args):
        if settings.read_only or settings.approval_policy == "never":
            return False
        if settings.approval_policy == "auto":
            return True
        # Pauses the graph. On resume the whole tools node re-runs, so nothing
        # before this point may have side effects.
        return bool(interrupt({"tool": name, "args": args}))

    def execute(call, history):
        name, raw_args = call["name"], call["args"]
        if name not in tools.tools:
            return f"error: unknown tool '{name}'"
        try:
            args = tools.validate(name, raw_args)
        except Exception as exc:
            example = json.dumps({"name": name, "args": TOOL_EXAMPLES[name]})
            return f"error: invalid arguments for {name}: {exc}\nexample: {example}"
        if repeated_tool_call(history, name, raw_args):
            return f"error: repeated identical tool call for {name}; choose a different tool or return a final answer"
        if tools.is_risky(name) and not approve(name, args):
            return f"error: approval denied for {name}"
        try:
            return clip(tools.run(name, args))
        except Exception as exc:
            return f"error: tool {name} failed: {exc}"

    def run_tools(state):
        call, *extra = state["messages"][-1].tool_calls
        result = execute(call, state["messages"][:-1])
        replies = [ToolMessage(content=result, tool_call_id=call["id"], name=call["name"])]
        replies += [ToolMessage(content=EXTRA_CALL_ERROR, tool_call_id=c["id"], name=c["name"]) for c in extra]
        return {
            "messages": replies,
            "tool_steps": state["tool_steps"] + 1,
            "memory": note_tool(state["memory"], call["name"], call["args"], result),
        }

    def route_budget(state):
        if state["tool_steps"] >= settings.max_steps or state["attempts"] >= settings.max_attempts:
            return "stop"
        return "agent"

    def finalize(state):
        final = str(state["messages"][-1].content).strip()
        long_term.add_note(clip(final, 220))
        return {"final": final, "memory": note_final(state["memory"], final)}

    def stop(state):
        malformed = state["attempts"] >= settings.max_attempts and state["tool_steps"] < settings.max_steps
        final = MALFORMED_MESSAGE if malformed else STEP_LIMIT_MESSAGE
        return {"messages": [AIMessage(final)], "final": final}

    graph = StateGraph(AgentState)
    graph.add_node("start_turn", start_turn)
    graph.add_node("agent", call_model)
    graph.add_node("tools", run_tools)
    graph.add_node("retry", retry)
    graph.add_node("finalize", finalize)
    graph.add_node("stop", stop)
    graph.add_edge(START, "start_turn")
    graph.add_edge("start_turn", "agent")
    graph.add_conditional_edges("agent", route_after_model, ["tools", "retry", "finalize"])
    graph.add_conditional_edges("tools", route_budget, ["agent", "stop"])
    graph.add_conditional_edges("retry", route_budget, ["agent", "stop"])
    graph.add_edge("finalize", END)
    graph.add_edge("stop", END)
    return graph.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: Implement `agent.py`**

`src/langchain_mini_agent/agent.py`:

```python
import uuid
from datetime import datetime
from typing import Callable

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from .graph import AgentSettings, build_graph
from .memory import LongTermMemory, empty_memory, memory_text
from .tools import WorkspaceTools

Approver = Callable[[str, dict], bool]


def deny_all(name, args):
    return False


def new_thread_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


class Agent:
    def __init__(self, graph, workspace, long_term, settings, thread_id=None, approver=deny_all):
        self.graph = graph
        self.workspace = workspace
        self.long_term = long_term
        self.settings = settings
        self.approver = approver
        self.thread_id = thread_id or new_thread_id()
        self.long_term.touch_session(self.thread_id)

    @property
    def config(self):
        return {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.settings.recursion_limit}

    def ask(self, text):
        result = self.graph.invoke({"messages": [HumanMessage(text)]}, self.config)
        while result.get("__interrupt__"):
            request = result["__interrupt__"][0].value
            approved = self.approver(request["tool"], request["args"])
            result = self.graph.invoke(Command(resume=approved), self.config)
        return result["final"]

    def state(self):
        return self.graph.get_state(self.config).values

    def memory(self):
        return self.state().get("memory") or empty_memory()

    def memory_text(self):
        return memory_text(self.memory(), self.long_term.notes())

    def reset(self):
        self.thread_id = new_thread_id()
        self.long_term.touch_session(self.thread_id)


def create_agent(model, workspace, checkpointer, store, settings=AgentSettings(), thread_id=None, approver=deny_all):
    tools = WorkspaceTools(workspace.repo_root)
    long_term = LongTermMemory(store, workspace.repo_root)
    graph = build_graph(model, workspace, tools, long_term, settings, checkpointer)
    return Agent(graph, workspace, long_term, settings, thread_id, approver)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_graph.py -v`
Expected: all PASS. If `test_stops_after_too_many_malformed_responses` raises `GraphRecursionError`, `recursion_limit` is not reaching `invoke` — check `Agent.config`.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/langchain_mini_agent/graph.py src/langchain_mini_agent/agent.py tests/test_graph.py
git commit -m "feat(graph): LangGraph agent loop with retries, limits, approvals"
```

---

### Task 9: CLI and REPL (`cli.py`, `__main__.py`)

**Files:**
- Create: `src/langchain_mini_agent/cli.py`, `src/langchain_mini_agent/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `Agent`, `create_agent` (T8); `AgentSettings` (T8); `LongTermMemory`, `open_persistence` (T5); `build_chat_model`, `describe_model_error`, `DEFAULT_MODEL`, `DEFAULT_HOST` (T7); `WorkspaceContext` (T3); `middle` (T2)
- Produces: `build_arg_parser()`, `build_welcome(agent, model) -> str`, `console_approver(name, args) -> bool`, `build_agent_from_args(args) -> tuple[Agent, Path]`, `main(argv=None, agent_factory=build_agent_from_args) -> int`

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:

```python
import pytest
from fakes import ScriptedChatModel, final
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from langchain_mini_agent.agent import create_agent
from langchain_mini_agent.cli import build_agent_from_args, build_arg_parser, build_welcome, console_approver, main
from langchain_mini_agent.graph import AgentSettings
from langchain_mini_agent.workspace import WorkspaceContext


def fake_factory(tmp_path, responses):
    def factory(args):
        (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
        agent = create_agent(
            model=ScriptedChatModel(responses=list(responses)),
            workspace=WorkspaceContext.build(tmp_path),
            checkpointer=InMemorySaver(),
            store=InMemoryStore(),
            settings=AgentSettings(approval_policy="never"),
        )
        return agent, tmp_path / "state.sqlite"
    return factory


def test_parser_defaults():
    args = build_arg_parser().parse_args([])
    assert (args.model, args.approval, args.max_steps, args.num_ctx) == ("qwen3.5:4b", "ask", 6, 16384)


def test_welcome_shows_model_and_session(tmp_path):
    agent, _ = fake_factory(tmp_path, [])(None)
    banner = build_welcome(agent, model="qwen3.5:4b")
    assert "MINI CODING AGENT" in banner and "qwen3.5:4b" in banner and agent.thread_id[:8] in banner


def test_one_shot_prints_final(tmp_path, capsys):
    assert main(["say", "hi"], agent_factory=fake_factory(tmp_path, [final("hello there")])) == 0
    assert "hello there" in capsys.readouterr().out


def test_one_shot_model_error_returns_1(tmp_path, capsys):
    factory = fake_factory(tmp_path, [])

    def failing_factory(args):
        agent, db_path = factory(args)

        def boom(text):
            raise ConnectionError("refused")

        agent.ask = boom
        return agent, db_path

    assert main(["hi"], agent_factory=failing_factory) == 1
    assert "ollama serve" in capsys.readouterr().err


def test_repl_slash_commands(tmp_path, capsys, monkeypatch):
    inputs = iter(["/help", "/memory", "/session", "/reset", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert main([], agent_factory=fake_factory(tmp_path, [])) == 0
    out = capsys.readouterr().out
    for text in ("Commands:", "Memory:", "thread ", "session reset"):
        assert text in out


@pytest.mark.parametrize("answer, expected", [("yes", True), ("Y", True), ("n", False), ("", False)])
def test_console_approver(monkeypatch, answer, expected):
    monkeypatch.setattr("builtins.input", lambda prompt="": answer)
    assert console_approver("write_file", {"path": "a"}) is expected


def test_console_approver_eof_denies(monkeypatch):
    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert console_approver("run_shell", {"command": "ls"}) is False


def test_build_agent_from_args_resume_latest_without_sessions(tmp_path):
    args = build_arg_parser().parse_args(["--cwd", str(tmp_path), "--resume", "latest"])
    agent, db_path = build_agent_from_args(args)
    assert agent.thread_id
    assert db_path.exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'langchain_mini_agent.cli'`.

- [ ] **Step 3: Implement `cli.py`**

`src/langchain_mini_agent/cli.py`:

```python
import argparse
import json
import shutil
import sys

from ollama import ResponseError

from .agent import create_agent
from .context import middle
from .graph import AgentSettings
from .memory import LongTermMemory, open_persistence
from .model import DEFAULT_HOST, DEFAULT_MODEL, build_chat_model, describe_model_error
from .workspace import WorkspaceContext

WELCOME_ART = (
    "/\\     /\\\\",
    "{  `---'  }",
    "{  O   O  }",
    "~~>  V  <~~",
    "\\\\  \\|/  /",
    "`-----'__",
)
HELP_DETAILS = "\n".join([
    "Commands:",
    "/help    Show this help message.",
    "/memory  Show working memory and long-term notes.",
    "/session Show the current thread id and state database.",
    "/reset   Start a fresh session (new thread, empty memory).",
    "/refresh Rebuild the workspace snapshot, keeping this session.",
    "/exit    Exit the agent.",
])
YES = {"y", "yes", "yee", "ye", "yed", "yea", "yeah"}
MODEL_ERRORS = (ConnectionError, ResponseError)


def console_approver(name, args):
    try:
        answer = input(f"approve {name} {json.dumps(args, ensure_ascii=True)}? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in YES


def build_welcome(agent, model):
    width = max(68, min(shutil.get_terminal_size((80, 20)).columns, 84))
    inner = width - 4
    gap = 3
    left_width = (inner - gap) // 2
    right_width = inner - gap - left_width

    def row(text):
        return f"| {middle(text, inner).ljust(inner)} |"

    def center(text):
        return f"| {middle(text, inner).center(inner)} |"

    def cell(label, value, size):
        return middle(f"{label:<9} {value}", size).ljust(size)

    def pair(left_label, left_value, right_label, right_value):
        return f"| {cell(left_label, left_value, left_width)}{' ' * gap}{cell(right_label, right_value, right_width)} |"

    line = "+" + "=" * (width - 2) + "+"
    rows = [center(text) for text in WELCOME_ART]
    rows += [
        center("MINI CODING AGENT"),
        "+" + "-" * (width - 2) + "+",
        row(""),
        row("WORKSPACE  " + middle(agent.workspace.cwd, inner - 11)),
        pair("MODEL", model, "BRANCH", agent.workspace.branch),
        pair("APPROVAL", agent.settings.approval_policy, "SESSION", agent.thread_id),
        row(""),
    ]
    return "\n".join([line, *rows, line])


def build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Minimal LangGraph coding agent for Ollama models.",
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt.")
    parser.add_argument("--cwd", default=".", help="Workspace directory.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama server URL.")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Ollama request timeout in seconds.")
    parser.add_argument("--resume", default=None, help="Session (thread) id to resume or 'latest'.")
    parser.add_argument(
        "--approval", choices=("ask", "auto", "never"), default="ask",
        help="Approval policy for risky tools; auto grants the model arbitrary command execution and file writes.",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum tool steps per request.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum model output tokens per step.")
    parser.add_argument("--num-ctx", type=int, default=16384, help="Ollama context window in tokens.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling value.")
    return parser


def build_agent_from_args(args):
    workspace = WorkspaceContext.build(args.cwd)
    checkpointer, store, db_path = open_persistence(workspace.repo_root)
    thread_id = args.resume
    if thread_id == "latest":
        thread_id = LongTermMemory(store, workspace.repo_root).latest_session()
    model = build_chat_model(
        model=args.model,
        host=args.host,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        num_ctx=args.num_ctx,
        timeout=args.ollama_timeout,
    )
    agent = create_agent(
        model=model,
        workspace=workspace,
        checkpointer=checkpointer,
        store=store,
        settings=AgentSettings(approval_policy=args.approval, max_steps=args.max_steps),
        thread_id=thread_id,
        approver=console_approver,
    )
    return agent, db_path


def main(argv=None, agent_factory=build_agent_from_args):
    args = build_arg_parser().parse_args(argv)
    agent, db_path = agent_factory(args)
    print(build_welcome(agent, model=args.model))

    def ask(text):
        try:
            print(agent.ask(text))
            return 0
        except MODEL_ERRORS as exc:
            print(describe_model_error(exc, args.host, args.model), file=sys.stderr)
            return 1

    if args.prompt:
        prompt = " ".join(args.prompt).strip()
        if not prompt:
            return 0
        print()
        return ask(prompt)

    while True:
        try:
            user_input = input("\nmini-coding-agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            return 0
        if user_input == "/help":
            print(HELP_DETAILS)
            continue
        if user_input == "/memory":
            print(agent.memory_text())
            continue
        if user_input == "/session":
            print(f"thread {agent.thread_id} in {db_path}")
            continue
        if user_input == "/reset":
            agent.reset()
            print("session reset")
            continue
        if user_input == "/refresh":
            answer = input("Rebuild the workspace context (keeps this session)? [y/N] ")
            if answer.strip().lower() in YES:
                args.resume = agent.thread_id
                agent, db_path = agent_factory(args)
                print("workspace context refreshed")
            continue

        print()
        ask(user_input)
```

`src/langchain_mini_agent/__main__.py`:

```python
from .cli import main

raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Check the entry point**

Run: `uv run lc-mini-agent --help`
Expected: argparse help listing all flags including `--num-ctx`.

- [ ] **Step 6: Commit**

```bash
git add src/langchain_mini_agent/cli.py src/langchain_mini_agent/__main__.py tests/test_cli.py
git commit -m "feat(cli): REPL, slash commands, and session resume"
```

---

### Task 10: Live Ollama check, prefix-cache check, README

**Files:**
- Create: `tests/test_live_ollama.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above
- Produces: documentation; an opt-in live test

- [ ] **Step 1: Write the opt-in live test**

`tests/test_live_ollama.py`:

```python
import os

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from langchain_mini_agent.agent import create_agent
from langchain_mini_agent.graph import AgentSettings
from langchain_mini_agent.model import DEFAULT_MODEL, build_chat_model
from langchain_mini_agent.workspace import WorkspaceContext

pytestmark = pytest.mark.skipif(os.environ.get("LIVE_OLLAMA") != "1", reason="set LIVE_OLLAMA=1 with `ollama serve` running")


def test_live_agent_reads_readme(tmp_path):
    (tmp_path / "README.md").write_text("The secret word is pelican.\n", encoding="utf-8")
    agent = create_agent(
        model=build_chat_model(model=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)),
        workspace=WorkspaceContext.build(tmp_path),
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        settings=AgentSettings(approval_policy="never"),
    )
    answer = agent.ask("Use read_file on README.md and tell me the secret word.")
    assert "pelican" in answer.lower()
```

- [ ] **Step 2: Run it against a local Ollama**

Run: `ollama pull qwen3.5:4b && LIVE_OLLAMA=1 uv run pytest tests/test_live_ollama.py -v -s`
Expected: PASS (small models are sometimes flaky; rerun once before debugging). Without `LIVE_OLLAMA=1` it is SKIPPED.

- [ ] **Step 3: Check prefix caching by hand**

Run in the repo root:

```bash
uv run python - <<'EOF'
from langchain_core.messages import HumanMessage
from langchain_mini_agent.model import build_chat_model
from langchain_mini_agent.prompts import build_prefix, render_prompt
from langchain_mini_agent.tools import WorkspaceTools
from langchain_mini_agent.workspace import WorkspaceContext

ws = WorkspaceContext.build(".")
tools = WorkspaceTools(ws.repo_root)
model = build_chat_model().bind_tools(tools.langchain_tools())
prefix = build_prefix(tools, ws)
for question in ("What files are here?", "What branch is this?"):
    reply = model.invoke(render_prompt(prefix, "Memory: -", [HumanMessage(question)]))
    meta = reply.response_metadata
    print(question, "prompt_eval_count=", meta.get("prompt_eval_count"), "prompt_eval_duration=", meta.get("prompt_eval_duration"))
EOF
```

Expected: the second call's `prompt_eval_duration` is clearly lower than the first, because Ollama reuses the KV cache for the shared system+tools prefix. If both are equally slow, check that `keep_alive` is set and that `num_ctx` is larger than the prompt.

- [ ] **Step 4: Update README**

Replace `README.md` with:

````markdown
# langchain-mini-coding-agent

A manual port of the core concepts of [rasbt/mini-coding-agent](https://github.com/rasbt/mini-coding-agent) to LangChain + LangGraph, for educational purposes.

## Components

| Capability | LangChain / LangGraph piece | Module |
|---|---|---|
| Inference on Ollama | `ChatOllama` + native tool calling | `model.py` |
| Short-term memory | LangGraph checkpointer (`SqliteSaver`), one thread per session | `memory.py`, `graph.py` |
| Long-term memory | LangGraph store (`SqliteStore`), notes shared across sessions per repo | `memory.py` |
| Context management | `trim_messages` + per-message clipping/dedup | `context.py` |
| Tools and validation | `StructuredTool` + Pydantic schemas, sandboxed paths, `interrupt()` approvals | `tools.py`, `graph.py` |
| Repo reading | git + docs snapshot in the system prompt | `workspace.py` |
| Prompt prefix caching | stable `ChatPromptTemplate` system prefix, Ollama `keep_alive` / `num_ctx` | `prompts.py`, `model.py` |

Not ported: delegation to subagents.

## Usage

```bash
ollama serve
ollama pull qwen3.5:4b
uv sync
uv run lc-mini-agent                 # interactive
uv run lc-mini-agent "list the files" --approval never
uv run lc-mini-agent --resume latest
```

Sessions and long-term notes are stored in `.mini-coding-agent/state.sqlite` under the workspace root.

REPL commands: `/help`, `/memory`, `/session`, `/reset`, `/refresh`, `/exit`.

## Tests

```bash
uv run pytest                        # offline
LIVE_OLLAMA=1 uv run pytest tests/test_live_ollama.py
```
````

- [ ] **Step 5: Full suite and lint**

Run: `uv run pytest -v && uv run ruff check src tests`
Expected: all PASS (live test SKIPPED), no lint errors.

- [ ] **Step 6: Commit**

```bash
git add tests/test_live_ollama.py README.md
git commit -m "docs: component map, usage, and opt-in live Ollama test"
```
