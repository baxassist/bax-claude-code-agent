"""История из файлов сессий Claude Code: что попадает в ленту, а что остаётся служебным."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bax_agent import history


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """Свой ~ на время теста: настоящие сессии пользователя трогать нельзя."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


def write_session(project: Path, session: str, entries: list[dict]) -> Path:
    file = history.session_file(project, session)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
    return file


ENTRIES = [
    {"type": "mode", "mode": "normal"},                                   # 0 — служебное
    {"type": "user", "isMeta": True, "message": {"content": "<caveat>"}},  # 1 — вставка CLI
    {"type": "user", "message": {"content": "собери проект"}},             # 2
    {"type": "assistant", "message": {"content": [                         # 3
        {"type": "thinking", "thinking": "прикидываю"},
        {"type": "tool_use", "name": "Bash", "input": {"description": "сборка"}},
        {"type": "text", "text": "готово"}]}},
    {"type": "ai-title", "aiTitle": "Сборка проекта"},                     # 4 — служебное
    {"type": "user", "message": {"content": [                              # 5 — вывод инструмента
        {"type": "tool_result", "tool_use_id": "t1", "content": "ок"}]}},
]


def test_project_dir_matches_claude_code(home):
    assert history.project_dir(Path("/Users/me/projects/api")).name == "-Users-me-projects-api"


def test_tail_skips_service_entries(home, tmp_path):
    project = tmp_path / "api"
    file = write_session(project, "s1", ENTRIES)

    messages = history.tail(file, limit=10)
    assert [(m.id, m.kind) for m in messages] == [
        (2, "user"), (3, "thinking"), (3, "tool"), (3, "assistant"),
    ], "служебные записи и вывод инструментов в ленту не идут"
    assert messages[2].text == "Bash: сборка", "вызов инструмента — одной строкой"


def test_tail_keeps_only_last(home, tmp_path):
    file = write_session(tmp_path / "api", "s1", ENTRIES)
    assert [m.kind for m in history.tail(file, limit=2)] == ["tool", "assistant"]


def test_before_pages_up(home, tmp_path):
    file = write_session(tmp_path / "api", "s1", ENTRIES)
    assert [m.id for m in history.before(file, before_id=3, limit=10)] == [2]


def test_next_id_counts_lines(home, tmp_path):
    file = write_session(tmp_path / "api", "s1", ENTRIES)
    assert history.next_id(file) == len(ENTRIES), "номер следующей записи — номер следующей строки"
    assert history.next_id(file.parent / "нет-такого.jsonl") == 0


def test_sessions_from_disk(home, tmp_path):
    project = tmp_path / "api"
    write_session(project, "s1", ENTRIES)
    write_session(project, "s2", [{"type": "user", "message": {"content": "вторая сессия"}}])

    found = {item.session: item for item in history.sessions(project)}
    assert found["s1"].title == "Сборка проекта", "название берём из ai-title, если оно есть"
    assert found["s2"].title == "вторая сессия", "иначе — первая реплика"
    assert found["s1"].entries == len(ENTRIES)


def test_broken_line_does_not_break_reading(home, tmp_path):
    file = write_session(tmp_path / "api", "s1", ENTRIES)
    with file.open("a", encoding="utf-8") as fh:
        fh.write('{"type": "assistant", "message": {"con')  # файл пишется прямо сейчас
    assert [m.id for m in history.tail(file, limit=10)][-1] == 3, "обрезанная строка просто пропускается"
