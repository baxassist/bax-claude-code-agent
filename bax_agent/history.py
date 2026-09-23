"""История — из файлов сессий самого Claude Code, своей копии переписки агент не держит.

Файл сессии: `~/.claude/projects/<путь-проекта-через-дефисы>/<id сессии>.jsonl`, одна запись —
одна строка. Записи бывают служебные (`mode`, `ai-title`, `file-history-*`, `attachment`…) —
в ленту они не идут, но **номера строк не пересчитываются**: `id` сообщения = номер строки
в файле. Поэтому id уникален и монотонен в пределах «поток + сессия», а листание вверх —
это просто чтение строк с номерами меньше `before`.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

# Записи, из которых получается лента. Всё остальное в файле — служебное.
CONVERSATION = ("user", "assistant")


@dataclass(frozen=True)
class Message:
    id: int
    kind: str  # user | assistant | thinking | tool
    text: str

    def as_frame_fields(self) -> dict:
        return {"id": self.id, "kind": self.kind, "text": self.text}


@dataclass(frozen=True)
class Session:
    session: str
    path: Path
    updated_at: float
    entries: int
    title: str


def project_dir(project_path: Path) -> Path:
    """Claude Code кладёт сессии проекта в каталог, где путь записан через дефисы."""
    return Path.home() / ".claude" / "projects" / str(project_path).replace("/", "-")


def session_file(project_path: Path, session: str) -> Path:
    return project_dir(project_path) / f"{session}.jsonl"


def _entries(file: Path):
    """Строка за строкой: (номер, запись). Битые строки пропускаем — файл пишется на лету."""
    if not file.exists():
        return
    with file.open(encoding="utf-8", errors="replace") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                yield index, json.loads(line)
            except json.JSONDecodeError:
                continue


def _tool_line(block: dict) -> str:
    """Вызов инструмента — одной строкой активности: простыни вывода в ленту не тащим."""
    name = str(block.get("name") or "инструмент")
    data = block.get("input")
    if isinstance(data, dict):
        hint = data.get("description") or data.get("command") or data.get("file_path") or data.get("pattern")
        if hint:
            return f"{name}: {str(hint)[:160]}"
    return name


def messages_from(index: int, entry: dict) -> list[Message]:
    """Запись файла → сообщения ленты. Одна запись ответа даёт несколько: рассуждения,
    вызовы инструментов и сам текст — приложение показывает их по-разному."""
    kind = entry.get("type")
    if kind not in CONVERSATION or entry.get("isSidechain"):
        return []
    message = entry.get("message") or {}
    content = message.get("content")

    if kind == "user":
        if entry.get("isMeta"):
            return []  # служебная вставка CLI, не реплика человека
        if isinstance(content, str):
            text = content.strip()
            return [Message(index, "user", text)] if text else []
        found = []
        for block in content or []:
            # tool_result — вывод инструмента, он уже показан строкой активности
            if block.get("type") == "text" and str(block.get("text") or "").strip():
                found.append(Message(index, "user", block["text"].strip()))
        return found

    found = []
    for block in content or []:
        kind_of = block.get("type")
        if kind_of == "text" and str(block.get("text") or "").strip():
            found.append(Message(index, "assistant", block["text"].strip()))
        elif kind_of == "thinking" and str(block.get("thinking") or "").strip():
            found.append(Message(index, "thinking", block["thinking"].strip()))
        elif kind_of == "tool_use":
            found.append(Message(index, "tool", _tool_line(block)))
    return found


def tail(file: Path, limit: int = 10) -> list[Message]:
    """Последние сообщения — то, что приложение получает при открытии потока."""
    found: deque[Message] = deque(maxlen=limit)
    for index, entry in _entries(file):
        found.extend(messages_from(index, entry))
    return list(found)


def before(file: Path, before_id: int, limit: int = 10) -> list[Message]:
    """Листание вверх: сообщения со строками меньше `before_id`."""
    found: deque[Message] = deque(maxlen=limit)
    for index, entry in _entries(file):
        if index >= before_id:
            break
        found.extend(messages_from(index, entry))
    return list(found)


def next_id(file: Path) -> int:
    """Номер следующей строки: с него нумеруются сообщения, которые агент шлёт на лету,
    пока Claude Code ещё не дописал их в файл."""
    return sum(1 for _ in _entries(file)) if file.exists() else 0


def sessions(project_path: Path) -> list[Session]:
    """Список сессий — с диска, своего реестра агент не ведёт."""
    folder = project_dir(project_path)
    if not folder.is_dir():
        return []
    found = []
    for file in folder.glob("*.jsonl"):
        title, entries = "", 0
        for _, entry in _entries(file):
            entries += 1
            if entry.get("type") == "ai-title" and entry.get("aiTitle"):
                title = str(entry["aiTitle"])
            elif not title and entry.get("type") == "user" and not entry.get("isMeta"):
                content = (entry.get("message") or {}).get("content")
                if isinstance(content, str) and content.strip():
                    title = content.strip()[:80]
        found.append(Session(
            session=file.stem,
            path=file,
            updated_at=file.stat().st_mtime,
            entries=entries,
            title=title or "Без названия",
        ))
    return sorted(found, key=lambda item: item.updated_at, reverse=True)
