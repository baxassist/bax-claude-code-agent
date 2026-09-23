"""Расширенные команды: отдать файл, дерево каталога, git, лог агента.

Всё это работает **только внутри каталога проекта**. Путь приводится к настоящему
(`resolve`) и сверяется через `commonpath`: строковый префикс тут не годится — он пропускает
соседний каталог (`/root/app` и `/root/app-secrets` начинаются одинаково).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("bax.commands")

CHUNK = 48 * 1024          # кусок файла до base64: кадр остаётся небольшим
MAX_FILE = 8 * 1024 * 1024  # больше — не отдаём: это не файловый менеджер
TREE_LIMIT = 400            # строк дерева: длиннее читать всё равно нечего
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".build", "DerivedData"}


class CommandError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class FilePart:
    name: str
    mime: str
    size: int
    index: int
    total: int
    data: str


def inside(root: Path, path: str) -> Path:
    """Путь внутри проекта или понятная ошибка. Симлинки раскрываются: уводить наружу нельзя.

    Сравнение через `commonpath`, а не по префиксу строки: «/root/app» — префикс
    «/root/app-secrets», и проверка по строке пропустила бы соседний каталог."""
    root_real = root.resolve()
    try:
        real = (root_real / str(path)).expanduser().resolve()
        if os.path.commonpath([str(root_real), str(real)]) != str(root_real):
            raise ValueError(path)
    except (OSError, ValueError) as error:
        raise CommandError("forbidden", f"Путь «{path}» вне каталога проекта") from error
    return real


def _mime(file: Path) -> str:
    import mimetypes

    return mimetypes.guess_type(file.name)[0] or "application/octet-stream"


def read_file(root: Path, path: str) -> list[FilePart]:
    """Файл кусками. Двоичный отдаём как есть — приложение покажет его как вложение."""
    file = inside(root, path)
    if not file.is_file():
        raise CommandError("not_found", f"Файла «{path}» нет")
    size = file.stat().st_size
    if size > MAX_FILE:
        raise CommandError("forbidden", f"Файл больше {MAX_FILE // 1024 // 1024} МБ — не отдаём")
    raw = file.read_bytes()
    chunks = [raw[i:i + CHUNK] for i in range(0, len(raw), CHUNK)] or [b""]
    return [
        FilePart(name=file.name, mime=_mime(file), size=size, index=index, total=len(chunks),
                 data=base64.b64encode(chunk).decode())
        for index, chunk in enumerate(chunks)
    ]


def tree(root: Path, depth: int = 2) -> str:
    """Дерево каталога проекта — чтобы с телефона понять, что где лежит."""
    depth = max(1, min(int(depth or 2), 5))
    lines: list[str] = []

    def walk(folder: Path, level: int, prefix: str) -> None:
        if level > depth or len(lines) >= TREE_LIMIT:
            return
        try:
            items = sorted(folder.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return
        for item in items:
            if item.name.startswith(".") or item.name in SKIP_DIRS:
                continue
            if len(lines) >= TREE_LIMIT:
                lines.append("…")
                return
            lines.append(f"{prefix}{item.name}{'/' if item.is_dir() else ''}")
            if item.is_dir():
                walk(item, level + 1, prefix + "  ")

    walk(root, 1, "")
    return "\n".join(lines) or "пусто"


async def git(root: Path, mode: str = "status") -> str:
    """Состояние проекта: что изменено, что в последних коммитах."""
    commands = {
        "status": ["git", "status", "--short", "--branch"],
        "diff": ["git", "diff", "--stat"],
        "log": ["git", "log", "-10", "--pretty=format:%h %ad %s", "--date=short"],
    }
    if mode not in commands:
        raise CommandError("not_found", f"git {mode} — такого режима нет")
    return await run(commands[mode], root)


async def run(command: list[str], cwd: Path, limit: int = 20000) -> str:
    try:
        process = await asyncio.create_subprocess_exec(
            *command, cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(process.communicate(), timeout=30)
    except FileNotFoundError as error:
        raise CommandError("not_found", f"нет команды {command[0]}") from error
    except TimeoutError as error:
        raise CommandError("internal", f"{command[0]} не ответила за 30 секунд") from error
    text = out.decode(errors="replace").strip()
    return text[:limit] or "(пусто)"


def logs(file: Path, lines: int = 100) -> str:
    """Хвост лога агента — когда с телефона непонятно, почему ничего не происходит."""
    lines = max(10, min(int(lines or 100), 1000))
    if not file.exists():
        return "лог пуст"
    try:
        return "\n".join(file.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError as error:
        raise CommandError("internal", str(error)) from error
