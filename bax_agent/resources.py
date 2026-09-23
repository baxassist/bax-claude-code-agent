"""Ресурсы: что показать на экране «Ресурсы» потока (AGENT_CLAUDE_CODE.md §8).

Ничего лишнего не ставим: память и загрузка берутся штатными средствами Python и системными
командами, git — самим git. Списки моделей и уровней усилий спрашиваются у CLI и кэшируются:
после обновления Claude Code набор меняется сам, а зашитый в код список устаревает молча.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger("bax.resources")

CACHE_TTL = 600  # десять минут: список моделей меняется не чаще, чем обновляется CLI

_cache: dict[str, tuple[float, list[str]]] = {}

# Запасные значения на случай, если спросить у CLI не вышло: пусть экран не будет пустым
FALLBACK_MODELS = ["opus", "sonnet", "haiku"]
FALLBACK_EFFORTS = ["low", "medium", "high"]


def machine() -> dict:
    """Железо: загрузка, память, место на диске. Всё — приблизительно, для взгляда с телефона."""
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    disk = shutil.disk_usage(Path.home())
    return {
        "cpu_load": round(load, 2),
        "cpu_count": os.cpu_count() or 0,
        "memory_total": _memory_total(),
        "disk_free": disk.free,
        "disk_total": disk.total,
        "uptime_sec": int(time.time() - _started),
    }


def _memory_total() -> int:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


async def git_branch(path: Path) -> dict:
    """Ветка и есть ли несохранённые правки — по ним видно, во что агент вносит изменения."""
    from .commands import CommandError, run

    try:
        branch = await run(["git", "rev-parse", "--abbrev-ref", "HEAD"], path, limit=200)
        dirty = await run(["git", "status", "--porcelain"], path, limit=2000)
    except CommandError:
        return {"branch": "", "dirty": False}
    return {"branch": branch.strip(), "dirty": dirty.strip() not in ("", "(пусто)")}


async def cli_version(binary: str) -> str:
    from .commands import CommandError, run

    try:
        return (await run([binary, "--version"], Path.home(), limit=100)).strip()
    except CommandError:
        return ""


async def choices(binary: str) -> dict[str, list[str]]:
    """Модели и уровни усилий — у самого CLI, с кэшем на десять минут."""
    return {
        "models": await _ask_cli(binary, "models"),
        "efforts": await _ask_cli(binary, "efforts"),
    }


async def _ask_cli(binary: str, what: str) -> list[str]:
    fresh = _cache.get(what)
    if fresh and time.time() - fresh[0] < CACHE_TTL:
        return fresh[1]
    values = await _read_help(binary, what)
    _cache[what] = (time.time(), values)
    return values


async def _read_help(binary: str, what: str) -> list[str]:
    """Разбор `--help`: у флагов `--model` и `--effort` там перечислены допустимые значения."""
    from .commands import CommandError, run

    try:
        help_text = await run([binary, "--help"], Path.home(), limit=60000)
    except CommandError:
        return FALLBACK_MODELS if what == "models" else FALLBACK_EFFORTS

    flag = "--model" if what == "models" else "--effort"
    for line in _flag_block(help_text, flag):
        if "choices:" in line:
            raw = line.split("choices:", 1)[1]
            values = [part.strip().strip('",.()') for part in raw.split(",")]
            found = [value for value in values if value and " " not in value]
            if found:
                return found
    return FALLBACK_MODELS if what == "models" else FALLBACK_EFFORTS


def _flag_block(help_text: str, flag: str) -> list[str]:
    """Строки описания одного флага: до начала следующего."""
    lines = help_text.splitlines()
    block: list[str] = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(flag):
            collecting = True
        elif collecting and stripped.startswith("-"):
            break
        if collecting:
            block.append(stripped)
    return block


_started = time.time()


async def collect(*, binary: str, project: Path, agent_state: dict) -> dict:
    """Полный кадр ресурсов: железо, проект, процесс, модель и лимиты."""
    branch, version, lists = await asyncio.gather(
        git_branch(project), cli_version(binary), choices(binary),
    )
    return {"machine": machine(), "git": branch, "cli_version": version, **lists, **agent_state}
