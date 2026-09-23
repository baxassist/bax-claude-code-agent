"""Свои события агента: остановлено, перезапуск, сменили модель, ответ на вопрос.

В файлы сессий Claude Code это не попадает — он про них не знает. Поэтому маленький
append-only журнал рядом с конфигом: строка на событие, с тем же номером записи (`id`),
что и у сообщений ленты, чтобы приложение показало их в правильном месте.

Только запись и чтение хвоста: чинить обрезанную строку не нужно — она просто пропускается.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("bax.events")


@dataclass(frozen=True)
class Event:
    id: int
    agent: str
    session: str
    kind: str   # event | error
    text: str


class EventLog:
    def __init__(self, file: Path) -> None:
        self.file = file

    def add(self, event: Event) -> None:
        try:
            self.file.parent.mkdir(parents=True, exist_ok=True)
            with self.file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event.__dict__, ensure_ascii=False) + "\n")
        except OSError as error:  # журнал не критичен: не пишется — работаем дальше
            logger.warning("событие не записалось: %s", error)

    def tail(self, agent: str, session: str, limit: int = 10) -> list[Event]:
        if not self.file.exists():
            return []
        found: list[Event] = []
        try:
            with self.file.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("agent") == agent and data.get("session") == session:
                        found.append(Event(**data))
        except OSError:
            return []
        return found[-limit:]
