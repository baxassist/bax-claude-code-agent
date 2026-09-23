"""Поток = проект: кадры приложения превращаются в работу Claude Code и обратно.

Здесь вся склейка: что показать в ленте, что считать активностью, когда поток занят, как
спросить разрешение и как остановить работу. Обмен с сервером — `connection.py`, сессия
Claude Code — `claude.py`, история — `history.py`, правила «больше не спрашивать» —
`rules.py`, команды — `commands.py`.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import (
    TERMINAL_TASK_STATUSES,
    AssistantMessage,
    PermissionResultAllow,
    PermissionResultDeny,
    PermissionUpdate,
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    ThinkingBlock,
    ToolPermissionContext,
    ToolUseBlock,
)

from . import commands, history, resources, rules
from .claude import ClaudeError, ClaudeProcess, resolve_cli
from .commands import CommandError
from .config import Registration
from .events import Event, EventLog

if TYPE_CHECKING:
    from .connection import Link

logger = logging.getLogger("bax.agent")


@dataclass
class Pending:
    """Заданный вопрос: кадр для приложения и ожидание ответа человека."""

    future: asyncio.Future
    frame: dict = field(default_factory=dict)
    tool_input: dict = field(default_factory=dict)
    #: правило «больше не спрашивать» — подсказка самого CLI или своё из `rules.py`
    update: PermissionUpdate | None = None


def _blocks(content: list) -> list[dict]:
    """Типизированные блоки SDK → тот же вид, в каком они лежат в файле сессии.

    Так лента рисуется одним кодом и на лету, и при листании истории (`history.messages_from`).
    """
    found: list[dict] = []
    for block in content:
        if isinstance(block, TextBlock):
            found.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingBlock):
            found.append({"type": "thinking", "thinking": block.thinking})
        elif isinstance(block, ToolUseBlock):
            found.append({"type": "tool_use", "name": block.name, "input": block.input})
    return found


HISTORY_LIMIT = 10
# Картинки из приложения кладутся сюда, внутрь проекта: Claude Code читает файлы только
# из каталога проекта. Папка чистится от старого при каждой выгрузке.
INBOX = ".bax-inbox"
INBOX_KEEP_HOURS = 24


class Agent:
    """Один проект: процесс Claude Code, состояние, лента и вопросы."""

    def __init__(
        self,
        config: Registration,
        link: Link,
        *,
        claude_path: str = "",
        events: EventLog | None = None,
        log_file: Path | None = None,
        task_timeout_min: int = 60,
        question_timeout_min: int = 10,
        semaphore: asyncio.Semaphore | None = None,
        client_factory: Any | None = None,
    ) -> None:
        self.config = config
        self.link = link
        # чем спрашивать у CLI списки моделей и версию: сессии его не касаются — им путь
        # уходит опцией `cli_path`, а пусто означает «CLI из самого пакета SDK»
        self.binary = resolve_cli(claude_path)
        self.events = events
        self.log_file = log_file
        self.task_timeout = max(1, task_timeout_min) * 60
        self.question_timeout = max(1, question_timeout_min) * 60
        self.state = "ready"
        self.model = config.model
        self.effort = config.effort
        self.permission_mode = config.permission_mode
        self.push_enabled = True
        self.limits: dict = {}
        self.context: dict = {}
        self.started_at = 0.0
        self.semaphore = semaphore
        self._live_id = 0          # номер записи, пока Claude Code не дописал её в файл сессии
        self._partial = ""         # накопленный незаконченный ответ: переживает обрыв связи
        self._streaming_id: int | None = None   # запись, которая сейчас приходит кусками
        self._questions: dict[str, Pending] = {}
        self._watchdog: asyncio.Task | None = None
        self._backgrounds: list[str] = []       # фоновые задачи CLI: прерывание хода их не гасит
        self._holding = False                   # занято ли место в общей очереди задач
        self.process = ClaudeProcess(
            config.path,
            self._on_event,
            claude_path=claude_path,
            permission_mode=config.permission_mode,
            model=config.model,
            effort=config.effort,
            restricted=config.restricted,
            on_permission=self._on_permission,
            on_choice=self._on_choice,
            question_timeout=self.question_timeout,
            client_factory=client_factory,
        )

    # --- что видно приложению -------------------------------------------------

    @property
    def session_file(self) -> Path | None:
        session = self.process.session_id
        return history.session_file(self.config.path, session) if session else None

    def apply_start_settings(self, settings: dict) -> None:
        """Начальные модель и усилие приходят с сервера в `ready`: их выбрали, заводя агента
        в приложении (заказчик 21.09).

        Локальная настройка сильнее: если при `bax-agent add` указали `--model`/`--effort`,
        оставляем их — человек за этой машиной знает лучше. Живой процесс не трогаем:
        у него своё состояние, его меняют командой из приложения.
        """
        if self.process.alive:
            return
        model = str(settings.get("model") or "")
        effort = str(settings.get("effort") or "")
        if model and not self.config.model:
            self.model = self.process.model = model
        if effort and not self.config.effort:
            self.effort = self.process.effort = effort

    async def hello(self) -> None:
        """Соединение поднялось: сообщаем, чем агент занят сейчас.

        Списка агентов больше нет — движок подключается ключом одного агента, и сервер
        знает, кто это, из ключа.
        """
        await self._status(self.state)

    async def _status(self, state: str) -> None:
        self.state = state
        await self.link.send("status", state=state)

    async def _stats(self) -> None:
        """Сводка для шапки: модель, усилие, контекст, лимиты.

        Модель и усилие спрашиваем у самого CLI (`get_settings`), а не берём свои переменные:
        после смены модели на ту, у которой уровней усилий нет, он честно вернёт `effort: null`,
        и в шапке не останется усилия, которого на самом деле нет.
        """
        if self.process.alive:
            try:
                applied = await self.process.settings()
            except ClaudeError as error:
                logger.debug("настройки сессии не пришли: %s", error)
            else:
                self.model = str(applied.get("model") or "") or self.model
                self.effort = "" if applied.get("effort") is None else str(applied["effort"])
            self.context = await self._context(self.context)
        await self.link.send(
            "stats", model=self.model, effort=self.effort,
            context=self.context, limits=self.limits,
        )

    async def _context(self, fallback: dict) -> dict:
        """Сколько занято контекста — у самого CLI (`get_context_usage`).

        Своими руками это не посчитать: у модели бывает окно на миллион, а в `usage` хода
        видно только базовое (проверено вживую 22.09 — в шапке стояло 1 000 000, а в `done`
        считалось 200 000).
        """
        usage = await self.process.context_usage()
        if not usage:
            return fallback
        return {"used": int(usage.get("totalTokens") or 0), "max": int(usage.get("maxTokens") or 0)}

    async def _error(self, code: str, message: str) -> None:
        await self.link.send("error", code=code, message=message)

    async def _say(self, text: str, kind: str = "event") -> int:
        """Своё событие: и в ленту, и в журнал — в файлы сессий Claude Code это не попадает."""
        entry = self._next_entry()
        session = self.process.session_id or ""
        if self.events:
            self.events.add(Event(id=entry, agent=self.config.agent_id, session=session,
                                  kind=kind, text=text))
        await self.link.send("message", session=session,
                             id=entry, kind=kind, text=text)
        return entry

    def _next_entry(self) -> int:
        """Следующий номер записи. Файл сессии — источник правды, но он отстаёт от событий,
        поэтому номер никогда не идёт назад: приложение показывает ленту по возрастанию id."""
        file = self.session_file
        self._live_id = max(self._live_id + 1, history.next_id(file) if file else 0)
        return self._live_id

    # --- кадры от приложения --------------------------------------------------

    async def handle(self, frame: dict) -> None:
        kind = frame.get("type")
        if kind == "subscribe":
            await self.subscribe()
        elif kind == "run":
            await self.run(str(frame.get("text") or ""), frame.get("attachments") or [])
        elif kind == "cancel":
            await self.cancel(str(frame.get("scope") or "turn"))
        elif kind == "answer":
            await self.answer(frame)
        elif kind == "history":
            await self.history_page(int(frame.get("before") or 0), int(frame.get("limit") or HISTORY_LIMIT))
        elif kind == "sessions.list":
            await self.list_sessions()
        elif kind == "session.select":
            await self.select_session(str(frame.get("session") or "new"))
        elif kind == "session.compact":
            await self.compact()
        elif kind == "model.set":
            await self.set_model(str(frame.get("model") or ""))
        elif kind == "effort.set":
            await self.set_effort(str(frame.get("effort") or ""))
        elif kind == "settings.set":
            await self.set_settings(frame)
        elif kind == "resources.get":
            await self.send_resources()
        elif kind == "command":
            await self.command(str(frame.get("name") or ""), frame.get("args") or {})
        else:
            await self._error("internal", f"кадр {kind!r} агент пока не умеет")

    async def subscribe(self) -> None:
        """Поток открыли: последние сообщения, состояние и сводка для шапки."""
        session = self.process.session_id or ""
        file = self.session_file
        messages: list[tuple[int, str, str]] = []
        if file:
            messages += [(m.id, m.kind, m.text) for m in history.tail(file, HISTORY_LIMIT)]
        if self.events:
            # свои события (остановлено, смена модели) в файлы Claude Code не попадают —
            # подмешиваем их по номеру записи
            messages += [(e.id, e.kind, e.text) for e in self.events.tail(self.config.agent_id, session)]
        for entry, kind, text in sorted(messages, key=lambda item: item[0])[-HISTORY_LIMIT:]:
            await self.link.send("message", session=session,
                                 id=entry, kind=kind, text=text)
        if self._partial:
            # ход шёл, пока приложение было закрыто: дошлём накопленное одним куском
            await self.link.send("delta", session=session,
                                 id=self._streaming_id or self._live_id, chunk=self._partial)
        for pending in self._questions.values():
            if not pending.future.done():
                # вопрос висит с прошлого раза — показать заново, иначе поток стоит молча
                await self.link.send("question", **pending.frame)
        await self._stats()
        await self._status(self.state)

    async def run(self, text: str, attachments: list[dict] | None = None) -> None:
        if not text.strip() and not attachments:
            return await self._error("internal", "пустая задача")
        if self.state in ("busy", "waiting"):
            return await self._error("busy", "Поток уже занят — дождитесь ответа или нажмите «Стоп»")
        prompt = text.strip()
        try:
            saved = self._save_attachments(attachments or [])
        except (OSError, ValueError) as error:
            return await self._error("internal", f"вложение не сохранилось: {error}")
        if saved:
            # передаём путь, а не base64: Claude Code сам прочитает файл из каталога проекта
            prompt = (prompt + "\n\nПрикреплённые файлы:\n"
                      + "\n".join(f"- {name}" for name in saved)).strip()

        await self._take_slot()
        try:
            await self.process.start(resume=self.process.session_id)
            await self.process.ask(prompt)
        except ClaudeError as error:
            self._free_slot()
            return await self._error("internal", str(error))
        self._partial = ""
        self._streaming_id = None
        self.started_at = time.time()
        await self.link.send("message", session=self.process.session_id,
                             id=self._next_entry(), kind="user", text=text.strip() or "(вложение)")
        await self._status("busy")
        self._watchdog = asyncio.create_task(self._watch_timeout())

    async def cancel(self, scope: str) -> None:
        if scope == "background":
            return await self.stop_background()
        await self.process.interrupt()
        await self._finish_turn()
        await self._say("Остановлено пользователем")
        await self._status("ready")

    async def stop_background(self) -> None:
        """Долгие команды CLI уводит в фон, и прерывание хода их не трогает — гасим отдельно."""
        if not self._backgrounds:
            return await self._say("Фоновых задач нет")
        stopped = 0
        for task_id in list(self._backgrounds):
            try:
                await self.process.stop_task(task_id)
                stopped += 1
            except ClaudeError as error:
                logger.warning("фоновая задача %s не остановилась: %s", task_id, error)
        self._backgrounds.clear()
        await self._say(f"Фоновые задачи остановлены: {stopped}")

    async def answer(self, frame: dict) -> None:
        """Ответ на вопрос из приложения: разрешить, запретить или выбрать вариант."""
        question_id = str(frame.get("question_id") or "")
        pending = self._questions.get(question_id)
        if pending is None or pending.future.done():
            return await self._error("not_found", "Этот вопрос уже закрыт")
        pending.future.set_result({
            "verdict": str(frame.get("verdict") or "deny"),
            "option": str(frame.get("option") or ""),
            "remember": bool(frame.get("remember")),
        })

    async def history_page(self, before: int, limit: int) -> None:
        file = self.session_file
        if not file:
            return
        for message in history.before(file, before, limit):
            await self.link.send("message", session=self.process.session_id,
                                 **message.as_frame_fields())

    async def list_sessions(self) -> None:
        items = [
            {"session": item.session, "title": item.title,
             "updated_at": int(item.updated_at), "entries": item.entries,
             "current": item.session == self.process.session_id}
            for item in history.sessions(self.config.path)
        ]
        await self.link.send("sessions", items=items)

    async def select_session(self, session: str) -> None:
        await self.process.close()
        self._partial = ""
        self._streaming_id = None
        try:
            if session == "new":
                self.process.session_id = None
                await self.process.start()
                await self._say("Начата новая сессия")
            else:
                self.process.session_id = session
                await self.process.start(resume=session)
                await self._say("Сессия переключена")
        except ClaudeError as error:
            return await self._error("internal", str(error))
        self._live_id = history.next_id(self.session_file) if self.session_file else 0
        await self.subscribe()

    async def compact(self) -> None:
        """`/compact` — сжать контекст, не теряя нить: дешевле, чем начинать сессию заново."""
        try:
            await self.process.slash("/compact")
        except ClaudeError as error:
            return await self._error("internal", str(error))
        await self._say("Контекст сжимается")
        await self._status("busy")

    async def set_model(self, model: str) -> None:
        if not model:
            return await self._error("internal", "не указана модель")
        await self._switch("model", model)

    async def set_effort(self, effort: str) -> None:
        if not effort:
            return await self._error("internal", "не указан уровень усилий")
        await self._switch("effort", effort)

    async def _switch(self, what: str, value: str) -> None:
        """Смена на живой сессии: у модели для этого есть метод SDK `set_model`, у уровня
        усилий публичного метода нет — уходит слэш-команда (CLI обрабатывает её локально,
        без обращения к модели и расхода токенов), а если и она не прошла — управляющий
        запрос `apply_flag_settings`. Не вышло ничего — перезапуск с новым значением
        и `resume` той же сессии: контекст сохраняется."""
        setattr(self, what, value)
        setattr(self.process, what, value)
        try:
            if what == "model":
                await self.process.set_model(value)
            else:
                await self.process.slash(f"/effort {value}")
                if not await self._applied("effort", value):
                    # слэш-команда уходит в очередь сессии и применяется не мгновенно —
                    # управляющий запрос применяет сразу (поймано живой проверкой 22.09:
                    # в шапке после смены оказывалось прежнее значение)
                    await self.process.apply_flag_settings(effortLevel=value)
        except ClaudeError as error:
            logger.info("смена %s на живой сессии не прошла (%s) — перезапускаю", what, error)
            if not await self._restart():
                return
        await self._say(f"{'Модель' if what == 'model' else 'Уровень усилий'}: {value}")
        # серверу это не сообщаем: он таких настроек не хранит — приложение видит их
        # в кадре `stats` (решение заказчика 20.09)
        await self._stats()

    async def _applied(self, what: str, value: str, tries: int = 6) -> bool:
        """Дождаться, пока CLI действительно применит значение.

        Значение спрашивается у него же (`get_settings`), а не считается применённым по факту
        отправки: слэш-команда — это сообщение в сессию, и между «отправили» и «применилось»
        проходит время. У модели без уровней усилий не применится никогда — тогда это
        не ошибка, просто в шапке усилия не будет.
        """
        for _ in range(tries):
            await asyncio.sleep(0.3)
            if str((await self.process.settings()).get(what) or "") == value:
                return True
        return False

    async def set_settings(self, frame: dict) -> None:
        mode = str(frame.get("permission_mode") or "")
        if mode:
            self.permission_mode = mode
            try:
                # у CLI это управляющий запрос: процесс и сессия остаются живы
                await self.process.set_permission_mode(mode)
            except ClaudeError as error:
                logger.info("режим разрешений на живом процессе не сменился (%s) — перезапускаю", error)
                self.process.permission_mode = mode
                if not await self._restart():
                    return
            await self._say(f"Режим разрешений: {mode}")
        if "push" in frame:
            self.push_enabled = bool(frame.get("push"))
        await self._stats()

    async def _restart(self) -> bool:
        session = self.process.session_id
        await self.process.close()
        try:
            await self.process.start(resume=session)
        except ClaudeError as error:
            await self._error("internal", str(error))
            return False
        return True

    async def send_resources(self) -> None:
        state = {
            "agent": self.config.agent_id,
            "path": str(self.config.path),
            "model": self.model,
            "effort": self.effort,
            "permission_mode": self.permission_mode,
            "session": self.process.session_id or "",
            "state": self.state,
            "busy_sec": int(time.time() - self.started_at) if self.state == "busy" else 0,
            "context": self.context,
            "limits": self.limits,
            "backgrounds": len(self._backgrounds),
        }
        data = await resources.collect(binary=self.binary, project=self.config.path, agent_state=state)
        await self.link.send("resources", **data)

    async def command(self, name: str, args: dict) -> None:
        """Расширенные команды из меню потока."""
        try:
            if name == "file":
                for part in commands.read_file(self.config.path, str(args.get("path") or "")):
                    await self.link.send("file", **part.__dict__)
                return
            if name == "tree":
                text = commands.tree(self.config.path, int(args.get("depth") or 2))
            elif name == "git":
                text = await commands.git(self.config.path, str(args.get("mode") or "status"))
            elif name == "logs":
                text = commands.logs(self.log_file, int(args.get("lines") or 100)) if self.log_file \
                    else "лог агента не настроен"
            elif name == "restart":
                if await self._restart():
                    await self._say("Процесс Claude Code перезапущен")
                return
            else:
                return await self._error("not_found", f"команда «{name}» не известна")
        except CommandError as error:
            return await self._error(error.code, error.message)
        await self.link.send("message", session=self.process.session_id,
                             id=self._next_entry(), kind="event", text=text)

    # --- вопросы разрешений ---------------------------------------------------

    async def _on_permission(
        self, tool: str, tool_input: dict, context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Разрешение: карточка в приложении и ожидание человека.

        Колбэк SDK — вместо прежнего MCP-сервера по сокету. Ждать здесь можно сколько угодно:
        SDK обрабатывает такие запросы отдельной задачей и поток сообщений не держит.
        """
        update = self._remember_update(tool, tool_input, context)
        pending = Pending(
            future=asyncio.get_running_loop().create_future(),
            tool_input=tool_input,
            update=update,
        )
        question_id = str(context.tool_use_id or uuid.uuid4())
        pending.frame = {
            "question_id": question_id,
            "kind": "permission",
            "tool": tool,
            "input": tool_input,
            "text": str(context.description or context.title or ""),
            "options": [],
            # что запомнится по «больше не спрашивать» — это видно на кнопке; поля нет —
            # и кнопки в приложении не будет
            "rule": rules.label_for(update) if update else "",
        }
        answer = await self._ask(question_id, pending)
        if answer.get("verdict") != "allow":
            return PermissionResultDeny(message=answer.get("message")
                                        or "Пользователь запретил это действие")
        updates = [update] if answer.get("remember") and update else None
        if updates:
            logger.info("правило на сессию: %s", rules.label_for(update))
        return PermissionResultAllow(updated_input=tool_input, updated_permissions=updates)

    async def _on_choice(self, question: str, options: list[str]) -> str:
        """`ask_user`: смысловой вопрос с вариантами — какой вариант делаем, продолжать ли."""
        question_id = str(uuid.uuid4())
        pending = Pending(future=asyncio.get_running_loop().create_future())
        pending.frame = {
            "question_id": question_id,
            "kind": "choice",
            "tool": "",
            "input": {},
            "text": question,
            "options": options,
            "rule": "",
        }
        answer = await self._ask(question_id, pending)
        return str(answer.get("option") or "")

    def _remember_update(
        self, tool: str, tool_input: dict, context: ToolPermissionContext
    ) -> PermissionUpdate | None:
        """Что запомнить по «больше не спрашивать» — точечным правилом самого CLI.

        Сначала берём подсказку из вопроса (`context.suggestions`) — это ровно то, что CLI
        предложил бы в терминале. Подсказки нет — строим правило сами (`rules.py`). Опасным
        командам (`rm`, `sudo`) правила не даём совсем: такое спрашивается каждый раз, сколько
        бы раз ни разрешали, — поэтому своё правило проверяется даже тогда, когда подсказка есть.
        """
        own = rules.rule_for(tool, tool_input, root=self.process.cwd)
        if own is None:
            return None
        for update in context.suggestions or []:
            if update.type == "addRules" and update.behavior == "allow" and (update.rules or []):
                # область только на сессию: в файлы настроек проекта не пишем
                return replace(update, destination="session")
        return own.as_permission_update()

    async def _ask(self, question_id: str, pending: Pending) -> dict:
        """Показать вопрос и дождаться человека. Молчание — не «да»: по тайм-ауту запрещаем."""
        self._questions[question_id] = pending
        previous = self.state
        await self.link.send("question", **pending.frame)
        await self._status("waiting")
        try:
            return await asyncio.wait_for(pending.future, timeout=self.question_timeout)
        except TimeoutError:
            logger.info("вопрос остался без ответа — запрещаем")
            return {"verdict": "deny", "message": "Никто не ответил вовремя — действие запрещено"}
        finally:
            self._questions.pop(question_id, None)
            await self._status("busy" if previous in ("busy", "waiting") else previous)

    # --- события Claude Code --------------------------------------------------

    async def _on_event(self, message: Any) -> None:
        """Сообщения приходят объектами SDK, а не сырыми словарями."""
        if isinstance(message, StreamEvent):
            return await self._on_delta(message.event or {})
        if isinstance(message, AssistantMessage):
            return await self._on_assistant(message)
        if isinstance(message, ResultMessage):
            return await self._on_result(message)
        if isinstance(message, RateLimitEvent):
            self.limits = self._limits_from(message.rate_limit_info)
            return await self._stats()
        if isinstance(message, TaskStartedMessage):
            # фоновые задачи CLI: прерывание хода их не гасит, их останавливают по id
            if message.task_id and message.task_id not in self._backgrounds:
                self._backgrounds.append(message.task_id)
            return
        if isinstance(message, TaskUpdatedMessage | TaskNotificationMessage):
            if message.status in TERMINAL_TASK_STATUSES and message.task_id in self._backgrounds:
                self._backgrounds.remove(message.task_id)
            return
        if isinstance(message, SystemMessage) and message.subtype == "init":
            self.model = str(message.data.get("model") or "") or self.model
            return await self._stats()

    async def _on_delta(self, event: dict) -> None:
        if event.get("type") != "content_block_delta":
            return
        delta = event.get("delta") or {}
        if delta.get("type") != "text_delta":
            return  # рассуждения приходят отдельным сообщением, в потоке их не показываем
        chunk = str(delta.get("text") or "")
        if not chunk:
            return
        if self._streaming_id is None:
            # номер берём на первом куске и отдаём его же готовому сообщению:
            # иначе приложение покажет один ответ дважды — потоком и целиком
            self._streaming_id = self._next_entry()
        self._partial += chunk
        await self.link.send("delta", session=self.process.session_id,
                             id=self._streaming_id, chunk=chunk)

    async def _on_assistant(self, message: AssistantMessage) -> None:
        """Одна запись ответа = один id. Рассуждение, вызов инструмента и текст приходят
        отдельными сообщениями с общим номером — ровно так же, как потом читаются из файла
        сессии, где всё это лежит одной строкой."""
        entry = self._streaming_id if self._streaming_id is not None else self._next_entry()
        self._streaming_id = None
        self._partial = ""
        record = {"type": "assistant", "message": {"content": _blocks(message.content)}}
        for line in history.messages_from(entry, record):
            await self.link.send("message", session=self.process.session_id,
                                 id=entry, kind=line.kind, text=line.text)
            if line.kind == "tool":
                await self.link.send("activity", label=line.text)

    async def _on_result(self, message: ResultMessage) -> None:
        self._streaming_id = None
        usage = message.usage or {}
        # окно модели CLI сообщает сам, в model_usage: хардкодить размеры контекста не нужно
        model, spent = next(iter((message.model_usage or {}).items()), ("", {}))
        self.model = model or self.model
        self.context = await self._context({
            "used": int(usage.get("input_tokens") or 0)
            + int(usage.get("cache_read_input_tokens") or 0)
            + int(usage.get("cache_creation_input_tokens") or 0),
            "max": int(spent.get("contextWindow") or 0),
        })
        await self._finish_turn()
        await self.link.send(
            "done", session=self.process.session_id, id=self._live_id,
            usage=usage, context=self.context, limits=self.limits,
            duration_ms=message.duration_ms, cost=message.total_cost_usd or 0,
        )
        if message.is_error:
            await self._error("internal", str(message.result or "Claude Code вернул ошибку"))
        # «свободен» — раньше сводки: она спрашивает CLI (`get_settings`, `get_context_usage`),
        # и пока ответы идут, задача, отправленная сразу после `done`, получала «занят»
        # (поймано живой проверкой 22.09)
        await self._status("ready")
        await self._stats()

    # --- очередь задач и тайм-аут ---------------------------------------------

    async def _take_slot(self) -> None:
        """Общий предел одновременных задач: машина у пользователя одна на все потоки."""
        if self.semaphore is not None and not self._holding:
            await self.semaphore.acquire()
            self._holding = True

    def _free_slot(self) -> None:
        if self.semaphore is not None and self._holding:
            self.semaphore.release()
        self._holding = False

    async def _finish_turn(self) -> None:
        """Ход закончился — снять сторожа и отпустить место в общей очереди."""
        if self._watchdog and not self._watchdog.done():
            self._watchdog.cancel()
        self._watchdog = None
        self._free_slot()

    async def _watch_timeout(self) -> None:
        """Задача не может идти вечно: иначе поток занят, а человек не понимает почему."""
        try:
            await asyncio.sleep(self.task_timeout)
        except asyncio.CancelledError:
            return
        if self.state not in ("busy", "waiting"):
            return
        logger.warning("задача идёт дольше %s минут — прерываю", self.task_timeout // 60)
        await self.process.interrupt()
        self._watchdog = None
        self._free_slot()
        await self._say(f"Задача шла дольше {self.task_timeout // 60} минут и была прервана", kind="error")
        await self._status("ready")

    # --- вложения -------------------------------------------------------------

    def _save_attachments(self, attachments: list[dict]) -> list[str]:
        """Картинки из приложения — в служебную папку внутри проекта: Claude Code читает
        файлы только оттуда. Старое подчищаем, чтобы папка не росла."""
        if not attachments:
            return []
        inbox = self.config.path / INBOX
        inbox.mkdir(exist_ok=True)
        edge = time.time() - INBOX_KEEP_HOURS * 3600
        for old in inbox.iterdir():
            if old.is_file() and old.stat().st_mtime < edge:
                old.unlink(missing_ok=True)

        saved = []
        for item in attachments:
            data = item.get("data")
            if not data:
                continue
            # имя приходит снаружи: берём только его последнюю часть, без путей
            name = Path(str(item.get("name") or "")).name or f"{uuid.uuid4().hex[:8]}.png"
            file = inbox / f"{uuid.uuid4().hex[:8]}-{name}"
            file.write_bytes(base64.b64decode(data))
            saved.append(f"{INBOX}/{file.name}")
        return saved

    @staticmethod
    def _limits_from(info: RateLimitInfo) -> dict:
        """Лимиты подписки — из событий самого CLI, своего учёта агент не ведёт.
        `utilization` приходит долей (0.08), а в шапке нужны проценты.

        В типизированном виде SDK отдаёт одно окно — то, которое ближе к пределу; оба сразу
        лежат в `raw.unifiedWindows`, оттуда и берём, когда они есть."""
        windows = (info.raw or {}).get("unifiedWindows") or {}
        limits = {}
        for window in ("five_hour", "seven_day"):
            data = windows.get(window) or {}
            if data:
                limits[window] = {
                    "used_pct": round(float(data.get("utilization") or 0) * 100),
                    "resets_at": int(data.get("resetsAt") or 0),
                }
        if not limits and info.rate_limit_type in ("five_hour", "seven_day"):
            limits[info.rate_limit_type] = {
                "used_pct": round(float(info.utilization or 0) * 100),
                "resets_at": int(info.resets_at or 0),
            }
        return limits

    async def close(self) -> None:
        if self._watchdog and not self._watchdog.done():
            self._watchdog.cancel()
        await self.process.close()
