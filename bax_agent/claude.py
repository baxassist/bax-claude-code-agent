"""Долгоживущая сессия Claude Code через официальный Claude Agent SDK.

Почему SDK, а не свой запуск `claude -p` (решение 22.09):
SDK поднимает тот же CLI с тем же `stream-json`, но управляющие запросы (`interrupt`,
`set_model`, `stop_task`) у него публичный API, а не находки из разбора бинарника; CLI идёт
внутри пакета и обновляется вместе с ним, а разрешения приходят колбэком прямо в процесс —
отдельный MCP-сервер и юникс-сокет больше не нужны.

Почему сессия живёт долго, а не поднимается на каждую задачу: реплики дописываются
в работающий процесс — сессию не нужно каждый раз прогревать заново, а ответ на вопрос
уходит в тот же ход.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    CLINotFoundError,
    PermissionResultAllow,
    PermissionResultDeny,
    SystemMessage,
    ToolPermissionContext,
    create_sdk_mcp_server,
    tool,
)

logger = logging.getLogger("bax.claude")

Event = Callable[[Any], Awaitable[None]]
Permission = Callable[
    [str, dict, ToolPermissionContext],
    Awaitable[PermissionResultAllow | PermissionResultDeny],
]
Choice = Callable[[str, list[str]], Awaitable[str]]

#: Сколько CLI ждёт ответа от нашего MCP-инструмента. У него свой предел в минутах, а человек
#: отвечает с телефона дольше — без этого вопрос обрывается раньше, чем его увидят.
MCP_TIMEOUT_ENV = "MCP_TOOL_TIMEOUT"

#: Управляющий запрос не должен висеть вечно: CLI отвечает на них мгновенно.
CONTROL_TIMEOUT = 15.0

ASK_USER_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "options": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["question"],
}


class ClaudeError(Exception):
    """Сессия не поднялась или умерла: наверх уходит понятной строкой."""


def ask_user_tool(handler: Choice) -> Any:
    """`ask_user` — смысловой вопрос с вариантами («какой вариант делаем?»).

    Отдельной функцией, а не внутри класса: так инструмент можно позвать в тесте так же,
    как его зовёт Claude Code, — без поднятия MCP.
    """

    @tool(
        "ask_user",
        "Задать пользователю смысловой вопрос с вариантами ответа, когда нужно решение "
        "человека: какой вариант делаем, продолжать ли.",
        ASK_USER_SCHEMA,
    )
    async def ask_user(args: dict) -> dict:
        answer = await handler(
            str(args.get("question") or ""),
            [str(option) for option in (args.get("options") or [])],
        )
        return {"content": [{"type": "text", "text": answer}]}

    return ask_user


def resolve_cli(configured: str = "") -> str:
    """Чем спрашивать у CLI справку (`--version`, `--help`): настроенный путь, иначе CLI
    из самого пакета SDK, иначе системный `claude`.

    Сессий это не касается — им путь уходит опцией `cli_path`, и пусто там означает
    «бери свой». Здесь пусто означало бы `claude`, которого на машине может не быть:
    с SDK его отдельная установка больше не обязательна.
    """
    if configured:
        return configured
    import claude_agent_sdk

    name = "claude.exe" if sys.platform == "win32" else "claude"
    bundled = Path(claude_agent_sdk.__file__).parent / "_bundled" / name
    return str(bundled) if bundled.is_file() else "claude"


class ClaudeProcess:
    """Одна сессия Claude Code в каталоге одного проекта."""

    def __init__(
        self,
        cwd: Path,
        on_event: Event,
        *,
        claude_path: str = "",
        permission_mode: str = "acceptEdits",
        model: str = "",
        effort: str = "",
        restricted: bool = False,
        on_permission: Permission | None = None,
        on_choice: Choice | None = None,
        question_timeout: float = 600.0,
        client_factory: Callable[[ClaudeAgentOptions], Any] | None = None,
    ) -> None:
        self.cwd = cwd
        self.on_event = on_event
        self.claude_path = claude_path
        self.permission_mode = permission_mode
        self.model = model
        self.effort = effort
        self.restricted = restricted
        self.on_permission = on_permission
        self.on_choice = on_choice
        self.question_timeout = question_timeout
        self.session_id: str | None = None
        # подставной клиент в тестах: настоящий SDK поднимает процесс и требует подписки
        self._factory = client_factory or ClaudeSDKClient
        self._client: Any | None = None
        self._pump: asyncio.Task | None = None

    @property
    def alive(self) -> bool:
        return self._client is not None

    def options(self, resume: str | None = None) -> ClaudeAgentOptions:
        """Опции запуска — те же флаги CLI, что были в самодельной обвязке.

        `strict_mcp_config` намеренно: у сессии только наш сервер с `ask_user`. Глобальные MCP
        (почта, диск, календарь) агенту для кода не нужны, а каждый запрос они раздувают.

        `system_prompt` — обязательно пресет `claude_code`: с `None` SDK передаёт CLI **пустую**
        системную подсказку, и это перестаёт быть Claude Code. `setting_sources` и `tools`
        не трогаем: их значения по умолчанию совпадают с поведением CLI, включая CLAUDE.md
        проекта.
        """
        extra: dict[str, str | None] = {}
        if self.restricted:
            # инструменты запираются в каталоге проекта, команд и WebFetch нет
            extra["restricted"] = None
        servers = {"bax": self._ask_user_server()} if self.on_choice else {}
        return ClaudeAgentOptions(
            cwd=str(self.cwd),
            permission_mode=self.permission_mode,  # type: ignore[arg-type]
            model=self.model or None,
            effort=self.effort or None,  # type: ignore[arg-type]
            resume=resume or None,
            session_id=None if resume else self.session_id,
            include_partial_messages=True,     # ответ на лету
            system_prompt={"type": "preset", "preset": "claude_code"},
            mcp_servers=servers,
            strict_mcp_config=True,
            can_use_tool=self.on_permission,
            extra_args=extra,
            env={MCP_TIMEOUT_ENV: str(int(self.question_timeout * 1000) + 30_000)},
            cli_path=self.claude_path or None,  # пусто — CLI из самого пакета SDK
        )

    async def start(self, resume: str | None = None) -> None:
        if self.alive:
            return
        if not resume:
            self.session_id = str(uuid.uuid4())
        client = self._factory(self.options(resume))
        logger.info("поднимаю сессию Claude Code в %s (resume=%s)", self.cwd, resume or "нет")
        try:
            await client.connect()
        except CLINotFoundError as error:
            raise ClaudeError(f"не нашёлся claude: {error}") from error
        except (ClaudeSDKError, OSError) as error:
            raise ClaudeError(f"Claude Code не запустился: {error}") from error
        self._client = client
        self._pump = asyncio.create_task(self._read(client))

    async def ask(self, text: str) -> None:
        """Реплика пользователя в работающую сессию."""
        if not self.alive:
            await self.start(resume=self.session_id)
        client = self._client
        if client is None:
            raise ClaudeError("сессия Claude Code не запущена")
        try:
            await client.query(text)
        except (ClaudeSDKError, OSError) as error:
            raise ClaudeError(f"Claude Code не принял реплику: {error}") from error

    async def slash(self, text: str) -> None:
        """Слэш-команда (`/compact`, `/effort high`): CLI обрабатывает её локально,
        без обращения к модели и расхода токенов."""
        await self.ask(text)

    async def interrupt(self) -> None:
        """Кнопка «Стоп»: ход прерывается, процесс и сессия остаются живы."""
        client = self._client
        if client is None:
            return
        try:
            await asyncio.wait_for(client.interrupt(), timeout=CONTROL_TIMEOUT)
        except (ClaudeSDKError, TimeoutError, OSError) as error:
            # прерывание не должно ронять обработку кадра: пишем в журнал и живём дальше
            logger.warning("прерывание не прошло: %s", error)

    async def set_model(self, model: str) -> None:
        await self._call("set_model", lambda client: client.set_model(model))
        self.model = model

    async def set_permission_mode(self, mode: str) -> None:
        await self._call("set_permission_mode", lambda client: client.set_permission_mode(mode))
        self.permission_mode = mode

    async def stop_task(self, task_id: str) -> None:
        """Фоновую задачу прерывание хода не трогает — её гасят отдельно, по её id."""
        await self._call("stop_task", lambda client: client.stop_task(task_id))

    async def settings(self) -> dict:
        """Что реально применено: `{"model": …, "effort": … | null}`.

        У модели без уровней усилий CLI честно вернёт `effort: null` — поэтому кадр `stats`
        собирается отсюда, а не из наших переменных. **Приватный API**: публичного метода
        для `get_settings` в Python SDK пока нет, запрос уходит внутренним
        `_send_control_request` — при обновлении SDK проверять это место первым.
        """
        answer = await self._control({"subtype": "get_settings"})
        return dict(answer.get("applied") or {})

    async def apply_flag_settings(self, **settings: Any) -> None:
        """Запасной способ сменить усилие, когда слэш-команда не прошла (тот же приватный путь)."""
        await self._control({"subtype": "apply_flag_settings", "settings": settings})

    async def context_usage(self) -> dict:
        """Сколько занято контекста. Не пришло — не беда: в шапке просто не обновится."""
        client = self._client
        if client is None:
            return {}
        try:
            return dict(await asyncio.wait_for(client.get_context_usage(), timeout=CONTROL_TIMEOUT))
        except (ClaudeSDKError, TimeoutError, OSError) as error:
            logger.debug("контекст не пришёл: %s", error)
            return {}

    async def close(self) -> None:
        client, self._client = self._client, None
        pump, self._pump = self._pump, None
        if pump is not None:
            pump.cancel()
        if client is None:
            return
        try:
            await asyncio.wait_for(client.disconnect(), timeout=5)
        except (ClaudeSDKError, TimeoutError, OSError) as error:
            logger.warning("сессия не закрылась по-хорошему: %s", error)

    # --- внутреннее -----------------------------------------------------------

    def _ask_user_server(self) -> Any:
        """Сервер MCP с одним инструментом — он живёт внутри процесса движка, отдельный
        процесс и юникс-сокет, как было до SDK, не нужны."""
        assert self.on_choice is not None
        return create_sdk_mcp_server(name="bax", tools=[ask_user_tool(self.on_choice)])

    async def _read(self, client: Any) -> None:
        """Насос сообщений: всё, что присылает CLI, уходит агенту типизированными объектами."""
        try:
            async for message in client.receive_messages():
                session = str(getattr(message, "session_id", "") or "")
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    session = str(message.data.get("session_id") or session)
                if session:
                    self.session_id = session
                await self.on_event(message)
        except asyncio.CancelledError:
            raise
        except (ClaudeSDKError, OSError) as error:
            logger.warning("поток сообщений оборвался: %s", error)
        finally:
            # сессия закончилась сама (CLI вышел) — следующая задача поднимет её заново
            if self._client is client:
                self._client = None

    async def _call(self, what: str, action: Callable[[Any], Awaitable[None]]) -> None:
        client = self._client
        if client is None:
            raise ClaudeError("сессия Claude Code не запущена")
        try:
            await asyncio.wait_for(action(client), timeout=CONTROL_TIMEOUT)
        except (ClaudeSDKError, TimeoutError, OSError) as error:
            raise ClaudeError(f"CLI отказал в {what}: {error}") from error

    async def _control(self, request: dict) -> dict:
        """Управляющий запрос без публичной обёртки в SDK (см. `settings`)."""
        client = self._client
        if client is None:
            raise ClaudeError("сессия Claude Code не запущена")
        query = getattr(client, "_query", None)
        if query is None or not hasattr(query, "_send_control_request"):
            raise ClaudeError("в этой версии SDK нет управляющих запросов — проверьте обновление")
        try:
            answer = await query._send_control_request(request, timeout=CONTROL_TIMEOUT)
        except (ClaudeSDKError, TimeoutError, OSError) as error:
            raise ClaudeError(f"CLI не ответил на {request.get('subtype')}: {error}") from error
        return dict(answer or {})
