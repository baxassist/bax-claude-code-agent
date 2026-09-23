"""Движок целиком: по соединению на каждого зарегистрированного агента.

Термины: **движок** — эта программа вместе с Claude Code, **агент** —
то, что пользователь завёл и назвал в приложении, **установка** — запущенный движок.

Каждый агент живёт своим соединением: у него свой ключ, и окончательная ошибка (ключ
перевыпустили, агента удалили, строку вставили не в тот движок) гасит только его —
остальные агенты этой установки работают дальше.
"""

from __future__ import annotations

import asyncio
import logging

from . import __version__
from .agents import Agent
from .config import Config, Registration, install_id
from .connection import HandshakeError, Link
from .events import EventLog

logger = logging.getLogger("bax.engine")


class Engine:
    def __init__(self, config: Config, *, claude_path: str = "",
                 client_factory: object | None = None) -> None:
        self.config = config
        # пусто — сессии поднимаются CLI из пакета SDK
        self.claude_path = claude_path or config.claude_path
        self.client_factory = client_factory
        self.events = EventLog(config.events_file)
        # общий предел одновременных задач: компьютер у пользователя один на всех агентов
        self.slots = asyncio.Semaphore(max(1, config.max_parallel))
        self.install = install_id(config)
        self.agents: dict[str, Agent] = {}
        self.links: dict[str, Link] = {}
        for registration in config.agents:
            self._prepare(registration)

    def _prepare(self, registration: Registration) -> None:
        link = Link(
            self.config.server, registration.key_id, registration.secret, __version__,
            install_id=self.install, install_name=self.config.name, path=str(registration.path),
        )
        self.links[registration.agent_id] = link
        self.agents[registration.agent_id] = Agent(
            registration, link,
            claude_path=self.claude_path,
            events=self.events,
            log_file=self.config.log_file,
            task_timeout_min=self.config.task_timeout_min,
            question_timeout_min=self.config.question_timeout_min,
            semaphore=self.slots,
            client_factory=self.client_factory,
        )

    async def run(self) -> None:
        """Держит все соединения, пока движок не остановят."""
        if not self.agents:
            logger.warning("не зарегистрирован ни один агент — нечего обслуживать. "
                           "Заведите агента в приложении и выполните `bax-agent add`")
            return
        try:
            await asyncio.gather(*(self._serve(agent_id) for agent_id in list(self.agents)))
        finally:
            await asyncio.gather(*(agent.close() for agent in self.agents.values()),
                                 return_exceptions=True)

    async def _serve(self, agent_id: str) -> None:
        """Одно соединение, один агент. Окончательная ошибка гасит только его."""
        agent, link = self.agents[agent_id], self.links[agent_id]

        async def on_ready(frame: dict) -> None:
            agent.apply_start_settings(frame.get("settings") or {})
            await agent.hello()

        async def on_frame(frame: dict) -> None:
            try:
                await agent.handle(frame)
            except Exception as error:  # один сломанный кадр не должен ронять агента целиком
                logger.exception("кадр %s не обработался", frame.get("type"))
                await link.send("error", code="internal", message=str(error))

        try:
            await link.run(on_ready, on_frame)
        except HandshakeError as error:
            logger.error("агент %s больше не подключается: %s", agent_id, error.message)
            await agent.close()
        except asyncio.CancelledError:
            raise
