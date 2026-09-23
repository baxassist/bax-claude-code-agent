"""Поддельные сервер Бакса и Claude Code: агент проверяется целиком, без сети и подписки."""

from __future__ import annotations

import asyncio
import json
import secrets
import time
import uuid
from pathlib import Path

import pytest
import websockets
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeSDKError,
    RateLimitEvent,
    RateLimitInfo,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TaskStartedMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)

from bax_agent import protocol

AGENT_ID = "6f0e6a1c-0000-4000-8000-000000000001"
KEY_ID = "11111111-1111-1111-1111-111111111111"
SECRET = "секрет-этого-агента"
REGISTRATION = f"{AGENT_ID}:{KEY_ID}:{SECRET}"
INSTALL = "aaaaaaaa-0000-4000-8000-000000000009"
SESSION = "11112222-3333-4444-5555-666677778888"

#: Модели без уровней усилий: после перехода на такую CLI честно отдаёт effort = null
NO_EFFORT = ("haiku",)


class FakeQuery:
    """Внутренность SDK, через которую уходят управляющие запросы без публичной обёртки
    (`get_settings`, `apply_flag_settings`) — движок ходит туда же."""

    def __init__(self, client: FakeClient) -> None:
        self.client = client

    async def _send_control_request(self, request: dict, timeout: float = 60.0) -> dict:
        subtype = request.get("subtype")
        if subtype == "get_settings":
            # у модели без усилий — null, ровно как настоящий CLI
            return {"applied": {"model": self.client.model, "effort": self.client.effort or None}}
        if subtype == "apply_flag_settings":
            settings = request.get("settings") or {}
            if "effortLevel" in settings:
                self.client.effort = str(settings["effortLevel"])
            return {}
        raise ClaudeSDKError(f"нет такого управляющего запроса: {subtype}")


class FakeClient:
    """Подставной `ClaudeSDKClient`: отдаёт те же объекты сообщений, что настоящий SDK.

    Сам Claude Code не запускается — проверяется склейка движка: реплика → события → кадры.
    Разрешения и `ask_user` проверяются отдельно, вызовом колбэков (`test_questions.py`).
    """

    #: Все созданные клиенты: тест достаёт последний, чтобы проверить, что до CLI дошло
    instances: list[FakeClient] = []

    def __init__(self, options) -> None:
        self.options = options
        self.model = options.model or "claude-opus-5"
        self.effort = options.effort or ""
        self.permission_mode = options.permission_mode or ""
        self.session = str(options.resume or options.session_id or SESSION)
        self.connected = False
        self.stopped: list[str] = []
        self.interrupted = 0
        self._messages: asyncio.Queue = asyncio.Queue()
        self._query = FakeQuery(self)
        FakeClient.instances.append(self)

    # --- то, чем пользуется движок ------------------------------------------

    async def connect(self, prompt=None) -> None:
        self.connected = True
        self._put(SystemMessage(subtype="init",
                                data={"session_id": self.session, "model": self.model}))

    async def query(self, prompt, session_id: str = "default") -> None:
        text = prompt if isinstance(prompt, str) else ""
        if text.startswith("/"):
            # слэш-команду CLI обрабатывает локально, без обращения к модели
            if text.startswith("/effort "):
                self.effort = text.split(maxsplit=1)[1].strip()
            return
        if text == "сломайся":
            return self._put(self._result(is_error=True, result="не вышло"))
        if text == "фоном":
            self._put(TaskStartedMessage(subtype="task_started", data={}, task_id="bg1",
                                         description="долгая сборка", uuid=str(uuid.uuid4()),
                                         session_id=self.session))
            return self._put(self._result())
        if text == "долго":
            return  # ответа не будет: проверяем тайм-аут
        for chunk in ("сде", "лал"):
            self._put(StreamEvent(uuid=str(uuid.uuid4()), session_id=self.session, event={
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": chunk},
            }))
        self._put(AssistantMessage(model=self.model, content=[
            ThinkingBlock(thinking="думаю", signature=""),
            ToolUseBlock(id="t1", name="Bash", input={"description": "собрать проект"}),
            TextBlock(text=f"сделал: {text}"),
        ]))
        self._put(RateLimitEvent(
            uuid=str(uuid.uuid4()), session_id=self.session,
            rate_limit_info=RateLimitInfo(status="allowed", raw={"unifiedWindows": {
                "five_hour": {"utilization": 0.08, "resetsAt": 1789919400},
                "seven_day": {"utilization": 0.1, "resetsAt": 1790352000},
            }}),
        ))
        self._put(self._result(
            usage={"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 3},
            model_usage={self.model: {"contextWindow": 200000, "costUSD": 0.01}},
        ))

    async def receive_messages(self):
        while True:
            message = await self._messages.get()
            if message is None:
                return
            yield message

    async def interrupt(self) -> None:
        self.interrupted += 1

    async def set_model(self, model: str | None = None) -> None:
        if model == "нельзя":
            raise ClaudeSDKError("нет такой модели")
        self.model = model or ""
        if self.model in NO_EFFORT:
            self.effort = ""

    async def set_permission_mode(self, mode: str) -> None:
        self.permission_mode = mode

    async def stop_task(self, task_id: str) -> None:
        self.stopped.append(task_id)

    async def get_context_usage(self) -> dict:
        return {"totalTokens": 100, "maxTokens": 200000, "percentage": 0.05}

    async def disconnect(self) -> None:
        self.connected = False
        self._messages.put_nowait(None)

    # --- внутреннее ----------------------------------------------------------

    def _put(self, message) -> None:
        self._messages.put_nowait(message)

    def _result(self, **fields) -> ResultMessage:
        base = {"subtype": "success", "duration_ms": 5, "duration_api_ms": 4, "is_error": False,
                "num_turns": 1, "session_id": self.session, "total_cost_usd": 0.01,
                "usage": {}, "model_usage": None}
        if fields.get("is_error"):
            base |= {"subtype": "error", "total_cost_usd": 0, "duration_ms": 1, "duration_api_ms": 1}
        return ResultMessage(**(base | fields))


@pytest.fixture
def fake_sdk():
    """Подставной клиент вместо настоящего SDK — тот же объект отдаётся движку фабрикой."""
    FakeClient.instances.clear()
    return FakeClient


class FakeBax:
    """Сервер Бакса на стороне теста: делает рукопожатие и складывает кадры агента."""

    def __init__(self, secret: str = SECRET) -> None:
        self.secret = secret
        self.frames: asyncio.Queue[dict] = asyncio.Queue()
        self.url = ""
        self.connections = 0
        self.hello: dict = {}
        self._ws: websockets.ServerConnection | None = None
        self._server = None

    async def start(self) -> str:
        self._server = await websockets.serve(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/agent"
        return self.url

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, ws) -> None:
        self.connections += 1
        hello = json.loads(await ws.recv())
        self.hello = hello
        if hello.get("v") != protocol.VERSION:
            await ws.send(protocol.frame("error", code="unsupported_version", message="обновите агента"))
            return
        nonce, ts = secrets.token_hex(8), int(time.time())
        await ws.send(protocol.frame("challenge", nonce=nonce, ts=ts))
        auth = json.loads(await ws.recv())
        if auth.get("sign") != protocol.sign(self.secret, nonce, ts, hello.get("key_id", "")):
            await ws.send(protocol.frame("error", code="unauthorized", message="подпись не сошлась"))
            return
        await ws.send(protocol.frame("ready", user=str(uuid.uuid4()), agent=AGENT_ID))
        self._ws = ws
        async for message in ws:
            frame = json.loads(message)
            # релей проставляет поле agent сам — подражаем ему, чтобы тесты видели то же,
            # что увидит приложение
            frame.setdefault("agent", AGENT_ID)
            await self.frames.put(frame)

    async def send(self, type_: str, **fields) -> None:
        assert self._ws is not None, "агент ещё не подключился"
        await self._ws.send(protocol.frame(type_, **fields))

    async def wait_status(self, state: str, timeout: float = 5.0) -> dict:
        """Состояний приходит несколько подряд — ждём нужное, а не первое попавшееся."""
        async with asyncio.timeout(timeout):
            while True:
                frame = await self.frames.get()
                if frame.get("type") == "status" and frame.get("state") == state:
                    return frame

    async def wait_stats(self, timeout: float = 5.0, **expect) -> dict:
        """Сводок приходит несколько подряд (запуск сессии, конец хода, смена настройки) —
        ждём ту, где нужные поля уже такие, как ожидаем, а не первую попавшуюся."""
        async with asyncio.timeout(timeout):
            while True:
                frame = await self.frames.get()
                if frame.get("type") != "stats":
                    continue
                if all(frame.get(field) == value for field, value in expect.items()):
                    return frame

    async def drain(self, pause: float = 0.2) -> None:
        """Выбросить всё, что уже пришло: дальше проверяем только новые кадры."""
        await asyncio.sleep(pause)
        while not self.frames.empty():
            self.frames.get_nowait()

    async def wait(self, type_: str, timeout: float = 5.0) -> dict:
        """Ждём кадр нужного вида, остальные пропускаем: порядок соседних кадров не проверяем."""
        async with asyncio.timeout(timeout):
            while True:
                frame = await self.frames.get()
                if frame.get("type") == type_:
                    return frame

    async def collect(self, until: str, timeout: float = 5.0) -> list[dict]:
        """Все кадры до кадра `until` включительно — когда важен состав, а не отдельный кадр."""
        found = []
        async with asyncio.timeout(timeout):
            while True:
                frame = await self.frames.get()
                found.append(frame)
                if frame.get("type") == until:
                    return found


@pytest.fixture
async def bax():
    server = FakeBax()
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def running(config_file, fake_sdk):
    """Запущенный движок с одним зарегистрированным агентом: живёт, пока идёт тест."""
    from bax_agent.config import load
    from bax_agent.engine import Engine

    engine = Engine(load(config_file), client_factory=fake_sdk)
    task = asyncio.create_task(engine.run())
    try:
        yield engine
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(*(agent.close() for agent in engine.agents.values()),
                             return_exceptions=True)


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "проект"
    folder.mkdir()
    return folder


@pytest.fixture
def config_file(tmp_path: Path, project_dir: Path, bax) -> Path:
    """Конфиг движка плюс один агент, зарегистрированный как это делает `bax-agent add`."""
    from bax_agent.config import add_agent, load

    file = tmp_path / "config.toml"
    file.write_text(f'[engine]\nname = "Тест"\nserver = "{bax.url}"\n', encoding="utf-8")
    add_agent(load(file), REGISTRATION, project_dir)
    return file
