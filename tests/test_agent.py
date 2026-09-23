"""Движок целиком: подключился ключом агента, сделал задачу, остановился.

Claude Code и сервер Бакса здесь поддельные (`conftest.py`), но путь кадра настоящий:
websocket → агент → процесс → события → websocket.
"""

from __future__ import annotations

import asyncio

from bax_agent import protocol
from bax_agent.config import add_agent, load
from bax_agent.engine import Engine
from tests.conftest import AGENT_ID, KEY_ID


async def test_tells_about_itself_in_hello(running, bax):
    """Кадра «вот мои агенты» больше нет: соединение открыто ключом одного агента.
    О себе движок сообщает в `hello` — установку, версию и каталог."""
    status = await bax.wait("status")
    assert status["state"] == "ready" and status["agent"] == AGENT_ID

    assert bax.hello["key_id"] == KEY_ID and bax.hello["engine"] == "claude_code"
    assert bax.hello["install_id"] and bax.hello["install_name"] == "Тест"
    assert bax.hello["path"].endswith("проект")


async def test_task_goes_through(running, bax):
    await bax.wait("status")
    await bax.send("run", agent=AGENT_ID, text="собери проект")

    frames = await bax.collect(until="done")
    kinds = [(f["type"], f.get("kind")) for f in frames]
    assert ("message", "user") in kinds, "запрос пользователя возвращается в ленту"
    assert ("message", "assistant") in kinds
    assert ("message", "thinking") in kinds, "рассуждения приходят отдельным видом"
    assert ("message", "tool") in kinds, "вызов инструмента — строкой активности"

    answer = next(f for f in frames if f["type"] == "message" and f.get("kind") == "assistant")
    assert answer["text"] == "сделал: собери проект"

    chunks = "".join(f["chunk"] for f in frames if f["type"] == "delta")
    assert chunks == "сделал", "ответ шёл кусками на лету"
    assert all(f["id"] == answer["id"] for f in frames if f["type"] == "delta"), \
        "куски и готовое сообщение — один id, иначе приложение покажет ответ дважды"

    done = frames[-1]
    assert done["duration_ms"] == 5 and done["cost"] == 0.01
    assert done["context"] == {"used": 100, "max": 200000}, "занятый контекст и окно модели"

    assert (await bax.wait("status"))["state"] == "ready", "после ответа агент снова свободен"


async def test_header_gets_model_effort_context_limits(running, bax):
    """В шапке агента всегда видно модель, усилие, контекст и остаток лимитов (заказчик 20.09)."""
    await bax.wait("status")
    await bax.send("run", agent=AGENT_ID, text="собери проект")
    frames = await bax.collect(until="done")

    stats = [f for f in frames if f["type"] == "stats"]
    assert stats, "сводка для шапки приходит"
    last = stats[-1]
    assert last["model"] == "claude-opus-5"
    assert last["limits"]["five_hour"] == {"used_pct": 8, "resets_at": 1789919400}, \
        "лимиты приходят процентами: CLI отдаёт долю"
    assert last["limits"]["seven_day"]["used_pct"] == 10


async def test_ids_grow_and_one_entry_shares_id(running, bax):
    """id — номер записи в файле сессии, а не номер сообщения: рассуждение, вызов инструмента
    и текст одного ответа лежат в файле одной строкой и приходят с одним id."""
    await bax.wait("status")
    await bax.send("run", agent=AGENT_ID, text="раз")
    first = await bax.collect(until="done")
    await bax.send("run", agent=AGENT_ID, text="два")
    second = await bax.collect(until="done")

    def ids(frames, kind=None):
        return [f["id"] for f in frames
                if f["type"] == "message" and (kind is None or f.get("kind") == kind)]

    assert ids(first) == sorted(ids(first)), "номера записей не идут назад"
    assert min(ids(second)) > max(ids(first)), "второй ход продолжает нумерацию"

    answer = [f for f in first if f["type"] == "message" and f.get("kind") != "user"]
    assert len({f["id"] for f in answer}) == 1, "одна запись ответа — один id"
    assert ids(first, "user")[0] < answer[0]["id"]


async def test_model_change_is_reported_in_stats(running, bax):
    """Сменили модель — приложение узнаёт об этом кадром stats, а не через сервер."""
    await bax.wait("status")
    await bax.send("model.set", agent=AGENT_ID, model="sonnet")
    await bax.wait("message")          # «Модель: sonnet» в ленте
    assert (await bax.wait("stats"))["model"] == "sonnet"


async def test_busy_agent_rejects_second_task(running, bax):
    await bax.wait("status")
    # «долго» — задача без ответа: пока она идёт, поток занят по-настоящему
    await bax.send("run", agent=AGENT_ID, text="долго")
    await bax.wait_status("busy")
    await bax.send("run", agent=AGENT_ID, text="два")
    assert (await bax.wait("error"))["code"] == "busy"


async def test_stop_ends_the_turn(running, bax):
    await bax.wait("status")
    await bax.send("run", agent=AGENT_ID, text="собери проект")
    await bax.wait("done")
    await bax.send("cancel", agent=AGENT_ID, scope="turn")

    event = await bax.wait("message")
    assert event["kind"] == "event" and "Остановлено" in event["text"]
    assert (await bax.wait("status"))["state"] == "ready"


async def test_ping_is_answered(running, bax):
    await bax.wait("status")
    await bax.send("ping")
    assert (await bax.wait("pong"))["type"] == "pong"


async def test_error_from_claude_reaches_app(running, bax):
    await bax.wait("status")
    await bax.send("run", agent=AGENT_ID, text="сломайся")
    frames = await bax.collect(until="error")
    assert frames[-1]["message"] == "не вышло"


async def test_wrong_key_stops_only_that_agent(tmp_path, project_dir, fake_sdk, bax):
    """Неверный ключ сам собой не починится: движок перестаёт переподключаться этим агентом.
    Остальные агенты установки работают дальше — здесь он один, и движок просто выходит."""
    file = tmp_path / "config.toml"
    file.write_text(f'[engine]\nname = "Тест"\nserver = "{bax.url}"\n', encoding="utf-8")
    add_agent(load(file), f"{AGENT_ID}:{KEY_ID}:не-тот-секрет", project_dir)

    engine = Engine(load(file), client_factory=fake_sdk)
    await asyncio.wait_for(engine.run(), timeout=5)
    assert bax.connections == 1, "повторных попыток быть не должно"


async def test_reconnects_after_break(config_file, fake_sdk, bax, monkeypatch):
    """Обрыв связи — обычное дело: движок возвращается сам и снова здоровается."""
    monkeypatch.setattr(protocol, "BACKOFF", (0,))
    engine = Engine(load(config_file), client_factory=fake_sdk)
    task = asyncio.create_task(engine.run())
    try:
        await bax.wait("status")
        await bax._ws.close()
        again = await bax.wait("status", timeout=10)
        assert again["agent"] == AGENT_ID
        assert bax.connections == 2
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(*(agent.close() for agent in engine.agents.values()),
                             return_exceptions=True)


def test_rules_are_narrow():
    """«Больше не спрашивать» разрешает именно эту команду, а не весь Bash (заказчик 21.09):
    раньше один тап на карточке открывал любые команды до конца сессии."""
    from pathlib import Path

    from bax_agent import rules

    bash = rules.rule_for("Bash", {"command": "npm test -- --watch"})
    assert bash.tool == "Bash" and bash.content == "npm test:*"
    assert bash.as_permission_update().to_dict() == {
        "type": "addRules",
        "rules": [{"toolName": "Bash", "ruleContent": "npm test:*"}],
        "behavior": "allow",
        "destination": "session",
    }

    # команды без подкоманд не режем пополам
    assert rules.rule_for("Bash", {"command": "pytest -q"}).content == "pytest:*"
    # подстановки и цепочки в правило не превращаем — там легко разрешить лишнее
    assert rules.rule_for("Bash", {"command": "rm -rf $HOME"}) is None
    assert rules.rule_for("Bash", {"command": "make build && rm -rf /"}) is None

    # правка файлов разрешается по каталогу проекта, а не по всему диску
    edit = rules.rule_for("Write", {"file_path": "/p/api/src/main.py"}, root=Path("/p/api"))
    assert edit.content == "./src/**"

    # сеть — по домену
    web = rules.rule_for("WebFetch", {"url": "https://docs.python.org/3/library/"})
    assert web.content == "domain:docs.python.org"
