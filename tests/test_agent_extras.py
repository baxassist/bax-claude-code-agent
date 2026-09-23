"""Остальное из потока: смена модели и усилий, настройки, сессии, картинки, тайм-аут, команды."""

from __future__ import annotations

import asyncio
import base64
import json

from bax_agent.agents import INBOX
from tests.conftest import AGENT_ID


async def test_model_switch_goes_to_live_process(running, bax):
    """Модель меняется управляющим запросом: процесс живёт, токены не тратятся."""
    await bax.wait("status")
    await bax.send("model.set", agent=AGENT_ID, model="sonnet")
    said = await bax.wait("message")
    assert said["kind"] == "event" and said["text"] == "Модель: sonnet"
    assert (await bax.wait("stats"))["model"] == "sonnet"


async def test_model_switch_falls_back_to_restart(running, bax):
    """CLI отказал — перезапускаем процесс с новым флагом и той же сессией."""
    await bax.wait("status")
    stream = running.agents[AGENT_ID]
    await stream.process.start()
    session_before = stream.process.session_id

    await bax.send("model.set", agent=AGENT_ID, model="нельзя")
    said = await bax.wait("message")
    assert said["text"] == "Модель: нельзя"
    assert stream.process.session_id == session_before, "сессия сохраняется при перезапуске"
    assert stream.process.alive


async def test_effort_switch(running, bax):
    await bax.wait("status")
    await bax.send("effort.set", agent=AGENT_ID, effort="high")
    said = await bax.wait("message")
    assert said["text"] == "Уровень усилий: high"
    assert (await bax.wait_stats(effort="high"))["effort"] == "high"


async def test_effort_disappears_for_model_without_efforts(running, bax):
    """Сменили модель на ту, у которой уровней усилий нет, — в шапке усилия больше нет.

    Своя переменная об этом не знает: сводка берётся из `get_settings`, где CLI честно
    отдаёт `effort: null`.
    """
    await bax.wait("status")
    await bax.send("effort.set", agent=AGENT_ID, effort="high")
    await bax.wait_stats(effort="high")

    await bax.send("model.set", agent=AGENT_ID, model="haiku")
    assert (await bax.wait_stats(model="haiku"))["effort"] == ""


async def test_context_comes_from_cli(running, bax):
    """Сколько занято контекста, знает сам CLI — размеры окон мы не хардкодим."""
    await bax.wait("status")
    await bax.send("run", agent=AGENT_ID, text="собери проект")
    frames = await bax.collect(until="done")

    stats = [frame for frame in frames if frame["type"] == "stats"][-1]
    assert stats["context"] == {"used": 100, "max": 200000}


async def test_permission_mode_change(running, bax):
    await bax.wait("status")
    await bax.send("settings.set", agent=AGENT_ID, permission_mode="manual", push=False)
    said = await bax.wait("message")
    assert said["text"] == "Режим разрешений: manual"
    assert running.agents[AGENT_ID].permission_mode == "manual"
    assert running.agents[AGENT_ID].push_enabled is False


async def test_background_tasks_are_stopped_separately(running, bax):
    """Прерывание хода фоновые задачи не гасит — для них отдельная команда."""
    await bax.wait("status")
    await bax.send("run", agent=AGENT_ID, text="фоном")
    await bax.wait("done")
    assert running.agents[AGENT_ID]._backgrounds == ["bg1"]

    await bax.send("cancel", agent=AGENT_ID, scope="background")
    said = await bax.wait("message")
    assert said["text"] == "Фоновые задачи остановлены: 1"
    assert running.agents[AGENT_ID]._backgrounds == []


async def test_task_timeout_frees_the_stream(running, bax):
    """Задача не может идти вечно: иначе поток занят, а человек не понимает почему."""
    await bax.wait("status")
    running.agents[AGENT_ID].task_timeout = 0.2
    await bax.send("run", agent=AGENT_ID, text="долго")
    await bax.wait_status("busy")

    said = await bax.wait("message", timeout=10)
    assert said["kind"] == "error" and "прервана" in said["text"]
    await bax.wait_status("ready")  # поток освободился, дождёмся именно этого состояния
    assert running.agents[AGENT_ID].state == "ready"


async def test_attachment_is_saved_into_project(running, bax, project_dir):
    """Картинка кладётся в служебную папку проекта, а модели уходит путь, не base64."""
    await bax.wait("status")
    data = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
    await bax.send("run", agent=AGENT_ID, text="что на картинке",
                   attachments=[{"name": "снимок.png", "mime": "image/png", "data": data}])

    frames = await bax.collect(until="done")
    answer = next(f for f in frames if f["type"] == "message" and f.get("kind") == "assistant")
    assert INBOX in answer["text"] and "снимок.png" in answer["text"], "модель получила путь к файлу"

    saved = list((project_dir / INBOX).iterdir())
    assert len(saved) == 1 and saved[0].read_bytes().startswith(b"\x89PNG")


async def test_attachment_name_cannot_escape(running, bax, project_dir):
    await bax.wait("status")
    data = base64.b64encode("нет".encode()).decode()
    await bax.send("run", agent=AGENT_ID, text="файл",
                   attachments=[{"name": "../../секрет.txt", "data": data}])
    await bax.collect(until="done")
    assert not (project_dir.parent / "секрет.txt").exists(), "путь из имени файла не выходит наружу"
    assert len(list((project_dir / INBOX).iterdir())) == 1


async def test_sessions_and_switch(running, bax, project_dir, tmp_path, monkeypatch):
    """Список сессий берётся с диска, выбор переключает процесс."""
    from pathlib import Path

    from bax_agent import history

    monkeypatch.setattr(Path, "home", lambda: tmp_path / "домик")
    file = history.session_file(project_dir, "старая")
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps({"type": "user", "message": {"content": "прошлая задача"}}) + "\n",
                    encoding="utf-8")

    await bax.wait("status")
    await bax.send("sessions.list", agent=AGENT_ID)
    sessions = await bax.wait("sessions")
    assert [item["session"] for item in sessions["items"]] == ["старая"]
    assert sessions["items"][0]["title"] == "прошлая задача"

    await bax.send("session.select", agent=AGENT_ID, session="старая")
    said = await bax.wait("message")
    assert said["text"] == "Сессия переключена"
    assert running.agents[AGENT_ID].process.session_id in ("старая", "11112222-3333-4444-5555-666677778888")


async def test_command_tree_and_unknown(running, bax, project_dir):
    (project_dir / "файл.txt").write_text("привет", encoding="utf-8")
    await bax.wait("status")

    await bax.send("command", agent=AGENT_ID, name="tree", args={"depth": 1})
    said = await bax.wait("message")
    assert "файл.txt" in said["text"]

    await bax.send("command", agent=AGENT_ID, name="неизвестно", args={})
    assert (await bax.wait("error"))["code"] == "not_found"


async def test_command_file_comes_in_parts(running, bax, project_dir):
    (project_dir / "файл.txt").write_text("привет", encoding="utf-8")
    await bax.wait("status")
    await bax.send("command", agent=AGENT_ID, name="file", args={"path": "файл.txt"})
    part = await bax.wait("file")
    assert part["name"] == "файл.txt" and base64.b64decode(part["data"]).decode() == "привет"


async def test_command_file_outside_project(running, bax):
    await bax.wait("status")
    await bax.send("command", agent=AGENT_ID, name="file", args={"path": "../../etc/passwd"})
    assert (await bax.wait("error"))["code"] == "forbidden"


async def test_resources_has_everything_for_the_screen(running, bax):
    await bax.wait("status")
    await bax.send("resources.get", agent=AGENT_ID)
    data = await bax.wait("resources", timeout=20)
    assert data["machine"]["cpu_count"] > 0 and data["machine"]["disk_total"] > 0
    assert data["models"] and data["efforts"], "списки спрашиваются у CLI"
    assert data["state"] == "ready" and data["agent"] == AGENT_ID


async def test_events_survive_reopening(running, bax):
    """Свои события агента в файлы Claude Code не попадают — агент помнит их сам."""
    await bax.wait("status")
    await bax.send("run", agent=AGENT_ID, text="раз")
    await bax.wait("done")
    await bax.send("cancel", agent=AGENT_ID, scope="turn")
    await bax.wait("message")
    await bax.drain()  # хвост кадров хода нам не нужен: проверяем то, что придёт на subscribe

    await bax.send("subscribe", agent=AGENT_ID)
    frames = await bax.collect(until="status")
    texts = [f.get("text") for f in frames if f["type"] == "message"]
    assert "Остановлено пользователем" in texts


async def test_unsupported_frame_is_explained(running, bax):
    await bax.wait("status")
    await bax.send("неизвестный.кадр", agent=AGENT_ID)
    error = await bax.wait("error")
    assert error["code"] == "internal" and "не умеет" in error["message"]


async def test_agents_share_the_task_limit(running, bax):
    """Общий предел одновременных задач: компьютер один на всех агентов."""
    await bax.wait("status")
    assert running.slots._value >= 1
    await bax.send("run", agent=AGENT_ID, text="долго")
    await bax.wait("status")
    await asyncio.sleep(0.1)
    assert running.slots._value == running.config.max_parallel - 1, "место в очереди занято"
