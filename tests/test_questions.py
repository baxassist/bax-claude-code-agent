"""Вопросы и разрешения: путь от Claude Code до кнопки в приложении и обратно.

Настоящий CLI здесь не нужен: с переходом на SDK разрешение — это колбэк `can_use_tool`,
а смысловой вопрос — вызов MCP-инструмента `ask_user` внутри самого движка. Тесты зовут
их напрямую: ровно то, что позовёт SDK.
"""

from __future__ import annotations

import asyncio

from claude_agent_sdk import PermissionUpdate, ToolPermissionContext
from claude_agent_sdk.types import PermissionRuleValue

from bax_agent.claude import ask_user_tool
from tests.conftest import AGENT_ID


def suggestion(tool: str = "Bash", content: str = "ls:*") -> PermissionUpdate:
    """Подсказка правила, как её присылает сам CLI: в терминале это «не спрашивать для ls»."""
    return PermissionUpdate(
        type="addRules",
        rules=[PermissionRuleValue(tool_name=tool, rule_content=content)],
        behavior="allow",
        destination="localSettings",  # CLI предлагает записать в настройки — мы сузим до сессии
    )


async def test_permission_reaches_app(running, bax):
    """Разрешение долетает до приложения кадром `question`, поток встаёт в «жду ответа»."""
    await bax.wait("status")
    agent = running.agents[AGENT_ID]

    task = asyncio.create_task(agent._on_permission(
        "Bash", {"command": "git push"},
        ToolPermissionContext(tool_use_id="q1", description="Отправить ветку"),
    ))
    question = await bax.wait("question")
    assert question["kind"] == "permission" and question["question_id"] == "q1"
    assert question["tool"] == "Bash" and question["input"] == {"command": "git push"}
    assert (await bax.wait("status"))["state"] == "waiting"

    await bax.send("answer", agent=AGENT_ID, question_id="q1", verdict="allow")
    verdict = await asyncio.wait_for(task, timeout=5)
    assert verdict.behavior == "allow" and verdict.updated_input == {"command": "git push"}
    assert verdict.updated_permissions is None, "без «запомнить» правил не добавляем"


async def test_deny_comes_back_as_deny(running, bax):
    await bax.wait("status")
    agent = running.agents[AGENT_ID]

    task = asyncio.create_task(agent._on_permission(
        "Bash", {"command": "rm -rf /"}, ToolPermissionContext(tool_use_id="q1")))
    await bax.wait("question")
    await bax.send("answer", agent=AGENT_ID, question_id="q1", verdict="deny")
    verdict = await asyncio.wait_for(task, timeout=5)
    assert verdict.behavior == "deny" and "запретил" in verdict.message


async def test_choice_question(running, bax):
    """`ask_user` — смысловой вопрос с вариантами: ответ уходит обратно в модель текстом."""
    await bax.wait("status")
    agent = running.agents[AGENT_ID]

    task = asyncio.create_task(agent._on_choice("Какой вариант делаем?", ["первый", "второй"]))
    question = await bax.wait("question")
    assert question["kind"] == "choice" and question["options"] == ["первый", "второй"]
    assert question["rule"] == "", "у смыслового вопроса «больше не спрашивать» не бывает"

    await bax.send("answer", agent=AGENT_ID, question_id=question["question_id"],
                   verdict="choice", option="первый")
    assert await asyncio.wait_for(task, timeout=5) == "первый"


async def test_remember_uses_cli_suggestion(running, bax):
    """«Больше не спрашивать» уходит подсказкой самого CLI — той же, что в терминале,
    но суженной до этой сессии: в файлы настроек проекта движок не пишет."""
    await bax.wait("status")
    agent = running.agents[AGENT_ID]

    task = asyncio.create_task(agent._on_permission(
        "Bash", {"command": "ls -la"},
        ToolPermissionContext(tool_use_id="q1", suggestions=[suggestion()]),
    ))
    question = await bax.wait("question")
    assert "ls" in question["rule"], "на кнопке видно, что именно запомнится"

    await bax.send("answer", agent=AGENT_ID, question_id="q1", verdict="allow", remember=True)
    verdict = await asyncio.wait_for(task, timeout=5)
    assert [update.to_dict() for update in verdict.updated_permissions] == [{
        "type": "addRules",
        "rules": [{"toolName": "Bash", "ruleContent": "ls:*"}],
        "behavior": "allow",
        "destination": "session",
    }]


async def test_remember_falls_back_to_own_rule(running, bax):
    """Подсказки нет — правило строим сами, и оно всё равно точечное: «ls …», а не весь Bash
    (заказчик 21.09: один тап на карточке не должен открывать любые команды)."""
    await bax.wait("status")
    agent = running.agents[AGENT_ID]

    task = asyncio.create_task(agent._on_permission(
        "Bash", {"command": "ls -la"}, ToolPermissionContext(tool_use_id="q1")))
    await bax.wait("question")
    await bax.send("answer", agent=AGENT_ID, question_id="q1", verdict="allow", remember=True)
    verdict = await asyncio.wait_for(task, timeout=5)

    rule = verdict.updated_permissions[0].to_dict()
    assert rule["rules"] == [{"toolName": "Bash", "ruleContent": "ls:*"}]
    assert rule["destination"] == "session"


async def test_dangerous_command_never_gets_a_rule(running, bax):
    """Опасной команде правила не даём никогда — даже если CLI его подсказал: такое
    спрашивается каждый раз, сколько бы раз ни разрешали."""
    await bax.wait("status")
    agent = running.agents[AGENT_ID]

    task = asyncio.create_task(agent._on_permission(
        "Bash", {"command": "rm -rf build"},
        ToolPermissionContext(tool_use_id="q1", suggestions=[suggestion(content="rm:*")]),
    ))
    question = await bax.wait("question")
    assert question["rule"] == "", "нет правила — нет и кнопки «больше не спрашивать»"

    await bax.send("answer", agent=AGENT_ID, question_id="q1", verdict="allow", remember=True)
    verdict = await asyncio.wait_for(task, timeout=5)
    assert verdict.updated_permissions is None


async def test_no_answer_means_deny(running, bax):
    """Молчание — не «да»: не ответили за отведённое время, действие запрещается."""
    await bax.wait("status")
    agent = running.agents[AGENT_ID]
    agent.question_timeout = 0.2

    verdict = await asyncio.wait_for(agent._on_permission(
        "Bash", {"command": "ls"}, ToolPermissionContext(tool_use_id="q1")), timeout=5)
    assert verdict.behavior == "deny" and "вовремя" in verdict.message
    await bax.wait_status("ready")   # поток не остаётся в «жду ответа»


async def test_ask_user_lives_inside_the_engine(running, bax):
    """Инструмент `ask_user` — внутри процесса движка, а не отдельным процессом с юникс-сокетом,
    как было до SDK. Зовём его так же, как позовёт Claude Code."""
    await bax.wait("status")
    agent = running.agents[AGENT_ID]
    options = agent.process.options()

    assert list(options.mcp_servers) == ["bax"], "у сессии только наш сервер (strict_mcp_config)"
    assert options.mcp_servers["bax"]["type"] == "sdk"
    assert options.can_use_tool == agent._on_permission, "разрешения — колбэком, а не через MCP"

    instrument = ask_user_tool(agent._on_choice)
    assert instrument.name == "ask_user"
    task = asyncio.create_task(instrument.handler({"question": "Ну как?", "options": ["так"]}))
    question = await bax.wait("question")
    await bax.send("answer", agent=AGENT_ID, question_id=question["question_id"],
                   verdict="choice", option="так")
    answer = await asyncio.wait_for(task, timeout=5)
    assert answer == {"content": [{"type": "text", "text": "так"}]}


async def test_options_follow_the_document(running, bax):
    """Опции запуска — те, на которые рассчитан движок. Особенно системная
    подсказка: без пресета SDK передаёт CLI пустую, и это перестаёт быть Claude Code."""
    await bax.wait("status")
    options = running.agents[AGENT_ID].process.options()

    assert options.system_prompt == {"type": "preset", "preset": "claude_code"}
    assert options.include_partial_messages, "ответ на лету"
    assert options.strict_mcp_config
    assert options.setting_sources is None, "источники настроек — как у CLI, вместе с CLAUDE.md"
    assert options.cwd.endswith("проект")
    assert options.extra_args == {}, "--restricted только у агентов, заведённых с ним"


async def test_answer_to_closed_question(running, bax):
    await bax.wait("status")
    await bax.send("answer", agent=AGENT_ID, question_id="нет-такого", verdict="allow")
    assert (await bax.wait("error"))["code"] == "not_found"
