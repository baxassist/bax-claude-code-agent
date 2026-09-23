"""«Больше не спрашивать» — точечным правилом самого Claude Code.

Раньше один тап на карточке `Bash` разрешал **любые** команды до конца сессии: движок
запоминал имя инструмента и молча пропускал всё, что им делается. Это слишком широко —
человек разрешал «npm test», а получал разрешение на «rm -rf».

Теперь из вопроса делается правило CLI. Оно уходит обратно тем же ответом на вопрос
(`updated_permissions` → `addRules`, область `session`), и дальше решения принимает сам
Claude Code — по своим правилам, а не по нашей памяти.

Главный источник правила — подсказки самого CLI в вопросе (`context.suggestions`, они же
видны в терминале). Здесь остаётся запасной путь на случай, когда подсказок нет, и — важнее —
запрет: командам из `_NEVER` правила не даём никогда, сколько бы раз их ни разрешали.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from claude_agent_sdk import PermissionUpdate
from claude_agent_sdk.types import PermissionRuleValue

#: Сколько слов команды берём в правило: «npm test», «git status», «pytest»
_COMMAND_WORDS = 2

#: Команды, у которых второе слово — уже аргумент, а не подкоманда
_SINGLE_WORD = {"pytest", "ruff", "mypy", "ls", "cat", "grep", "rg", "find", "make", "swift"}

#: Команды, которым правила не даём никогда: разрешить их «навсегда» — значит однажды
#: удалить не то. Такое спрашиваем каждый раз, сколько бы раз ни разрешали
_NEVER = {"rm", "rmdir", "sudo", "dd", "mkfs", "shutdown", "reboot", "halt",
          "chmod", "chown", "mv", "kill", "pkill", "killall", "eval", "exec"}

#: Символы оболочки: с ними «npm test:*» разрешило бы и всё, что дописано через && или |
_SHELL = ("&&", "||", ";", "|", ">", "<", "$(", "`", "$")


@dataclass(frozen=True)
class Rule:
    """Правило для CLI и его человеческая подпись для кнопки в приложении."""

    tool: str
    content: str = ""
    #: Что написать на кнопке: «больше не спрашивать про npm test *»
    label: str = ""

    def as_permission_update(self) -> PermissionUpdate:
        """Ответ для Claude Code: добавить правило на эту сессию."""
        return PermissionUpdate(
            type="addRules",
            rules=[PermissionRuleValue(tool_name=self.tool, rule_content=self.content or None)],
            behavior="allow",
            destination="session",  # на эту сессию: в настройки на диске не лезем
        )


def rule_for(tool: str, tool_input: dict, *, root: Path | None = None) -> Rule | None:
    """Правило по вопросу: чем оно у́же, тем лучше. Не получилось — вернём None,
    и «больше не спрашивать» останется прежним (на весь инструмент)."""
    if not tool:
        return None
    if tool == "Bash":
        return _bash_rule(str(tool_input.get("command") or ""))
    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        return _path_rule(tool, str(tool_input.get("file_path") or ""), root)
    if tool in ("WebFetch", "WebSearch"):
        return _web_rule(tool, str(tool_input.get("url") or ""))
    # остальным инструментам точечного правила не придумать — разрешаем инструмент целиком
    return Rule(tool=tool, label=f"больше не спрашивать про {tool}")


def _bash_rule(command: str) -> Rule | None:
    """Из команды берём начало: «npm test …» → «npm test:*». Так разрешается именно она,
    а не весь Bash. Разобрать не вышло (кавычки, подстановки) — правила не будет."""
    # цепочка или подстановка: правило по началу разрешило бы и дописанное справа
    if any(symbol in command for symbol in _SHELL):
        return None
    try:
        words = [word for word in shlex.split(command) if word]
    except ValueError:
        return None
    if not words or words[0] in _NEVER:
        return None
    if any(symbol in words[0] for symbol in "*?"):
        return None
    limit = 1 if words[0] in _SINGLE_WORD else _COMMAND_WORDS
    prefix = " ".join(words[:limit])
    return Rule(tool="Bash", content=f"{prefix}:*", label=f"больше не спрашивать про «{prefix} …»")


def _path_rule(tool: str, file_path: str, root: Path | None) -> Rule | None:
    """Правка файлов разрешается по каталогу: «весь src», а не «все файлы на диске»."""
    if not file_path:
        return None
    folder = Path(file_path).parent
    if root is not None:
        try:
            relative = folder.resolve().relative_to(root.resolve())
            content = f"./{relative}/**" if str(relative) != "." else "./**"
        except ValueError:
            content = f"{folder}/**"  # вне проекта — по полному пути
    else:
        content = f"{folder}/**"
    return Rule(tool=tool, content=content, label=f"больше не спрашивать про файлы в {folder.name or '/'}")


def _web_rule(tool: str, url: str) -> Rule | None:
    """Сеть разрешается по домену: «docs.python.org», а не «весь интернет»."""
    host = urlparse(url).hostname or ""
    if not host:
        return None
    return Rule(tool=tool, content=f"domain:{host}", label=f"больше не спрашивать про {host}")


def label_for(update: PermissionUpdate) -> str:
    """Подпись кнопки «больше не спрашивать» по правилу — своему или подсказанному CLI."""
    for rule in update.rules or []:
        content = (rule.rule_content or "").strip()
        return f"больше не спрашивать про «{content}»" if content \
            else f"больше не спрашивать про {rule.tool_name}"
    return ""
