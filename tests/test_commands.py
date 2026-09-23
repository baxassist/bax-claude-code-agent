"""Расширенные команды и защита путей: наружу каталога проекта агент не ходит."""

from __future__ import annotations

import base64

import pytest

from bax_agent import commands
from bax_agent.commands import CommandError


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "проект"
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("print('привет')\n", encoding="utf-8")
    (root / "README.md").write_text("# Проект\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("секрет\n", encoding="utf-8")
    (tmp_path / "проект-секреты").mkdir()
    (tmp_path / "проект-секреты" / "ключи.txt").write_text("ключ\n", encoding="utf-8")
    return root


def test_file_is_read_in_chunks(project):
    parts = commands.read_file(project, "src/main.py")
    assert len(parts) == 1 and parts[0].name == "main.py"
    assert base64.b64decode(parts[0].data).decode() == "print('привет')\n"
    assert parts[0].total == 1 and parts[0].mime.startswith("text/")


def test_big_file_is_refused(project, monkeypatch):
    monkeypatch.setattr(commands, "MAX_FILE", 5)
    with pytest.raises(CommandError) as error:
        commands.read_file(project, "src/main.py")
    assert error.value.code == "forbidden"


def test_missing_file(project):
    with pytest.raises(CommandError) as error:
        commands.read_file(project, "нет-такого.txt")
    assert error.value.code == "not_found"


@pytest.mark.parametrize("path", ["../проект-секреты/ключи.txt", "/etc/passwd", "src/../../проект-секреты"])
def test_outside_project_is_forbidden(project, path):
    """Сосед с похожим именем — та самая ловушка, из-за которой сравниваем не по префиксу строки."""
    with pytest.raises(CommandError) as error:
        commands.inside(project, path)
    assert error.value.code == "forbidden"


def test_symlink_out_is_forbidden(project, tmp_path):
    (project / "наружу").symlink_to(tmp_path / "проект-секреты")
    with pytest.raises(CommandError):
        commands.inside(project, "наружу/ключи.txt")


def test_tree_skips_service_dirs(project):
    text = commands.tree(project, depth=3)
    assert "README.md" in text and "src/" in text and "main.py" in text
    assert ".git" not in text, "служебные каталоги в дерево не идут"


async def test_git_on_plain_folder(project):
    """Не репозиторий — не падаем, отдаём то, что сказал git."""
    text = await commands.git(project, "status")
    assert text


async def test_unknown_git_mode(project):
    with pytest.raises(CommandError):
        await commands.git(project, "push")


def test_logs_tail(tmp_path):
    log = tmp_path / "agent.log"
    log.write_text("\n".join(f"строка {i}" for i in range(200)), encoding="utf-8")
    text = commands.logs(log, lines=10)
    assert text.splitlines()[-1] == "строка 199" and len(text.splitlines()) == 10
    assert commands.logs(tmp_path / "нет.log") == "лог пуст"
