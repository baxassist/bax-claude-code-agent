"""Запуск движка и регистрация агентов из терминала.

    bax-agent run                                   # обслуживать зарегистрированных агентов
    bax-agent add "<строка>" --path ~/projects/api  # строка берётся в приложении
    bax-agent list
    bax-agent remove <id агента>

Агента заводят в приложении Бакса — там он получает имя и строку регистрации. Каталог
проекта задаётся только здесь, на самом компьютере: приложение добавить его не может.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
from pathlib import Path

from . import __version__
from .config import (
    PERMISSION_MODES,
    Config,
    ConfigError,
    add_agent,
    install_id,
    load,
    read_agents,
    remove_agent,
)
from .engine import Engine


def setup_logging(config: Config) -> None:
    """Лог и в терминал, и в файл: файл потом отдаёт команда «logs» из приложения."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(config.log_file, encoding="utf-8"))
    except OSError as error:
        print(f"Лог в файл не пишется ({error}) — продолжаю без него", file=sys.stderr)
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def command_add(config: Config, args) -> int:
    registration = add_agent(
        config, args.registration, args.path,
        permission_mode=args.permission_mode, model=args.model, effort=args.effort,
        restricted=args.restricted,
    )
    print(f"Агент {registration.agent_id} зарегистрирован: {registration.path}")
    print("Перезапустите движок (`bax-agent run`), чтобы он вышел на связь.")
    return 0


def command_list(config: Config, _args) -> int:
    agents = read_agents(config)
    if not agents:
        print("Пока никого. Заведите агента в приложении и выполните «bax-agent add <строка>».")
        return 0
    print(f"Установка {config.name} ({install_id(config)})")
    for item in agents:
        extra = [item.permission_mode]
        if item.model:
            extra.append(item.model)
        if item.restricted:
            extra.append("restricted")
        print(f"  {item.agent_id}  {item.path}  [{', '.join(extra)}]")
    return 0


def command_remove(config: Config, args) -> int:
    if remove_agent(config, args.agent_id):
        print(f"Агент {args.agent_id} больше не обслуживается на этой машине.")
        print("В приложении он останется: удалить его можно там.")
        return 0
    print(f"Агент {args.agent_id} и так не зарегистрирован", file=sys.stderr)
    return 1


def command_run(config: Config, args) -> int:
    setup_logging(config)
    engine = Engine(config, claude_path=args.claude_path)
    print(f"«{config.name}»: агентов {len(config.agents)}, сервер {config.server}")
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(engine.run())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="bax-agent", description="Движок Claude Code для Бакса")
    parser.add_argument("--config", default="config.toml", help="файл настроек (по умолчанию config.toml)")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command")

    run = commands.add_parser("run", help="обслуживать зарегистрированных агентов")
    run.add_argument("--claude-path", default="",
                     help="свой Claude Code вместо того, что идёт внутри SDK")
    run.set_defaults(func=command_run)

    add = commands.add_parser("add", help="зарегистрировать агента строкой из приложения")
    add.add_argument("registration", help="<id агента>:<id ключа>:<секрет>")
    add.add_argument("--path", required=True, help="каталог проекта")
    add.add_argument("--permission-mode", default="acceptEdits", choices=PERMISSION_MODES)
    add.add_argument("--model", default="", help="модель по умолчанию")
    add.add_argument("--effort", default="", help="уровень усилий по умолчанию")
    add.add_argument("--restricted", action="store_true",
                     help="без запуска команд, файлы только внутри проекта")
    add.set_defaults(func=command_add)

    listing = commands.add_parser("list", help="что зарегистрировано на этой машине")
    listing.set_defaults(func=command_list)

    remove = commands.add_parser("remove", help="перестать обслуживать агента здесь")
    remove.add_argument("agent_id")
    remove.set_defaults(func=command_remove)

    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args([*sys.argv[1:], "run"]) if Path(args.config).exists() else args
        if args.command is None:
            parser.print_help()
            return 2

    try:
        config = load(args.config)
    except ConfigError as error:
        print(f"Настройки: {error}", file=sys.stderr)
        return 2

    try:
        return args.func(config, args)
    except ConfigError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
