"""Конфигурация движка и регистрации агентов.

Два файла рядом друг с другом:

- `config.toml` — сам движок: адрес сервера, пределы, имя установки. Агентов в нём нет,
  поэтому секция называется `[engine]`;
- `agents.json` — что зарегистрировано командой `bax-agent add`: id агента, каталог,
  режим разрешений и ключ. Права 600: в нём секреты.

Плюс `.install` — случайный uuid4 этой установки, сделанный при первом запуске. Сервер
закрепляет ключ за первой установкой, которая им подключилась, поэтому скопированная
строка регистрации на второй машине получит отказ.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import stat
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger("bax.config")

# Ровно те значения, что понимает сам CLI (`claude --help`): своих слов не придумываем,
# иначе процесс не запустится. «Спрашивать про всё» — это `manual`. По умолчанию
# `acceptEdits`: при `auto` решает встроенный классификатор и вопросы до телефона
# не доходят вовсе.
PERMISSION_MODES = ("plan", "acceptEdits", "auto", "manual", "dontAsk", "bypassPermissions")

# Адреса, где открытый ws:// допустим: это разработка на своей же машине или в домашней сети
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
LOCAL_PREFIXES = ("192.168.", "10.", "172.16.", "172.17.", "172.18.", "172.19.",
                  "172.2", "172.30.", "172.31.")


class ConfigError(Exception):
    """Понятная человеку ошибка в настройках: печатается, и движок не стартует."""


@dataclass(frozen=True)
class Registration:
    """Один зарегистрированный агент: что о нём знает движок."""

    #: id агента — его выдал сервер при создании в приложении
    agent_id: str
    #: открытая часть ключа и секрет: ими подписывается рукопожатие
    key_id: str
    secret: str
    #: каталог проекта; за него движок не выходит
    path: Path
    permission_mode: str = "acceptEdits"
    model: str = ""
    effort: str = ""
    #: без запуска команд, файловые операции только внутри проекта
    restricted: bool = False

    def as_json(self) -> dict:
        return {
            "agent_id": self.agent_id, "key_id": self.key_id, "secret": self.secret,
            "path": str(self.path), "permission_mode": self.permission_mode,
            "model": self.model, "effort": self.effort, "restricted": self.restricted,
        }


@dataclass(frozen=True)
class Config:
    name: str
    server: str
    log_level: str = "info"
    max_parallel: int = 2
    task_timeout_min: int = 60
    question_timeout_min: int = 10
    #: Свой Claude Code вместо того, что идёт внутри SDK. Пусто — берётся CLI из пакета:
    #: отдельная установка Claude Code на машине не нужна
    claude_path: str = ""
    path: Path | None = None  # откуда прочитан — рядом лежат agents.json и .install
    agents: list[Registration] = field(default_factory=list)

    @property
    def folder(self) -> Path:
        return self.path.parent if self.path else Path.cwd()

    @property
    def agents_file(self) -> Path:
        return self.folder / "agents.json"

    @property
    def install_file(self) -> Path:
        return self.folder / ".install"

    @property
    def log_file(self) -> Path:
        """Журнал движка: его же отдаёт команда `logs` из приложения."""
        return self.folder / "agent.log"

    @property
    def events_file(self) -> Path:
        """Свои события движка — остановки, перезапуски, смена модели."""
        return self.folder / "events.jsonl"


def split_key(registration: str) -> tuple[str, str, str]:
    """`<id агента>:<id ключа>:<секрет>` — так строка приходит из приложения."""
    parts = registration.strip().split(":")
    if len(parts) != 3 or not all(parts):
        raise ConfigError(
            "Строка регистрации должна выглядеть как «<id агента>:<id ключа>:<секрет>» — "
            "скопируйте её целиком из приложения"
        )
    return parts[0], parts[1], parts[2]


def _check_tls(server: str) -> None:
    """Открытый `ws://` наружу — это разговор с движком открытым текстом: и задачи, и код,
    и ответы модели. Внутри своей сети это допустимо для разработки, в интернет — нет.
    TLS здесь единственное, что защищает канал от подмены на пути (рукопожатие HMAC
    доказывает только, кто такой движок)."""
    parts = urlsplit(server)
    if parts.scheme == "wss":
        return
    if parts.scheme != "ws":
        raise ConfigError(f"[engine]: адрес сервера должен начинаться с wss:// — получено «{server}»")
    host = (parts.hostname or "").lower()
    local = host in LOCAL_HOSTS or any(host.startswith(prefix) for prefix in LOCAL_PREFIXES) \
        or host.endswith(".local")
    if not local:
        raise ConfigError(
            f"[engine]: «{server}» — открытый ws:// на внешний адрес. Так задачи и код уйдут "
            "незашифрованными; для сервера в интернете нужен wss://"
        )


def load(path: str | Path) -> Config:
    """Настройки движка плюс зарегистрированные агенты."""
    path = Path(path).expanduser()
    if not path.exists():
        raise ConfigError(f"Нет файла настроек {path}. Возьмите config.example.toml за образец.")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: не разобрался — {error}") from error

    engine = data.get("engine") or data.get("agent") or {}
    server = str(engine.get("server") or "").strip()
    if not server:
        raise ConfigError("[engine]: не хватает «server»")
    _check_tls(server)

    config = Config(
        name=str(engine.get("name") or socket.gethostname()),
        server=server,
        log_level=str(engine.get("log_level") or "info"),
        max_parallel=int(engine.get("max_parallel") or 2),
        task_timeout_min=int(engine.get("task_timeout_min") or 60),
        question_timeout_min=int(engine.get("question_timeout_min") or 10),
        claude_path=str(engine.get("claude_path") or "").strip(),
        path=path.resolve(),
    )
    return Config(**{**config.__dict__, "agents": read_agents(config)})


def read_agents(config: Config) -> list[Registration]:
    """Зарегистрированные агенты. Файла нет — движку нечего обслуживать, и это не ошибка:
    агента добавляют командой `bax-agent add`."""
    file = config.agents_file
    if not file.exists():
        return []
    try:
        items = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"{file}: не разобрался — {error}") from error
    if not isinstance(items, list):
        raise ConfigError(f"{file}: ожидался список агентов")

    found: list[Registration] = []
    for item in items:
        folder = Path(str(item.get("path") or "")).expanduser()
        if not folder.is_dir():
            logger.warning("каталог %s пропал — агент %s пропущен", folder, item.get("agent_id"))
            continue
        mode = str(item.get("permission_mode") or "acceptEdits")
        if mode not in PERMISSION_MODES:
            raise ConfigError(f"агент {item.get('agent_id')}: режим разрешений «{mode}» — "
                              f"не из {PERMISSION_MODES}")
        found.append(Registration(
            agent_id=str(item.get("agent_id") or ""),
            key_id=str(item.get("key_id") or ""),
            secret=str(item.get("secret") or ""),
            path=folder.resolve(),
            permission_mode=mode,
            model=str(item.get("model") or ""),
            effort=str(item.get("effort") or ""),
            restricted=bool(item.get("restricted") or False),
        ))
    return found


def write_agents(config: Config, agents: list[Registration]) -> None:
    """Права 600: в файле лежат секреты ключей."""
    file = config.agents_file
    file.write_text(json.dumps([item.as_json() for item in agents], ensure_ascii=False, indent=2),
                    encoding="utf-8")
    file.chmod(stat.S_IRUSR | stat.S_IWUSR)


def install_id(config: Config) -> str:
    """Id этой установки. Сервер закрепляет ключ за первой установкой, которая им подключилась,
    поэтому скопированная строка регистрации на второй машине получит `key_claimed`.

    Файл потеряли — сделаем новый; сервер тогда попросит перевыпустить строку. Это честнее,
    чем молча пустить неизвестно кого.
    """
    file = config.install_file
    try:
        saved = file.read_text(encoding="utf-8").strip()
        if saved:
            return saved
    except OSError:
        pass
    fresh = str(uuid.uuid4())
    try:
        file.write_text(fresh + "\n", encoding="utf-8")
        file.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError as error:
        logger.warning("id установки не сохранился (%s) — в следующий раз будет новый", error)
    return fresh


def add_agent(config: Config, registration: str, path: str | Path, permission_mode: str = "acceptEdits",
              model: str = "", effort: str = "", restricted: bool = False) -> Registration:
    """`bax-agent add`: строка из приложения плюс каталог проекта.

    Каталог задаётся только здесь, на самом компьютере: приложение добавить новый не может,
    так ошибка или чужая команда не откроют доступ к лишнему.
    """
    agent_id, key_id, secret = split_key(registration)
    folder = Path(str(path)).expanduser()
    if not folder.is_dir():
        raise ConfigError(f"Каталога {folder} нет")
    if permission_mode not in PERMISSION_MODES:
        raise ConfigError(f"Режим разрешений «{permission_mode}» — не из {PERMISSION_MODES}")

    fresh = Registration(
        agent_id=agent_id, key_id=key_id, secret=secret, path=folder.resolve(),
        permission_mode=permission_mode, model=model, effort=effort, restricted=restricted,
    )
    agents = [item for item in read_agents(config) if item.agent_id != agent_id]
    agents.append(fresh)
    write_agents(config, agents)
    return fresh


def remove_agent(config: Config, agent_id: str) -> bool:
    agents = read_agents(config)
    left = [item for item in agents if item.agent_id != agent_id]
    if len(left) == len(agents):
        return False
    write_agents(config, left)
    return True


def read_secret_file(config: Config) -> str | None:
    """Переходная совместимость: ключ прежней схемы лежал в `.secret` рядом с конфигом."""
    file = config.folder / ".secret"
    if not file.exists():
        return None
    return file.read_text(encoding="utf-8").strip() or None


def hostname() -> str:
    return socket.gethostname() or os.uname().nodename
