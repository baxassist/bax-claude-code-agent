"""Настройки движка и регистрация агентов."""

from __future__ import annotations

import json

import pytest

from bax_agent.config import ConfigError, add_agent, install_id, load, read_agents, remove_agent, split_key

REGISTRATION = "6f0e6a1c-0000-4000-8000-000000000001:9a1b0000-0000-4000-8000-000000000002:секретная-строка"


def write_config(tmp_path, server: str = "ws://localhost:8010/agent"):
    file = tmp_path / "config.toml"
    file.write_text(f'[engine]\nname = "Тест"\nserver = "{server}"\n', encoding="utf-8")
    return file


def test_engine_config_without_agents(tmp_path):
    """Агентов в конфиге нет: они регистрируются отдельно, командой add."""
    config = load(write_config(tmp_path))
    assert config.name == "Тест" and config.max_parallel == 2
    assert config.agents == [], "пока никого не зарегистрировали"


def test_old_section_name_still_works(tmp_path):
    """Секция называется [engine], но прежнее имя [agent] понимаем — чтобы не ломать конфиги."""
    file = tmp_path / "config.toml"
    file.write_text('[agent]\nserver = "ws://localhost:8010/agent"\n', encoding="utf-8")
    assert load(file).server.endswith("/agent")


def test_add_list_remove(tmp_path):
    folder = tmp_path / "api"
    folder.mkdir()
    config = load(write_config(tmp_path))

    registration = add_agent(config, REGISTRATION, folder, permission_mode="manual", restricted=True)
    assert registration.agent_id.startswith("6f0e6a1c")
    assert registration.path == folder.resolve() and registration.restricted

    saved = read_agents(config)
    assert [item.agent_id for item in saved] == [registration.agent_id]
    assert oct(config.agents_file.stat().st_mode)[-3:] == "600", "в файле лежат секреты"
    assert json.loads(config.agents_file.read_text(encoding="utf-8"))[0]["permission_mode"] == "manual"

    assert remove_agent(config, registration.agent_id) is True
    assert read_agents(config) == []
    assert remove_agent(config, registration.agent_id) is False


def test_add_replaces_the_same_agent(tmp_path):
    """Перевыпустили строку — регистрируем поверх, а не вторым экземпляром."""
    first, second = tmp_path / "api", tmp_path / "api2"
    first.mkdir()
    second.mkdir()
    config = load(write_config(tmp_path))

    add_agent(config, REGISTRATION, first)
    add_agent(config, REGISTRATION.replace("9a1b", "0000"), second)
    saved = read_agents(config)
    assert len(saved) == 1 and saved[0].path == second.resolve()


def test_missing_directory_and_bad_mode(tmp_path):
    config = load(write_config(tmp_path))
    with pytest.raises(ConfigError, match="Каталога"):
        add_agent(config, REGISTRATION, tmp_path / "нет-такого")

    folder = tmp_path / "api"
    folder.mkdir()
    with pytest.raises(ConfigError, match="Режим разрешений"):
        add_agent(config, REGISTRATION, folder, permission_mode="ask")


def test_registration_string_must_have_three_parts():
    assert split_key(REGISTRATION)[0].startswith("6f0e6a1c")
    with pytest.raises(ConfigError, match="id агента"):
        split_key("просто-строка")
    with pytest.raises(ConfigError):
        split_key("один:два")


def test_agent_with_lost_directory_is_skipped(tmp_path, caplog):
    """Каталог унесли — агент пропускается, движок продолжает с остальными."""
    folder = tmp_path / "api"
    folder.mkdir()
    config = load(write_config(tmp_path))
    add_agent(config, REGISTRATION, folder)
    folder.rmdir()

    assert read_agents(config) == []


def test_install_id_is_made_once(tmp_path):
    """id установки делается при первом запуске и дальше не меняется: по нему сервер
    закрепляет ключ за одной машиной."""
    config = load(write_config(tmp_path))
    first = install_id(config)
    assert len(first) == 36 and install_id(config) == first
    assert oct(config.install_file.stat().st_mode)[-3:] == "600"


def test_plain_ws_to_the_internet_is_refused(tmp_path):
    """Открытый ws:// наружу — это задачи и код открытым текстом. В своей сети можно."""
    with pytest.raises(ConfigError, match="wss"):
        load(write_config(tmp_path, "ws://bax.example.com/agent"))
    assert load(write_config(tmp_path, "ws://192.168.1.75:8010/agent")).server
    assert load(write_config(tmp_path, "wss://bax.example.com/agent")).server
