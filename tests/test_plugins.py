import json
from pathlib import Path

import pytest

from benchmark_ladder.plugins import (
    AdapterFactoryError,
    load_adapter,
    load_adapter_config,
    load_adapter_factory,
)


def test_load_adapter_factory_requires_module_colon_attribute() -> None:
    with pytest.raises(ValueError, match="module:factory"):
        load_adapter_factory("tests.fixture_adapter_plugin")


def test_load_adapter_factory_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="not callable"):
        load_adapter_factory("tests.fixture_adapter_plugin:NOT_CALLABLE")


def test_load_adapter_config_requires_object(tmp_path: Path) -> None:
    config_path = tmp_path / "adapter.json"
    config_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(TypeError, match="JSON object"):
        load_adapter_config(config_path)


def test_load_adapter_instantiates_external_factory(tmp_path: Path) -> None:
    config_path = tmp_path / "adapter.json"
    config_path.write_text(json.dumps({"scores": {"a": -1.0, "b": -2.0}}), encoding="utf-8")

    adapter = load_adapter("tests.fixture_adapter_plugin:create_adapter", config_path)

    assert adapter.continuation_logprob(b"ctx", b"a") == -1.0
    assert adapter.continuation_logprob(b"ctx", b"b") == -2.0


def test_load_adapter_rejects_factory_result_missing_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "adapter.json"
    config_path.write_text("{}", encoding="utf-8")

    with pytest.raises(TypeError, match="missing required adapter methods"):
        load_adapter("tests.fixture_adapter_plugin:create_invalid_adapter", config_path)


def test_load_adapter_redacts_external_factory_exception_text(tmp_path: Path) -> None:
    config_path = tmp_path / "adapter.json"
    config_path.write_text(json.dumps({"secret": "do-not-print"}), encoding="utf-8")

    with pytest.raises(AdapterFactoryError) as exc_info:
        load_adapter("tests.fixture_adapter_plugin:create_failing_adapter", config_path)

    assert "ValueError" in str(exc_info.value)
    assert "do-not-print" not in str(exc_info.value)
