"""Runtime loading for externally installed model adapters."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

from benchmark_ladder.adapters import ModelAdapter


class AdapterFactory(Protocol):
    def __call__(self, config: Mapping[str, object]) -> ModelAdapter:
        """Build an adapter from caller-supplied configuration."""
        ...


def load_adapter_factory(spec: str) -> AdapterFactory:
    """Load ``module:attribute`` and validate that the target is callable."""

    module_name, separator, attribute_name = spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("adapter spec must have the form module:factory")

    module = importlib.import_module(module_name)
    value: object = getattr(module, attribute_name)
    if not callable(value):
        raise TypeError(f"adapter target {spec!r} is not callable")
    return cast(AdapterFactory, value)


def load_adapter_config(path: Path | None) -> Mapping[str, object]:
    """Load an optional adapter configuration object without logging its contents."""

    if path is None:
        return {}
    try:
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid adapter config JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"{path}: adapter config must be a JSON object")
    if any(not isinstance(key, str) for key in parsed):
        raise TypeError(f"{path}: adapter config keys must be strings")
    return cast(Mapping[str, object], parsed)


def load_adapter(spec: str, config_path: Path | None = None) -> ModelAdapter:
    """Instantiate an external adapter in the current trusted process."""

    factory = load_adapter_factory(spec)
    config = load_adapter_config(config_path)
    return factory(config)
