"""Runtime loading for externally installed model adapters."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

from benchmark_ladder.adapters import ModelAdapter


class AdapterFactoryError(RuntimeError):
    """Redacted failure raised when external adapter construction fails."""


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
        raise TypeError("adapter target is not callable")
    return cast(AdapterFactory, value)


def load_adapter_config(path: Path | None) -> Mapping[str, object]:
    """Load an optional adapter configuration object without logging its contents."""

    if path is None:
        return {}
    try:
        parsed: object = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid adapter config JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise TypeError("adapter config must be a JSON object")
    if any(not isinstance(key, str) for key in parsed):
        raise TypeError("adapter config keys must be strings")
    return cast(Mapping[str, object], parsed)


def load_adapter(spec: str, config_path: Path | None = None) -> ModelAdapter:
    """Instantiate an external adapter in the current trusted process.

    External factory exception text is deliberately not propagated: configuration values or
    model-local data can appear in third-party exception messages. The original exception remains
    available as the chained cause to a trusted caller that explicitly handles it privately.
    """

    factory = load_adapter_factory(spec)
    config = load_adapter_config(config_path)
    try:
        adapter = factory(config)
    except Exception as exc:
        raise AdapterFactoryError(
            f"adapter factory failed ({type(exc).__name__})"
        ) from exc
    if not isinstance(adapter, ModelAdapter):
        raise TypeError("adapter factory returned an object missing required adapter methods")
    return adapter
