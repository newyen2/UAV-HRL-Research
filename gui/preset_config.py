"""Sparse, reusable JSON presets for GUI experiment settings."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from runtime import configuration as core


PRESET_DIRECTORY = Path(__file__).resolve().parent.parent / "Outputs" / "Presets"
_COMMON_GROUPS = frozenset(core._COMMON_SETTINGS)
_KNOWN_GROUPS = _COMMON_GROUPS | frozenset(core._METHOD_SETTING_GROUPS.values())
_DEFAULT_PRESET_NAME = "設定範本"


def _validate_name(name: Any, *, optional: bool = False) -> str:
    if name is None and optional:
        return _DEFAULT_PRESET_NAME
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    name = name.strip()
    if len(name) > 120 or any(ord(character) < 32 for character in name):
        raise ValueError("name must be at most 120 characters and contain no control characters")
    return name


def _validate_method(method: Any, *, field: str = "method") -> str:
    if not isinstance(method, str) or method not in core.METHODS:
        raise ValueError(f"Unknown {field}: {method!r}")
    return method


def _validate_envelope(preset: Mapping) -> tuple[str, str | None, Mapping]:
    if not isinstance(preset, Mapping):
        raise ValueError("preset must be an object")
    extra = set(preset) - {"name", "method", "settings"}
    if extra:
        raise ValueError(f"Unknown preset fields: {sorted(extra)}")
    if "settings" not in preset:
        raise ValueError("settings must be an object")
    name = _validate_name(preset.get("name"), optional="name" not in preset)
    method = None
    if "method" in preset:
        method = _validate_method(preset["method"])
    supplied = preset["settings"]
    if not isinstance(supplied, Mapping):
        raise ValueError("settings must be an object")
    return name, method, supplied


def _validate_group_envelope(supplied: Mapping) -> bool:
    """Check group containers and return whether method-specific groups occur."""
    method_specific = False
    for group, fields in supplied.items():
        if not isinstance(group, str) or group not in _KNOWN_GROUPS:
            if group == "RECORD" and isinstance(fields, Mapping) and "PATH" in fields:
                raise ValueError("RECORD.PATH is restricted and assigned by the worker")
            raise ValueError(f"Unknown or restricted settings group: {group!r}")
        if not isinstance(fields, Mapping):
            raise ValueError(f"settings.{group} must be an object")
        if group == "RECORD" and "PATH" in fields:
            raise ValueError("RECORD.PATH is restricted and assigned by the worker")
        if group not in _COMMON_GROUPS:
            method_specific = True
    return method_specific


def build_preset(name: str, method: str,
                 values: Mapping[tuple[str, str], Any]) -> dict:
    """Build a validated sparse preset from selected form fields only."""
    name = _validate_name(name)
    method = _validate_method(method)
    if not isinstance(values, Mapping):
        raise ValueError("values must be a mapping of (group, field) paths")

    supplied: dict[str, dict] = {}
    for path, value in values.items():
        if (not isinstance(path, tuple) or len(path) != 2
                or not all(isinstance(part, str) for part in path)):
            raise ValueError("setting paths must be (group, field) string pairs")
        group, field = path
        supplied.setdefault(group, {})[field] = value

    method_specific = _validate_group_envelope(supplied)
    normalized = core.validate_settings(method, supplied)

    sparse_settings = {
        group: {field: normalized[group][field] for field in fields}
        for group, fields in supplied.items()
    }
    preset = {"name": name, "settings": sparse_settings}
    if method_specific:
        preset["method"] = method
    return preset


def resolve_preset(preset: Mapping, current_method: str) -> dict:
    """Apply a sparse preset over fresh defaults and return a detached spec."""
    current_method = _validate_method(current_method, field="current method")
    name, preset_method, supplied = _validate_envelope(preset)
    method_specific = _validate_group_envelope(supplied)

    if method_specific and preset_method is None:
        raise ValueError("method is required for method-specific settings")
    if preset_method is not None:
        method = preset_method
    else:
        method = current_method

    # Validate against the selected target's defaults and method-dependent bounds.
    normalized = core.validate_settings(method, supplied)
    return {
        "name": name,
        "method": method,
        "settings": normalized,
    }
