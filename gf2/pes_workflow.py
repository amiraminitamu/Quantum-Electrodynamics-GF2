# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Utilities for reproducible diatomic QED-GF2 potential-energy scans."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


_SECTION_RE = re.compile(r"^\s*\[([^]]+)\]\s*$")


def distance_tag(distance: float) -> str:
    """Return a filesystem-safe fixed-width tag for a bond distance."""
    value = float(distance)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("distance must be finite and positive")
    return f"{value:.5f}".replace("-", "m").replace(".", "p")


def _section_bounds(lines: list[str], section: str) -> tuple[int, int]:
    target = section.strip().lower()
    start = None
    for index, line in enumerate(lines):
        match = _SECTION_RE.match(line)
        if match and match.group(1).strip().lower() == target:
            start = index + 1
            break
    if start is None:
        raise ValueError(f"input is missing [{section}] section")
    end = len(lines)
    for index in range(start, len(lines)):
        if _SECTION_RE.match(lines[index]):
            end = index
            break
    return start, end


def _set_key(lines: list[str], section: str, key: str, value: str) -> None:
    start, end = _section_bounds(lines, section)
    key_lower = key.lower()
    for index in range(start, end):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        lhs = stripped.split("=", 1)[0].strip().lower()
        if lhs == key_lower:
            indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
            lines[index] = f"{indent}{key} = {value}"
            return
    lines.insert(end, f"{key} = {value}")


def _coord_start(tokens: list[str]) -> int:
    if len(tokens) >= 5:
        try:
            float(tokens[1])
        except ValueError:
            return 1
        return 2
    return 1


def _render_classic_diatomic(
    template_text: str,
    distance: float,
    *,
    method: str | None,
    job_name: str | None,
) -> str:
    from .classic_input import classic_layout, replace_method_line

    value = float(distance)
    layout = classic_layout(template_text)
    if layout is None:
        raise ValueError("input is not a route or namelist deck")
    lines = template_text.splitlines()
    first_index, second_index = layout.atom_indexes
    first_tokens = lines[first_index].split()
    second_tokens = lines[second_index].split()
    first_xyz = _coord_start(first_tokens)
    second_xyz = _coord_start(second_tokens)
    r1 = np.asarray([float(token) for token in first_tokens[first_xyz : first_xyz + 3]])
    r2 = np.asarray([float(token) for token in second_tokens[second_xyz : second_xyz + 3]])
    axis = r2 - r1
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-14:
        raise ValueError("template diatomic bond length must be nonzero")
    r2_new = r1 + value * axis / norm
    indent = lines[second_index][: len(lines[second_index]) - len(lines[second_index].lstrip())]
    head = second_tokens[:second_xyz]
    tail = second_tokens[second_xyz + 3 :]
    coords = f"{r2_new[0]:.12f} {r2_new[1]:.12f} {r2_new[2]:.12f}"
    trailing = f" {' '.join(tail)}" if tail else ""
    lines[second_index] = f"{indent}{' '.join(head)} {coords}{trailing}"
    if job_name is not None and layout.title_index is not None:
        lines[layout.title_index] = job_name
    if method is not None:
        for index in layout.method_indexes:
            updated = replace_method_line(lines[index], method)
            if updated != lines[index]:
                lines[index] = updated
                break
    return "\n".join(lines) + "\n"


def render_diatomic_input(
    template_text: str,
    distance: float,
    *,
    method: str | None = None,
    job_name: str | None = None,
) -> str:
    "Render one geometry from a two-atom input template."
    from .classic_input import deck_kind

    value = float(distance)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("distance must be finite and positive")
    if deck_kind(template_text) in {"route", "namelist"}:
        return _render_classic_diatomic(
            template_text,
            value,
            method=method,
            job_name=job_name,
        )

    lines = template_text.splitlines()
    start, end = _section_bounds(lines, "molecule")
    atom_rows: list[tuple[int, list[str]]] = []
    for index in range(start, end):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#") or "=" in stripped:
            continue
        tokens = stripped.split()
        if len(tokens) < 4:
            continue
        try:
            [float(token) for token in tokens[1:4]]
        except ValueError:
            continue
        atom_rows.append((index, tokens))

    if len(atom_rows) != 2:
        raise ValueError(
            "diatomic PES workflow requires exactly two Cartesian atom rows in [molecule]"
        )

    _, first_tokens = atom_rows[0]
    second_index, second_tokens = atom_rows[1]
    r1 = np.asarray([float(token) for token in first_tokens[1:4]], dtype=float)
    r2 = np.asarray([float(token) for token in second_tokens[1:4]], dtype=float)
    axis = r2 - r1
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-14:
        raise ValueError("template diatomic bond length must be nonzero")
    r2_new = r1 + value * axis / norm
    indent = lines[second_index][: len(lines[second_index]) - len(lines[second_index].lstrip())]
    trailing = "" if len(second_tokens) == 4 else " " + " ".join(second_tokens[4:])
    lines[second_index] = (
        f"{indent}{second_tokens[0]} {r2_new[0]:.12f} {r2_new[1]:.12f} {r2_new[2]:.12f}{trailing}"
    )

    if method is not None:
        _set_key(lines, "gf2", "method", method)
    if job_name is not None:
        _set_key(lines, "job", "name", job_name)

    return "\n".join(lines) + "\n"


def write_diatomic_input(
    template: str | Path,
    destination: str | Path,
    distance: float,
    *,
    method: str | None = None,
    job_name: str | None = None,
) -> Path:
    source = Path(template).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_diatomic_input(
            source.read_text(encoding="utf-8"),
            distance,
            method=method,
            job_name=job_name,
        ),
        encoding="utf-8",
    )
    return target
