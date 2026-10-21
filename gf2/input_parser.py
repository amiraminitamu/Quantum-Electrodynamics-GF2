# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"GF2 input-file parser."

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import psi4

from .options import GF2Options
from .qed import CavityConfiguration, PhotonMode


@dataclass
class JobConfig:
    """Metadata from the ``[job]`` block."""

    name: str
    output_file: str | None = None
    method: str = "gf2"
    cavity: CavityConfiguration | None = None


_SECTION_HEADER = re.compile(
    r"^\s*\[(?P<name>[A-Za-z_][A-Za-z_0-9]*)\]\s*$",
    re.MULTILINE,
)
_LEGACY_BLOCK = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z_0-9]*)\s*\{(?P<body>[^{}]*)\}",
    re.MULTILINE,
)
_ATOM_LINE = re.compile(
    r"^\s*(?P<sym>[A-Za-z]{1,3})\s+"
    r"(?P<x>[-+0-9.eE]+)\s+(?P<y>[-+0-9.eE]+)\s+(?P<z>[-+0-9.eE]+)\s*$"
)
_KV_LINE = re.compile(r"^\s*(?P<key>[A-Za-z_][A-Za-z_0-9]*)\s*(?:=\s*)?(?P<val>.+?)\s*$")


def _strip_comments(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].split("//", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def _parse_scalar(token: str) -> Any:
    t = token.strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        return t[1:-1]
    low = t.lower()
    if low in {"none", "null", "nil"}:
        return None
    if low in {"true", "yes", "on", "t", "1"}:
        return True
    if low in {"false", "no", "off", "f", "0"}:
        return False
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t


def _parse_kv_lines(body: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.endswith(":"):
            continue
        m = _KV_LINE.match(line)
        if not m:
            raise ValueError(f"Could not parse line: {raw!r}")
        key = m.group("key").lower()
        result[key] = _parse_scalar(m.group("val"))
    return result


def _uses_section_layout(text: str) -> bool:
    return bool(_SECTION_HEADER.search(text))


def _parse_sections(text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current: str | None = None
    body: list[str] = []
    for raw in text.splitlines():
        m = _SECTION_HEADER.match(raw)
        if m:
            if current is not None:
                sections[current] = "\n".join(body)
            current = m.group("name").lower()
            body = []
        elif current is not None:
            body.append(raw)
    if current is not None:
        sections[current] = "\n".join(body)
    return sections


def _parse_legacy_blocks(text: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    for match in _LEGACY_BLOCK.finditer(text):
        blocks[match.group("name").strip().lower()] = match.group("body")
    return blocks


def _looks_like_int(token: str) -> bool:
    try:
        int(token)
        return True
    except ValueError:
        return False


def _build_molecule_from_body(body: str) -> "psi4.core.Molecule":
    geometry_lines: list[str] = []
    charge_mult: tuple[int, int] | None = None
    passthrough: list[str] = []

    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.endswith(":"):
            continue

        atom = _ATOM_LINE.match(line)
        if atom:
            geometry_lines.append(f"{atom['sym']} {atom['x']} {atom['y']} {atom['z']}")
            continue

        tokens = line.split()
        if len(tokens) == 2 and all(_looks_like_int(t) for t in tokens):
            charge_mult = (int(tokens[0]), int(tokens[1]))
            continue

        if _KV_LINE.match(line):
            continue

        passthrough.append(line)

    kv = _parse_kv_lines(body)
    if "charge" in kv or "multiplicity" in kv:
        charge_mult = (
            int(kv.get("charge", 0)),
            int(kv.get("multiplicity", 1)),
        )

    if not geometry_lines:
        raise ValueError("Molecule section must list at least one atom (symbol x y z).")
    if charge_mult is None:
        charge_mult = (0, 1)

    psi4_geom = f"{charge_mult[0]} {charge_mult[1]}\n" + "\n".join(geometry_lines)
    extras: list[str] = []
    for key in ("symmetry", "units", "no_reorient", "no_com"):
        if key in kv:
            extras.append(f"{key} {kv[key]}")
    extras.extend(passthrough)
    if extras:
        psi4_geom += "\n" + "\n".join(dict.fromkeys(extras))
    return psi4.geometry(psi4_geom)


def _parse_job(blocks: dict[str, str]) -> JobConfig:
    if "job" not in blocks:
        return JobConfig(name="untitled")
    kv = _parse_kv_lines(blocks["job"])
    name = str(kv.get("name", kv.get("job_name", "untitled")))
    output = kv.get("output", kv.get("output_file"))
    if output is not None:
        output = str(output).strip()
        if not output:
            output = None
    return JobConfig(name=name, output_file=output)


def _parse_cavity(body: str) -> CavityConfiguration:
    """Parse ``modeN = omega, coupling, px, py, pz`` entries."""
    values = _parse_kv_lines(body)
    units = str(values.pop("units", "atomic")).strip().lower()
    if units not in {"atomic", "au", "a.u."}:
        raise ValueError("Cavity mode frequency and coupling units must be atomic units.")
    modes: list[PhotonMode] = []
    for key, raw in values.items():
        if not key.startswith("mode"):
            raise ValueError(f"Unknown cavity setting {key!r}; expected modeN or units.")
        fields = str(raw).replace(",", " ").split()
        if len(fields) != 5:
            raise ValueError(f"{key} must contain: frequency, coupling, px, py, pz.")
        try:
            omega, coupling, px, py, pz = (float(value) for value in fields)
        except ValueError as exc:
            raise ValueError(f"{key} contains a non-numeric cavity value.") from exc
        modes.append(PhotonMode(omega, coupling, (px, py, pz)))
    return CavityConfiguration(modes)


def parse_input_text(
    text: str,
    *,
    input_path: str | Path | None = None,
) -> tuple[JobConfig, psi4.core.Molecule, dict[str, Any], GF2Options]:
    """Parse input text and return job metadata plus calculation settings."""
    text = _strip_comments(text)
    if _uses_section_layout(text):
        blocks = _parse_sections(text)
    else:
        blocks = _parse_legacy_blocks(text)

    unknown = set(blocks) - {"job", "molecule", "psi4", "gf2", "cavity"}
    if unknown:
        raise ValueError("Unsupported input section(s): " + ", ".join(sorted(unknown)))

    if "molecule" not in blocks:
        raise ValueError("Input must define a molecule ([molecule] or molecule { ... }).")
    if "gf2" not in blocks:
        raise ValueError("Input must define GF2 settings ([gf2] or gf2 { ... }).")

    job = _parse_job(blocks)
    mol = _build_molecule_from_body(blocks["molecule"])

    psi4_opts: dict[str, Any] = {}
    if "psi4" in blocks:
        psi4_opts = _parse_kv_lines(blocks["psi4"])

    gf2_values = _parse_kv_lines(blocks["gf2"])
    method = str(gf2_values.pop("method", "gf2")).lower().replace("_", "-")
    if method not in {"gf2", "electronic", "cs-gf2", "lf-gf2"}:
        raise ValueError("method must be 'gf2'/'electronic', 'cs-gf2', or 'lf-gf2'.")
    cavity = _parse_cavity(blocks["cavity"]) if "cavity" in blocks else None
    coupled = method in {"cs-gf2", "lf-gf2"}
    if coupled != (cavity is not None):
        raise ValueError("A [cavity] section is required exactly for method=cs-gf2 or lf-gf2.")

    job = replace(
        job,
        method=method,
        cavity=cavity,
    )
    gf2_opts = GF2Options.from_mapping(gf2_values)
    return job, mol, psi4_opts, gf2_opts


def parse_input_file(
    path: str | Path,
) -> tuple[JobConfig, psi4.core.Molecule, dict[str, Any], GF2Options]:
    """Read and parse a ``.inp`` file."""
    source = Path(path).expanduser().resolve()
    with open(source, "r", encoding="utf-8") as fh:
        return parse_input_text(fh.read(), input_path=source)
