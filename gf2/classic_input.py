# Authors: Amirhosein Amini, Jaime Cerda, Leopoldo Mejía, Arkajit Mandal.
# 2026. Licensed under the MIT License. See LICENSE.
"""Route-card and namelist decks for QED-GF2.

The older ``[section]`` and ``name { }`` decks stay valid. This module
turns a route card or a ``$`` namelist into those same sections so the
calculation path does not change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_SECTION_HEADER = re.compile(
    r"^\s*\[(?P<name>[A-Za-z_][A-Za-z_0-9]*)\]\s*$",
    re.MULTILINE,
)
_LEGACY_BLOCK = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z_0-9]*)\s*\{(?P<body>[^{}]*)\}",
    re.MULTILINE,
)
_NAMELIST_HINT = re.compile(r"(?im)^\s*\$(molecule|rem|contrl|control|data)\b")
_ROUTE_HINT = re.compile(r"(?im)^\s*#")
_LINK0_HINT = re.compile(r"(?im)^\s*%[A-Za-z]")
_ELEMENT = re.compile(r"^[A-Za-z]{1,2}\d*$")

_METHOD_ALIASES = {
    "gf2": ("gf2", None),
    "electronic": ("gf2", None),
    "ugf2": ("gf2", "unrestricted"),
    "u-gf2": ("gf2", "unrestricted"),
    "cs-gf2": ("cs-gf2", None),
    "csgf2": ("cs-gf2", None),
    "ucs-gf2": ("cs-gf2", "unrestricted"),
    "ucsgf2": ("cs-gf2", "unrestricted"),
    "lf-gf2": ("lf-gf2", None),
    "lfgf2": ("lf-gf2", None),
}
_PSI4_KEYS = {
    "basis",
    "scf_type",
    "memory",
    "reference",
    "guess",
    "puream",
    "e_convergence",
    "d_convergence",
    "maxiter",
    "ints_tolerance",
    "screening",
}
_MOLECULE_KEYS = {"symmetry", "units", "no_reorient", "no_com"}
_BARE_FLAGS = {
    "mp2": ("mp2", True),
    "nomp2": ("mp2", False),
    "uhf": ("spin_mode", "unrestricted"),
    "rhf": ("spin_mode", "restricted"),
    "unrestricted": ("spin_mode", "unrestricted"),
    "restricted": ("spin_mode", "restricted"),
}
_KEY_ALIAS = {
    "eri": "eri_type",
    "scf": "scf_type",
    "spin": "spin_mode",
    "se_fit": "se_fit_basis",
    "fock_fit": "fock_fit_basis",
    "nproc": "n_threads",
    "nprocshared": "n_threads",
    "nprocs": "n_threads",
    "mem": "memory",
    "icharg": "charge",
    "mult": "multiplicity",
    "job_name": "name",
    "jobname": "name",
    "output_file": "output",
    "scftyp": "scf_type",
}


@dataclass(frozen=True)
class ClassicLayout:
    """Line indexes a diatomic scan rewrites inside a classic deck."""

    atom_indexes: tuple[int, ...]
    title_index: int | None
    method_indexes: tuple[int, ...]


def deck_kind(text: str) -> str:
    """Classify a deck as ``sections``, ``legacy``, ``namelist``, or ``route``."""
    if _SECTION_HEADER.search(text):
        return "sections"
    if _NAMELIST_HINT.search(text):
        return "namelist"
    if _LEGACY_BLOCK.search(text):
        return "legacy"
    if _ROUTE_HINT.search(text) or _LINK0_HINT.search(text):
        return "route"
    return "sections"


def classic_blocks(text: str) -> tuple[dict[str, str], str] | None:
    """Translate a classic deck into ``[section]`` bodies plus the route card.

    Returns ``None`` when ``text`` is an older section or brace deck.
    """
    kind = deck_kind(text)
    if kind == "route":
        return _route_blocks(text.splitlines())
    if kind == "namelist":
        return _namelist_blocks(text.splitlines())
    return None


def classic_layout(text: str) -> ClassicLayout | None:
    """Locate geometry, title, and method lines in a classic deck."""
    kind = deck_kind(text)
    lines = text.splitlines()
    if kind == "route":
        walk = _walk_route(lines)
        _require_two_atoms(walk.atom_indexes)
        return ClassicLayout(
            atom_indexes=tuple(walk.atom_indexes),
            title_index=walk.title_index,
            method_indexes=tuple(walk.route_indexes),
        )
    if kind == "namelist":
        atoms, title, methods = _namelist_indexes(lines)
        _require_two_atoms(atoms)
        return ClassicLayout(
            atom_indexes=tuple(atoms),
            title_index=title,
            method_indexes=tuple(methods),
        )
    return None


@dataclass
class _Bucket:
    name: str | None = None
    output: str | None = None
    checkpoint: str | None = None
    charge: int | None = None
    multiplicity: int | None = None
    molecule: dict[str, str] = None  # type: ignore[assignment]
    psi4: dict[str, str] = None  # type: ignore[assignment]
    gf2: dict[str, str] = None  # type: ignore[assignment]
    atoms: list[str] = None  # type: ignore[assignment]
    modes: list[tuple[str, str]] = None  # type: ignore[assignment]
    spin_mode: str | None = None
    method: str | None = None
    route: str = ""
    explicit_units: bool = False

    def __post_init__(self) -> None:
        self.molecule = {}
        self.psi4 = {}
        self.gf2 = {}
        self.atoms = []
        self.modes = []


@dataclass
class _RouteWalk:
    route_indexes: list[int]
    title_index: int | None
    atom_indexes: list[int]
    route_text: str
    link0: list[tuple[str, str]]
    charge: int
    multiplicity: int
    tail: list[str]


def _require_two_atoms(indexes: list[int]) -> None:
    if len(indexes) != 2:
        raise ValueError(
            "diatomic PES workflow requires exactly two Cartesian atom rows"
        )


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def replace_method_line(line: str, method: str) -> str:
    """Rewrite a route token ``GF2/BASIS`` or a ``method`` keyword."""
    if re.search(r"(?i)\bmethod\b", line):
        updated = re.sub(
            r"(?i)(method\s*=\s*)(\S+)",
            lambda match: match.group(1) + method,
            line,
            count=1,
        )
        return re.sub(
            r"(?i)(method\s+)(\S+)",
            lambda match: match.group(1) + method,
            updated,
            count=1,
        )
    return re.sub(
        r"(?i)(^|\s)([A-Za-z][A-Za-z0-9_-]*)(/)",
        lambda match: f"{match.group(1)}{method.upper()}{match.group(3)}",
        line,
        count=1,
    )


def is_cartesian_atom(line: str) -> bool:
    """True for ``H x y z`` and ``H Z x y z`` rows."""
    tokens = line.split()
    if len(tokens) < 4 or not _ELEMENT.match(tokens[0]):
        return False
    start = 2 if len(tokens) >= 5 and _is_float(tokens[1]) else 1
    if len(tokens) < start + 3:
        return False
    return all(_is_float(tokens[start + offset]) for offset in range(3))


def _strip_bang(line: str) -> str:
    if "!" in line:
        return line[: line.index("!")].rstrip()
    return line.rstrip()


def _normalize_method(token: str) -> tuple[str, str | None]:
    key = token.strip().lower().replace("_", "-")
    if key not in _METHOD_ALIASES:
        raise ValueError("method must be GF2, UGF2, CS-GF2, UCS-GF2, or LF-GF2.")
    return _METHOD_ALIASES[key]


def _store(bucket: _Bucket, key: str, value: str) -> None:
    name = _KEY_ALIAS.get(key, key).lower()
    raw = value.strip()
    low = raw.lower()
    if name in {"scf_type", "reference"} and low in {"rhf", "uhf", "rohf"}:
        if low == "rohf":
            raise ValueError("ROHF is not supported. Use RHF or UHF.")
        bucket.spin_mode = "unrestricted" if low == "uhf" else "restricted"
        return
    if name == "method":
        method, spin = _normalize_method(raw)
        bucket.method = method
        if spin is not None:
            bucket.spin_mode = spin
        return
    if name == "spin_mode":
        spin = low.replace("_", "-")
        if spin not in {"restricted", "unrestricted"}:
            raise ValueError("spin must be restricted or unrestricted.")
        bucket.spin_mode = spin
        return
    if name == "charge":
        bucket.charge = int(raw)
        return
    if name == "multiplicity":
        bucket.multiplicity = int(raw)
        return
    if name == "name":
        bucket.name = raw
        return
    if name == "output":
        bucket.output = raw
        return
    if name == "n_threads":
        bucket.gf2["n_threads"] = str(int(raw))
        return
    if name in _MOLECULE_KEYS:
        if name == "units":
            bucket.explicit_units = True
            raw = _geometry_units(raw)
        elif name == "symmetry":
            raw = raw.lower()
        bucket.molecule[name] = raw
        return
    if name in _PSI4_KEYS:
        if name in {"basis", "scf_type"}:
            raw = raw.lower()
        bucket.psi4[name] = raw
        return
    if name.startswith("mode"):
        bucket.modes.append((name, raw))
        return
    bucket.gf2[name] = raw


def _geometry_units(token: str) -> str:
    key = token.strip().lower()
    if key in {"bohr", "au", "a.u.", "atomic"}:
        return "bohr"
    if key in {"angstrom", "angstroms", "ang"}:
        return "angstrom"
    raise ValueError("geometry units must be angstrom or bohr.")


def _render(bucket: _Bucket) -> tuple[dict[str, str], str]:
    if bucket.method is None:
        raise ValueError("The route card must name a method, for example GF2/cc-pVDZ.")
    if not bucket.atoms:
        raise ValueError("The deck must list at least one atom (symbol x y z).")
    charge = 0 if bucket.charge is None else bucket.charge
    multiplicity = 1 if bucket.multiplicity is None else bucket.multiplicity
    name = bucket.name or bucket.checkpoint or "untitled"
    job_lines = [f"name = {name}"]
    if bucket.output:
        job_lines.append(f"output = {bucket.output}")
    molecule = [
        f"charge = {charge}",
        f"multiplicity = {multiplicity}",
    ]
    if not bucket.explicit_units:
        molecule.append("units = angstrom")
    for key in ("symmetry", "units", "no_reorient", "no_com"):
        if key in bucket.molecule:
            molecule.append(f"{key} = {bucket.molecule[key]}")
    molecule.extend(bucket.atoms)
    gf2 = [f"method = {bucket.method}"]
    if bucket.spin_mode is not None:
        gf2.append(f"spin_mode = {bucket.spin_mode}")
    for key in sorted(bucket.gf2):
        gf2.append(f"{key} = {bucket.gf2[key]}")
    blocks = {
        "job": "\n".join(job_lines),
        "molecule": "\n".join(molecule),
        "gf2": "\n".join(gf2),
    }
    if bucket.psi4:
        blocks["psi4"] = "\n".join(
            f"{key} = {bucket.psi4[key]}" for key in sorted(bucket.psi4)
        )
    if bucket.modes:
        cavity = ["units = atomic"]
        for index, (key, value) in enumerate(bucket.modes, start=1):
            label = key if key.startswith("mode") and key[-1].isdigit() else f"mode{index}"
            cavity.append(f"{label} = {value}")
        blocks["cavity"] = "\n".join(cavity)
    return blocks, bucket.route


def _apply_route_token(bucket: _Bucket, token: str) -> None:
    if "/" in token and "=" not in token:
        method, basis = token.split("/", 1)
        _store(bucket, "method", method)
        if basis:
            _store(bucket, "basis", basis)
        return
    if "=" in token:
        key, value = token.split("=", 1)
        _store(bucket, key.strip().lower(), value.strip())
        return
    flag = token.lower()
    if flag not in _BARE_FLAGS:
        raise ValueError(f"Unrecognized route keyword {token!r}.")
    key, value = _BARE_FLAGS[flag]
    _store(bucket, key, "true" if value is True else "false" if value is False else str(value))


def _consume_keywords(bucket: _Bucket, lines: list[str]) -> None:
    for raw in lines:
        line = _strip_bang(raw).strip()
        if not line or line.startswith("#") or line.startswith("$"):
            continue
        if "=" in line:
            pieces = line.split()
            if len(pieces) > 1 and all("=" in piece for piece in pieces):
                for piece in pieces:
                    key, value = piece.split("=", 1)
                    _store(bucket, key.strip().lower(), value.strip())
                continue
            key, value = line.split("=", 1)
            _store(bucket, key.strip().lower(), value.strip())
            continue
        tokens = line.split()
        if tokens[0].lower().startswith("mode") and len(tokens) >= 6:
            _store(bucket, tokens[0].lower(), " ".join(tokens[1:]))
            continue
        if len(tokens) == 1:
            flag = tokens[0].lower()
            if flag not in _BARE_FLAGS:
                raise ValueError(f"Could not parse line: {raw!r}")
            key, value = _BARE_FLAGS[flag]
            _store(bucket, key, "true" if value is True else "false" if value is False else str(value))
            continue
        if len(tokens) % 2 != 0:
            raise ValueError(f"Could not parse line: {raw!r}")
        for index in range(0, len(tokens), 2):
            _store(bucket, tokens[index].lower(), tokens[index + 1])


def _walk_route(lines: list[str]) -> _RouteWalk:
    stripped = [_strip_bang(line) for line in lines]
    index = 0
    count = len(stripped)

    def blank(cursor: int) -> bool:
        return cursor >= count or not stripped[cursor].strip()

    link0: list[tuple[str, str]] = []
    while index < count:
        line = stripped[index].strip()
        if not line:
            index += 1
            continue
        if not line.startswith("%"):
            break
        match = re.match(r"(?i)%([A-Za-z]+)\s*=?\s*(.*)", line)
        if match is None:
            raise ValueError(f"Could not parse % line: {lines[index]!r}")
        link0.append((match.group(1).lower(), match.group(2).strip()))
        index += 1
    if index >= count or not stripped[index].lstrip().startswith("#"):
        raise ValueError("The input is missing a route card (a line beginning with #).")
    route_indexes: list[int] = []
    route_parts: list[str] = []
    while index < count and stripped[index].strip():
        route_indexes.append(index)
        route_parts.append(stripped[index].strip())
        index += 1
    if index < count and blank(index):
        index += 1
    title_index = None
    title_parts: list[str] = []
    while index < count and stripped[index].strip():
        if title_index is None:
            title_index = index
        title_parts.append(stripped[index].strip())
        index += 1
    if index < count and blank(index):
        index += 1
    if index >= count or not stripped[index].strip():
        raise ValueError("The input is missing the charge and multiplicity line.")
    charge_tokens = stripped[index].replace(",", " ").split()
    if len(charge_tokens) < 2 or not all(_is_int(token) for token in charge_tokens[:2]):
        raise ValueError("Expected charge and multiplicity after the title, for example: 0 1")
    charge, multiplicity = int(charge_tokens[0]), int(charge_tokens[1])
    index += 1
    atom_indexes: list[int] = []
    while index < count and stripped[index].strip():
        if is_cartesian_atom(stripped[index]):
            atom_indexes.append(index)
        index += 1
    if index < count and blank(index):
        index += 1
    route_text = " ".join(route_parts)
    route_text = re.sub(r"^\s*#[pPnNtT]?", "", route_text).strip()
    route_text = "#P " + route_text if route_text else "#P"
    return _RouteWalk(
        route_indexes=route_indexes,
        title_index=title_index,
        atom_indexes=atom_indexes,
        route_text=route_text,
        link0=link0,
        charge=charge,
        multiplicity=multiplicity,
        tail=stripped[index:],
    )


def _is_int(token: str) -> bool:
    try:
        int(token)
        return True
    except ValueError:
        return False


def _route_blocks(lines: list[str]) -> tuple[dict[str, str], str]:
    walk = _walk_route(lines)
    bucket = _Bucket(route=walk.route_text, charge=walk.charge, multiplicity=walk.multiplicity)
    if walk.title_index is not None:
        bucket.name = _strip_bang(lines[walk.title_index]).strip()
    for key, value in walk.link0:
        if key in {"chk", "chkfile"}:
            bucket.checkpoint = value
            continue
        _store(bucket, key, value)
    route_body = re.sub(r"(?i)^\s*#[pPnNtT]?\s*", "", " ".join(
        _strip_bang(lines[index]).strip() for index in walk.route_indexes
    ))
    for token in route_body.split():
        _apply_route_token(bucket, token)
    bucket.atoms = [_strip_bang(lines[index]).strip() for index in walk.atom_indexes]
    tail, cavity_lines = _split_cavity_tail(walk.tail)
    _consume_keywords(bucket, tail)
    _consume_keywords(bucket, cavity_lines)
    return _render(bucket)


def _split_cavity_tail(lines: list[str]) -> tuple[list[str], list[str]]:
    text = "\n".join(lines)
    match = re.search(r"(?is)\$cavity\b(.*?)\$end", text)
    if match is None:
        return lines, []
    cavity = match.group(1).splitlines()
    kept = (text[: match.start()] + text[match.end() :]).splitlines()
    return kept, cavity


def _namelist_blocks(lines: list[str]) -> tuple[dict[str, str], str]:
    groups = _namelist_groups(lines)
    bucket = _Bucket(route="")
    comment = groups.get("comment", "")
    comment_lines = [line.strip() for line in comment.splitlines() if line.strip()]
    if comment_lines:
        bucket.name = comment_lines[0]
    if "data" in groups:
        _load_data_block(bucket, groups["data"])
    if "molecule" in groups:
        _load_molecule_block(bucket, groups["molecule"])
    rem = "\n".join(part for part in (groups.get("rem", ""), groups.get("contrl", "")) if part)
    _consume_keywords(bucket, rem.splitlines())
    if "cavity" in groups:
        _consume_keywords(bucket, groups["cavity"].splitlines())
    method = bucket.method or "gf2"
    basis = bucket.psi4.get("basis", "")
    bucket.route = f"#P {method.upper()}/{basis}" if basis else f"#P {method.upper()}"
    if bucket.spin_mode == "unrestricted" and not bucket.route.split()[1].startswith("U"):
        card = bucket.route.split()[1]
        bucket.route = bucket.route.replace(card, "U" + card, 1)
    return _render(bucket)


def _load_molecule_block(bucket: _Bucket, body: str) -> None:
    rows = [line.strip() for line in body.splitlines() if line.strip() and not line.strip().startswith("#")]
    if rows:
        tokens = rows[0].replace(",", " ").split()
        if len(tokens) >= 2 and all(_is_int(token) for token in tokens[:2]):
            bucket.charge = int(tokens[0])
            bucket.multiplicity = int(tokens[1])
            rows = rows[1:]
    for row in rows:
        if is_cartesian_atom(row):
            bucket.atoms.append(row)
            continue
        _consume_keywords(bucket, [row])


def _load_data_block(bucket: _Bucket, body: str) -> None:
    rows = [line.strip() for line in body.splitlines() if line.strip()]
    if not rows:
        return
    if not is_cartesian_atom(rows[0]) and not _is_point_group(rows[0]):
        bucket.name = bucket.name or rows[0]
        rows = rows[1:]
    if rows and _is_point_group(rows[0]):
        bucket.molecule["symmetry"] = rows[0].lower()
        rows = rows[1:]
    for row in rows:
        if is_cartesian_atom(row):
            tokens = row.split()
            if len(tokens) >= 5 and _is_float(tokens[1]):
                bucket.atoms.append(" ".join((tokens[0], *tokens[2:5])))
            else:
                bucket.atoms.append(row)


def _is_point_group(line: str) -> bool:
    token = line.strip()
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", token)) and not is_cartesian_atom(line)


def _namelist_groups(lines: list[str]) -> dict[str, str]:
    groups: dict[str, list[str]] = {}
    current: str | None = None
    buffer: list[str] = []

    def close() -> None:
        nonlocal current, buffer
        if current is not None:
            groups.setdefault(current, []).append("\n".join(buffer))
        current = None
        buffer = []

    for raw in lines:
        remainder = _strip_bang(raw).strip()
        while remainder:
            match = re.match(r"(?i)\$([A-Za-z_][A-Za-z0-9_]*)\b\s*(.*)", remainder)
            if match is None:
                embedded = re.search(r"(?i)\$[A-Za-z_]", remainder)
                if embedded is not None and current is not None:
                    head = remainder[: embedded.start()].strip()
                    if head:
                        buffer.append(head)
                    remainder = remainder[embedded.start() :]
                    continue
                if current is not None:
                    buffer.append(remainder)
                break
            name = match.group(1).lower()
            remainder = match.group(2).strip()
            if name == "end":
                close()
                continue
            close()
            if name in {"rem", "contrl", "control"}:
                current = "rem" if name == "rem" else "contrl"
            else:
                current = name
        else:
            continue
    close()
    return {key: "\n".join(parts) for key, parts in groups.items()}


def _namelist_indexes(lines: list[str]) -> tuple[list[int], int | None, list[int]]:
    atoms: list[int] = []
    methods: list[int] = []
    title: int | None = None
    current: str | None = None
    for index, raw in enumerate(lines):
        remainder = _strip_bang(raw).strip()
        header = re.match(r"(?i)\$([A-Za-z_][A-Za-z0-9_]*)\b\s*(.*)", remainder)
        if header:
            name = header.group(1).lower()
            if name == "end":
                current = None
            elif name in {"rem", "contrl", "control"}:
                current = "rem"
            else:
                current = name
            remainder = header.group(2).strip()
            if not remainder:
                continue
        if current == "comment" and title is None and remainder:
            title = index
        if current == "data" and title is None and remainder and not _is_point_group(remainder):
            if not is_cartesian_atom(remainder):
                title = index
        if current in {"molecule", "data"} and remainder and is_cartesian_atom(remainder):
            atoms.append(index)
        if current == "rem" and re.match(r"(?i)method\b", remainder):
            methods.append(index)
    return atoms, title, methods
