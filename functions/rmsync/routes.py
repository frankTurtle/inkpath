"""Route each watched folder to its own destination in the vault.

Without this every note lands under one path, so book notes and journals end up
in the same flat pile. A route says where a folder's notes go and how they are
grouped once they get there.

Spec (env var VAULT_ROUTES), entries separated by ";"::

    <watchFolderId>=<vaultPath>[|<mode>]

    ba5dfa9f-...=Book Notes;d18f316e-...=Journals|year

Modes:
    notebook  (default)  <vaultPath>/<notebook name>/<title>.md
    year                 <vaultPath>/<year from notebook name>/<title>.md
    flat                 <vaultPath>/<title>.md
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MODE_NOTEBOOK = "notebook"
MODE_YEAR = "year"
MODE_FLAT = "flat"
MODES = {MODE_NOTEBOOK, MODE_YEAR, MODE_FLAT}

_YEAR_RE = re.compile(r"(19|20)\d{2}")


@dataclass(frozen=True)
class Route:
    vault_path: str
    mode: str = MODE_NOTEBOOK


def parse_routes(raw: str) -> dict[str, Route]:
    """Parse the VAULT_ROUTES spec. An unparseable entry is skipped, loudly."""
    routes: dict[str, Route] = {}
    for entry in (raw or "").split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            logger.warning("Ignoring malformed route %r (expected id=path)", entry)
            continue
        folder_id, _, dest = entry.partition("=")
        path, _, mode = dest.partition("|")
        mode = (mode or MODE_NOTEBOOK).strip().lower()
        if mode not in MODES:
            logger.warning("Unknown route mode %r for %s; using %s", mode, folder_id, MODE_NOTEBOOK)
            mode = MODE_NOTEBOOK
        routes[folder_id.strip()] = Route(path.strip().strip("/"), mode)
    return routes


def year_of(notebook: str) -> str | None:
    """First four-digit year in a notebook name: 'Journal 2021' -> '2021'."""
    match = _YEAR_RE.search(notebook or "")
    return match.group(0) if match else None


def vault_dir(
    notebook: str,
    *,
    route: Route | None,
    default_path: str,
) -> str:
    """Directory a notebook's notes belong in, without the filename."""
    from .enrich import sanitize_path_component

    if route is None:
        base, mode = default_path, MODE_NOTEBOOK
    else:
        base, mode = route.vault_path, route.mode

    base = base.strip("/")
    if mode == MODE_FLAT:
        return base

    if mode == MODE_YEAR:
        year = year_of(notebook)
        if year:
            return f"{base}/{year}" if base else year
        # No year in the name - keep the notebook rather than dumping it at the
        # route root, where unrelated notebooks would collide.
        logger.info("No year found in notebook %r; grouping by notebook name", notebook)

    folder = sanitize_path_component(notebook, fallback="reMarkable")
    return f"{base}/{folder}" if base else folder
