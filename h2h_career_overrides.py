"""
Career H2H overrides (bat vs pitch) — lightweight add-on.

Statcast history in this repo is typically only ~1–2 seasons, so all-time
batter-vs-pitcher lines from ESPN / StatMuse can be richer. Those pairs live
in ``data/reference/h2h_career_overrides.csv`` (git-tracked, never written by
``run_daily.sh`` / feature builds).

Lookup order for board H2H and Batter Score pitcher-form H2H:
  1. Career override row for (batter, opposing SP) when present
  2. Else Statcast-derived H2H from merged shards

Appending new rows to the CSV is enough; no migrations or schema changes to
feature parquets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from utils import BASE_DIR

REFERENCE_DIR = BASE_DIR / "data" / "reference"
H2H_CAREER_OVERRIDES_PATH = REFERENCE_DIR / "h2h_career_overrides.csv"

REQUIRED_COLUMNS = ("batter", "pitcher", "hits", "ab")


@dataclass(frozen=True)
class CareerH2H:
    batter: str
    pitcher: str
    hits: int
    ab: int
    hr: Optional[int] = None
    rbi: Optional[int] = None
    team: Optional[str] = None
    source_url: Optional[str] = None

    @property
    def avg(self) -> float:
        return float(self.hits) / float(self.ab) if self.ab > 0 else 0.0

    @property
    def pa(self) -> int:
        """Board / score PA threshold uses AB when full PA is unknown."""
        return int(self.ab)


def _player_key(name: str) -> str:
    from utils import normalize_player_key

    return normalize_player_key(name)


def _fuzzy_player_key(player_name: str, candidate_keys) -> Optional[str]:
    from utils import strip_name_suffix

    if not isinstance(player_name, str):
        return None

    key = strip_name_suffix(_player_key(player_name))
    stripped_candidates = {
        candidate: strip_name_suffix(candidate)
        for candidate in candidate_keys
    }
    if key in stripped_candidates.values():
        for candidate, stripped in stripped_candidates.items():
            if stripped == key:
                return candidate

    parts = key.split()
    if len(parts) < 2:
        return None

    first_name, last_name = parts[0], parts[-1]
    matches = [
        candidate
        for candidate, stripped in stripped_candidates.items()
        if stripped.split()[-1] == last_name
        and stripped.split()[0] == first_name
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def overrides_cache_key() -> tuple:
    path = H2H_CAREER_OVERRIDES_PATH
    if not path.exists():
        return (None, 0)
    return (str(path), path.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _load_overrides_frame(cache_key: tuple) -> pd.DataFrame:
    path_str, _mtime = cache_key
    if path_str is None:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))

    frame = pd.read_csv(path_str)
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(
            f"h2h_career_overrides.csv missing columns: {missing}"
        )

    working = frame.copy()
    working["batter"] = working["batter"].astype(str).str.strip()
    working["pitcher"] = working["pitcher"].astype(str).str.strip()
    working["hits"] = pd.to_numeric(working["hits"], errors="coerce")
    working["ab"] = pd.to_numeric(working["ab"], errors="coerce")
    working = working.dropna(subset=["batter", "pitcher", "hits", "ab"])
    working = working[
        (working["ab"] > 0)
        & (working["hits"] >= 0)
        & (working["hits"] <= working["ab"])
    ]
    working["batter_key"] = working["batter"].map(_player_key)
    working["pitcher_key"] = working["pitcher"].map(_player_key)
    return working.reset_index(drop=True)


def load_career_h2h_overrides() -> pd.DataFrame:
    """Return the current overrides table (empty if the CSV is absent)."""
    return _load_overrides_frame(overrides_cache_key()).copy()


def lookup_career_h2h(
    batter_name: str,
    pitcher_name: str,
) -> Optional[CareerH2H]:
    """
    Return career H2H for *batter_name* vs *pitcher_name* when an override exists.

    Name matching uses the same fuzzy keys as the rest of the UI (suffixes,
    accents, Jr./Sr.).
    """
    if not batter_name or not pitcher_name:
        return None

    frame = _load_overrides_frame(overrides_cache_key())
    if frame.empty:
        return None

    batter_key = _fuzzy_player_key(batter_name, frame["batter_key"].tolist())
    if batter_key is None:
        return None

    subset = frame[frame["batter_key"].eq(batter_key)]
    if subset.empty:
        return None

    pitcher_key = _fuzzy_player_key(
        pitcher_name,
        subset["pitcher_key"].tolist(),
    )
    if pitcher_key is None:
        return None

    row = subset[subset["pitcher_key"].eq(pitcher_key)].iloc[-1]
    hr = row.get("hr")
    rbi = row.get("rbi")
    team = row.get("team")
    source = row.get("source_url")
    return CareerH2H(
        batter=str(row["batter"]),
        pitcher=str(row["pitcher"]),
        hits=int(row["hits"]),
        ab=int(row["ab"]),
        hr=None if pd.isna(hr) else int(hr),
        rbi=None if pd.isna(rbi) else int(rbi),
        team=None if (team is None or (isinstance(team, float) and pd.isna(team))) else str(team),
        source_url=(
            None
            if (source is None or (isinstance(source, float) and pd.isna(source)))
            else str(source)
        ),
    )


def upsert_career_h2h_rows(
    rows: list[dict],
    *,
    path: Optional[Path] = None,
) -> tuple[Path, dict[str, int]]:
    """
    Merge already-parsed *rows* into the overrides CSV.

    Safe merge rules (nothing else is deleted):
      • New (batter, pitcher) pairs are **appended**
      • Existing pairs that appear in *rows* are **updated** (replaced)
      • All other existing pairs are **kept unchanged**

    Keyed by normalized batter + pitcher name. This does **not** parse pasted
    tables — pass dicts with at least ``batter``, ``pitcher``, ``hits``, ``ab``.
    For file imports use ``import_career_h2h_from_text_file``.

    Returns ``(csv_path, {"kept": n, "updated": n, "added": n, "incoming": n})``.
    """
    target = path or H2H_CAREER_OVERRIDES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    empty_stats = {"kept": 0, "updated": 0, "added": 0, "incoming": 0}
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        if target.exists():
            existing_n = len(pd.read_csv(target))
            empty_stats["kept"] = existing_n
        return target, empty_stats

    for col in REQUIRED_COLUMNS:
        if col not in incoming.columns:
            raise ValueError(f"upsert rows missing column: {col}")

    # Dedupe incoming on the same key (last row wins)
    incoming["batter"] = incoming["batter"].astype(str).str.strip()
    incoming["pitcher"] = incoming["pitcher"].astype(str).str.strip()
    incoming["hits"] = pd.to_numeric(incoming["hits"], errors="coerce").astype(int)
    incoming["ab"] = pd.to_numeric(incoming["ab"], errors="coerce").astype(int)
    incoming["batter_key"] = incoming["batter"].map(_player_key)
    incoming["pitcher_key"] = incoming["pitcher"].map(_player_key)
    incoming = incoming.drop_duplicates(
        subset=["batter_key", "pitcher_key"],
        keep="last",
    )

    existing = (
        pd.read_csv(target)
        if target.exists()
        else pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    )
    incoming_keys = set(
        zip(incoming["batter_key"].tolist(), incoming["pitcher_key"].tolist())
    )
    incoming_n = len(incoming)

    if existing.empty:
        kept = 0
        updated = 0
        added = incoming_n
        kept_rows = existing
    else:
        existing["batter_key"] = existing["batter"].map(_player_key)
        existing["pitcher_key"] = existing["pitcher"].map(_player_key)
        existing_keys = list(
            zip(existing["batter_key"].tolist(), existing["pitcher_key"].tolist())
        )
        updated = sum(1 for key in existing_keys if key in incoming_keys)
        kept_mask = [
            key not in incoming_keys for key in existing_keys
        ]
        kept_rows = existing.loc[kept_mask].copy()
        kept = int(kept_mask.count(True))
        added = incoming_n - updated

    combined = pd.concat([kept_rows, incoming], ignore_index=True)
    drop_cols = [
        col for col in ("batter_key", "pitcher_key") if col in combined.columns
    ]
    combined = combined.drop(columns=drop_cols)
    preferred = [
        "batter",
        "pitcher",
        "hits",
        "ab",
        "hr",
        "rbi",
        "team",
        "source_url",
    ]
    ordered = [col for col in preferred if col in combined.columns]
    ordered.extend(col for col in combined.columns if col not in ordered)
    combined[ordered].to_csv(target, index=False)
    _load_overrides_frame.cache_clear()

    return target, {
        "kept": kept,
        "updated": updated,
        "added": added,
        "incoming": incoming_n,
    }


_MD_LINK_HAB_RE = re.compile(
    r"\[(?:\*\*)?(\d+)\s*/\s*(\d+)(?:\*\*)?\]\((https?://[^)\s]+)\)",
)
_BARE_HAB_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_URL_RE = re.compile(r"(https?://[^\s|]+)")


def _strip_md_noise(value: str) -> str:
    """Strip bold markers; keep markdown links intact for H/AB parsing."""
    text = str(value or "").strip()
    text = text.replace("**", "").strip()
    return " ".join(text.split())


def _strip_name_cell(value: str) -> str:
    """Name cells may be bare text or ``[Name](url)``."""
    text = _strip_md_noise(value)
    text = re.sub(r"^\[([^\]]+)\]\([^)]+\)$", r"\1", text)
    return " ".join(text.split())


def _parse_hab_cell(cell: str) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """Return (hits, ab, source_url) from an H/AB markdown cell."""
    text = str(cell or "").strip()
    if not text or text in {"—", "-", "–"}:
        return None, None, None

    link = _MD_LINK_HAB_RE.search(text)
    if link:
        return int(link.group(1)), int(link.group(2)), link.group(3)

    bare = _BARE_HAB_RE.search(text)
    if not bare:
        return None, None, None

    url_match = _URL_RE.search(text)
    return (
        int(bare.group(1)),
        int(bare.group(2)),
        url_match.group(1) if url_match else None,
    )


def _is_table_separator(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", c.replace(" ", "")) for c in cells if c)


def _is_header_row(cells: list[str]) -> bool:
    joined = " ".join(c.lower() for c in cells)
    return "batter" in joined and ("opposing" in joined or "pitcher" in joined or "sp" in joined)


def parse_career_h2h_table_text(text: str) -> list[dict]:
    """
    Parse a pasted markdown (or pipe-delimited) H2H table into row dicts.

    Expected columns (order flexible if a header row is present)::

        | Team | Batter | Opposing SP | H/AB | AVG | HR | RBI |

    ``H/AB`` may be plain ``5/11`` or a markdown link
    ``[5/11](https://...)`` / ``[**5/11**](https://...)``.

    Also accepts Word ``.docx`` exports where each cell is on its own line
    (repeating Team / Batter / Opposing SP / H/AB / AVG / HR / RBI).
    """
    if not text or not str(text).strip():
        return []

    pipe_rows = _parse_pipe_table_text(text)
    if pipe_rows:
        return pipe_rows
    return _parse_line_per_cell_table_text(text)


def _parse_pipe_table_text(text: str) -> list[dict]:
    rows: list[dict] = []
    col_map: Optional[dict[str, int]] = None

    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or "|" not in line:
            continue
        if _is_table_separator(line):
            continue

        cells = [_strip_md_noise(c) for c in line.strip("|").split("|")]
        if not any(cells):
            continue

        if _is_header_row(cells):
            col_map = {}
            for idx, name in enumerate(cells):
                key = name.lower()
                if "team" in key:
                    col_map["team"] = idx
                elif "batter" in key or key == "hitter":
                    col_map["batter"] = idx
                elif "opposing" in key or "pitcher" in key or key in {"sp", "vs sp"}:
                    col_map["pitcher"] = idx
                elif "h/ab" in key or key in {"hab", "h-ab", "hits/ab"}:
                    col_map["hab"] = idx
                elif key in {"hr", "home runs", "home run"}:
                    col_map["hr"] = idx
                elif key in {"rbi", "rbis"}:
                    col_map["rbi"] = idx
                elif key in {"avg", "ba", "average"}:
                    col_map["avg"] = idx
            continue

        # Default layout when no header was seen: Team | Batter | SP | H/AB | AVG | HR | RBI
        if col_map is None:
            if len(cells) < 4:
                continue
            team = _strip_name_cell(cells[0])
            batter = _strip_name_cell(cells[1])
            pitcher = _strip_name_cell(cells[2])
            hab_cell = cells[3]
            hr_cell = cells[5] if len(cells) > 5 else ""
            rbi_cell = cells[6] if len(cells) > 6 else ""
        else:
            def _cell(key: str) -> str:
                idx = col_map.get(key)
                if idx is None or idx >= len(cells):
                    return ""
                return cells[idx]

            team = _strip_name_cell(_cell("team"))
            batter = _strip_name_cell(_cell("batter"))
            pitcher = _strip_name_cell(_cell("pitcher"))
            hab_cell = _cell("hab")
            hr_cell = _cell("hr")
            rbi_cell = _cell("rbi")

        row = _row_from_fields(
            team=team,
            batter=batter,
            pitcher=pitcher,
            hab_cell=hab_cell,
            hr_cell=hr_cell,
            rbi_cell=rbi_cell,
        )
        if row is not None:
            rows.append(row)

    return rows


def _normalize_header_label(value: str) -> str:
    key = _strip_md_noise(value).lower()
    # Word sometimes exports "H/AB ↗"
    key = re.sub(r"[^\w/]+", " ", key).strip()
    if key.startswith("h/ab") or key in {"hab", "h ab", "hits/ab"}:
        return "hab"
    if "batter" in key or key == "hitter":
        return "batter"
    if "opposing" in key or "pitcher" in key or key in {"sp", "vs sp"}:
        return "pitcher"
    if key == "team":
        return "team"
    if key in {"hr", "home runs", "home run"}:
        return "hr"
    if key in {"rbi", "rbis"}:
        return "rbi"
    if key in {"avg", "ba", "average"}:
        return "avg"
    return ""


def _parse_line_per_cell_table_text(text: str) -> list[dict]:
    """
    Parse Word exports where each table cell is on its own line.

    Example::

        Team
        Batter
        Opposing SP
        H/AB
        AVG
        HR
        RBI
        LAA
        Zach Neto
        Joe Ryan
        5/11
        .455
        0
        1
    """
    lines = [_strip_md_noise(line) for line in str(text).splitlines()]
    lines = [line for line in lines if line]
    if len(lines) < 11:
        return []

    # Detect a header block of known labels at the start
    header_keys: list[str] = []
    idx = 0
    while idx < len(lines) and len(header_keys) < 8:
        label = _normalize_header_label(lines[idx])
        if not label:
            break
        if label in header_keys:
            break
        header_keys.append(label)
        idx += 1

    expected = ["team", "batter", "pitcher", "hab", "avg", "hr", "rbi"]
    if header_keys[:4] != expected[:4]:
        # Fall back to fixed 7-column order if the header is missing/odd
        header_keys = expected
        idx = 0
        # Skip a leading header if present even when order differed slightly
        probe = [_normalize_header_label(x) for x in lines[:7]]
        if probe[:4] == expected[:4]:
            idx = 7

    width = len(header_keys)
    if width < 4:
        return []

    rows: list[dict] = []
    while idx + width <= len(lines):
        chunk = lines[idx : idx + width]
        idx += width
        fields = dict(zip(header_keys, chunk))
        row = _row_from_fields(
            team=_strip_name_cell(fields.get("team", "")),
            batter=_strip_name_cell(fields.get("batter", "")),
            pitcher=_strip_name_cell(fields.get("pitcher", "")),
            hab_cell=fields.get("hab", ""),
            hr_cell=fields.get("hr", ""),
            rbi_cell=fields.get("rbi", ""),
        )
        if row is not None:
            rows.append(row)

    return rows


def _row_from_fields(
    *,
    team: str,
    batter: str,
    pitcher: str,
    hab_cell: str,
    hr_cell: str = "",
    rbi_cell: str = "",
) -> Optional[dict]:
    hits, ab, source_url = _parse_hab_cell(hab_cell)
    if not batter or not pitcher or hits is None or ab is None:
        return None
    if ab <= 0 or hits < 0 or hits > ab:
        return None

    # Skip leftover header leftovers mistaken as data
    if _normalize_header_label(batter) or _normalize_header_label(pitcher):
        return None

    row: dict = {
        "batter": batter,
        "pitcher": pitcher,
        "hits": hits,
        "ab": ab,
    }
    if team and not _normalize_header_label(team):
        row["team"] = team
    if source_url:
        row["source_url"] = source_url

    hr_val = pd.to_numeric(hr_cell, errors="coerce")
    if not pd.isna(hr_val):
        row["hr"] = int(hr_val)
    rbi_val = pd.to_numeric(rbi_cell, errors="coerce")
    if not pd.isna(rbi_val):
        row["rbi"] = int(rbi_val)
    return row


def _extract_text_from_docx(raw: bytes) -> Optional[str]:
    """Pull visible text from a .docx (Office Open XML) byte blob."""
    import zipfile
    from xml.etree import ElementTree as ET
    from io import BytesIO

    if not raw.startswith(b"PK"):
        return None

    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            if "word/document.xml" not in archive.namelist():
                return None
            xml_bytes = archive.read("word/document.xml")
    except zipfile.BadZipFile:
        return None

    root = ET.fromstring(xml_bytes)
    # WordprocessingML text nodes
    texts = [
        node.text
        for node in root.iter()
        if node.tag.endswith("}t") and node.text
    ]
    if not texts:
        return None

    # Join runs; paragraphs often need newlines — insert newline before <w:p>
    parts: list[str] = []
    for node in root.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "p":
            if parts and not parts[-1].endswith("\n"):
                parts.append("\n")
        elif tag == "t" and node.text:
            parts.append(node.text)
        elif tag == "tab":
            parts.append("\t")
    return "".join(parts).strip()


def _read_text_file(path: Path) -> str:
    """
    Read *path* as text.

    Supports UTF-8 / common legacy encodings, and Word ``.docx`` files
    (including ``.docx`` saved with a ``.txt`` extension).
    """
    raw = path.read_bytes()

    # Word documents are zip archives starting with PK
    if raw.startswith(b"PK"):
        docx_text = _extract_text_from_docx(raw)
        if docx_text:
            return docx_text
        raise ValueError(
            f"{path} looks like a zip/Office file but no Word document.xml "
            "was found. Export as Plain Text (.txt) or real .docx and retry."
        )

    for encoding in ("utf-8", "utf-8-sig", "cp1252", "mac_roman", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def import_career_h2h_from_text_file(
    text_path: str | Path,
    *,
    csv_path: Optional[Path] = None,
) -> tuple[Path, int, dict[str, int]]:
    """
    Read a text/markdown/docx file of H2H rows, parse it, and upsert into the CSV.

    Only adds new pairs or updates matching (batter, pitcher) pairs — never
    deletes unrelated existing rows.

    Returns ``(csv_path, n_rows_parsed, merge_stats)``.
    """
    path = Path(text_path)
    if not path.exists():
        raise FileNotFoundError(f"H2H text file not found: {path}")

    rows = parse_career_h2h_table_text(_read_text_file(path))
    if not rows:
        raise ValueError(
            f"No H2H rows parsed from {path}. Expected a pipe table with "
            "Team | Batter | Opposing SP | H/AB | …"
        )

    out, stats = upsert_career_h2h_rows(rows, path=csv_path)
    return out, len(rows), stats
