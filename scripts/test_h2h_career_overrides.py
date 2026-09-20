"""Tests for career H2H override store."""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from h2h_career_overrides import (  # noqa: E402
    import_career_h2h_from_text_file,
    lookup_career_h2h,
    load_career_h2h_overrides,
    parse_career_h2h_table_text,
)


SAMPLE_TABLE = """
| Team | Batter              | Opposing SP      | H/AB | AVG | HR | RBI |
| ---- | ------------------- | ---------------- | ---- | --- | -- | --- |
| LAA  | **Zach Neto**       | **Joe Ryan**     | [**5/11**](https://www.espn.com/mlb/player/batvspitch/_/id/42450/joe-ryan) | **.455** |  0 |   1 |
| NYY  | Aaron Judge         | Brandon Pfaadt   | [3/3](https://www.espn.com/mlb/player/batvspitch/_/id/4721302/brandon-pfaadt) | 1.000 | 2 | 5 |
| TEX  | Jake Burger         | José Soriano     | 4/7 | .571 | 0 | 2 |
"""


def test_seeded_neto_vs_ryan():
    row = lookup_career_h2h("Zach Neto", "Joe Ryan")
    assert row is not None
    assert row.hits == 5
    assert row.ab == 11
    assert abs(row.avg - 5 / 11) < 1e-9


def test_accent_insensitive_pitcher_match():
    row = lookup_career_h2h("Jake Burger", "José Soriano")
    assert row is not None
    assert row.hits == 4
    assert row.ab == 7


def test_missing_pair_returns_none():
    assert lookup_career_h2h("Zach Neto", "Unknown Pitcher") is None
    assert lookup_career_h2h("Nobody", "Joe Ryan") is None


def test_overrides_table_loads():
    frame = load_career_h2h_overrides()
    assert not frame.empty
    assert {"batter", "pitcher", "hits", "ab"}.issubset(frame.columns)
    assert len(frame) >= 30


def test_parse_career_h2h_table_text():
    rows = parse_career_h2h_table_text(SAMPLE_TABLE)
    assert len(rows) == 3

    by_batter = {row["batter"]: row for row in rows}
    assert by_batter["Zach Neto"]["hits"] == 5
    assert by_batter["Zach Neto"]["ab"] == 11
    assert "espn.com" in by_batter["Zach Neto"]["source_url"]
    assert by_batter["Aaron Judge"]["hr"] == 2
    assert by_batter["Jake Burger"]["pitcher"] == "José Soriano"
    assert by_batter["Jake Burger"]["hits"] == 4


def test_parse_line_per_cell_word_export():
    text = """
Team
Batter
Opposing SP
H/AB ↗
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
NYY
Aaron Judge
Brandon Pfaadt
3/3
1.000
2
5
"""
    rows = parse_career_h2h_table_text(text)
    assert len(rows) == 2
    assert rows[0]["batter"] == "Zach Neto"
    assert rows[0]["hits"] == 5
    assert rows[0]["ab"] == 11
    assert rows[1]["batter"] == "Aaron Judge"


def test_import_career_h2h_from_text_file():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        text_path = out_dir / "paste.txt"
        text_path.write_text(SAMPLE_TABLE, encoding="utf-8")
        csv_path = out_dir / "h2h_career_overrides.csv"

        written, n, stats = import_career_h2h_from_text_file(
            text_path,
            csv_path=csv_path,
        )
        assert n == 3
        assert written == csv_path
        assert stats["added"] == 3
        assert stats["updated"] == 0
        assert stats["kept"] == 0

        import pandas as pd

        frame = pd.read_csv(csv_path)
        assert len(frame) == 3
        neto = frame[frame["batter"].eq("Zach Neto")].iloc[0]
        assert int(neto["hits"]) == 5
        assert int(neto["ab"]) == 11


def test_upsert_keeps_unrelated_rows():
    """Importing new pairs must not delete other batters' H2H rows."""
    from h2h_career_overrides import upsert_career_h2h_rows

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "h2h.csv"
        upsert_career_h2h_rows(
            [
                {
                    "batter": "Zach Neto",
                    "pitcher": "Joe Ryan",
                    "hits": 5,
                    "ab": 11,
                },
                {
                    "batter": "Aaron Judge",
                    "pitcher": "Brandon Pfaadt",
                    "hits": 3,
                    "ab": 3,
                },
            ],
            path=csv_path,
        )

        # Update Neto vs Ryan only; Judge must remain
        _, stats = upsert_career_h2h_rows(
            [
                {
                    "batter": "Zach Neto",
                    "pitcher": "Joe Ryan",
                    "hits": 6,
                    "ab": 12,
                },
                {
                    "batter": "Manny Machado",
                    "pitcher": "Sandy Alcantara",
                    "hits": 9,
                    "ab": 15,
                },
            ],
            path=csv_path,
        )
        assert stats["kept"] == 1
        assert stats["updated"] == 1
        assert stats["added"] == 1

        import pandas as pd

        frame = pd.read_csv(csv_path)
        assert len(frame) == 3
        judge = frame[frame["batter"].eq("Aaron Judge")].iloc[0]
        assert int(judge["hits"]) == 3
        neto = frame[frame["batter"].eq("Zach Neto")].iloc[0]
        assert int(neto["hits"]) == 6
        assert int(neto["ab"]) == 12


if __name__ == "__main__":
    test_seeded_neto_vs_ryan()
    test_accent_insensitive_pitcher_match()
    test_missing_pair_returns_none()
    test_overrides_table_loads()
    test_parse_career_h2h_table_text()
    test_parse_line_per_cell_word_export()
    test_import_career_h2h_from_text_file()
    test_upsert_keeps_unrelated_rows()
    print("OK")
