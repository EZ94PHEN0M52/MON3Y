"""Shared batter score board / pick card conditional highlight styles."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd

HIT_RATE_THRESHOLD = 0.80
VS_PITCHER_AVG_THRESHOLD = 0.300
STYLE_FANTASY_LOWER = "background-color: rgba(255, 152, 0, 0.45)"
STYLE_FANTASY_EQUAL = "background-color: rgba(135, 206, 235, 0.55)"
STYLE_L5_L10_YELLOW = "background-color: rgba(255, 235, 59, 0.45)"
STYLE_L5_L10_GREEN = "background-color: rgba(76, 175, 80, 0.40)"
STYLE_VS_PITCHER_AVG = "background-color: rgba(144, 238, 144, 0.50)"
STYLE_ROW_HIGHLIGHT = "box-shadow: inset 0 0 0 2px #e53935"


def join_styles(*parts: str) -> str:
    return "; ".join(part for part in parts if part)


def fantasy_cell_styles(pp_line, ud_line) -> tuple[str, str, bool]:
    """
    PP/UD fantasy highlights: orange on the lower line, sky blue when equal.

    Returns ``(pp_style, ud_style, ud_is_lower)`` for row-border combo logic.
    """
    if (
        pp_line is None
        or ud_line is None
        or pd.isna(pp_line)
        or pd.isna(ud_line)
    ):
        return "", "", False

    pp_value = float(pp_line)
    ud_value = float(ud_line)

    if pp_value == ud_value:
        return STYLE_FANTASY_EQUAL, STYLE_FANTASY_EQUAL, False

    if pp_value < ud_value:
        return STYLE_FANTASY_LOWER, "", False

    return "", STYLE_FANTASY_LOWER, True


def l5_l10_style(l5_pct, l10_pct) -> str:
    l5_hit = not pd.isna(l5_pct) and float(l5_pct) >= HIT_RATE_THRESHOLD
    l10_hit = not pd.isna(l10_pct) and float(l10_pct) >= HIT_RATE_THRESHOLD
    l10_below = not pd.isna(l10_pct) and float(l10_pct) < HIT_RATE_THRESHOLD

    if l5_hit and l10_hit:
        return STYLE_L5_L10_GREEN
    if l5_hit and l10_below:
        return STYLE_L5_L10_YELLOW
    return ""


def h2h_avg_from_vs_pitcher(text) -> float | None:
    """Parse AVG from a Vs pitcher cell like ``4/10 .400`` (first line only)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None

    cell = str(text).splitlines()[0].strip()
    if not cell or cell == "—" or cell.startswith("SP ERA"):
        return None

    parts = cell.split()
    if len(parts) < 2:
        return None

    avg_token = parts[-1]
    if not avg_token.startswith("."):
        return None

    try:
        return float(f"0{avg_token}")
    except ValueError:
        return None


def vs_pitcher_style(vs_pitcher_text) -> str:
    h2h_avg = h2h_avg_from_vs_pitcher(vs_pitcher_text)
    if h2h_avg is not None and h2h_avg > VS_PITCHER_AVG_THRESHOLD:
        return STYLE_VS_PITCHER_AVG
    return ""


def combo_row_highlight(
    pp_line,
    ud_line,
    l5_pct,
    l10_pct,
) -> bool:
    _, _, ud_is_lower = fantasy_cell_styles(pp_line, ud_line)
    l5_hit = not pd.isna(l5_pct) and float(l5_pct) >= HIT_RATE_THRESHOLD
    l10_hit = not pd.isna(l10_pct) and float(l10_pct) >= HIT_RATE_THRESHOLD
    return ud_is_lower and l5_hit and l10_hit


def pick_field_styles(pick: dict) -> dict[str, str]:
    """CSS background styles for colored fields on a frozen batter score pick."""
    pp_line = pick.get("pp_line")
    ud_line = pick.get("ud_line")
    l5_pct = pick.get("l5_pct")
    l10_pct = pick.get("l10_pct")

    pp_style, ud_style, _ = fantasy_cell_styles(pp_line, ud_line)

    return {
        "pp_fantasy_line": pp_style,
        "ud_fantasy_line": ud_style,
        "l5_l10_pct": l5_l10_style(l5_pct, l10_pct),
        "vs_pitcher": vs_pitcher_style(pick.get("vs_pitcher")),
        "card_border": (
            STYLE_ROW_HIGHLIGHT
            if combo_row_highlight(pp_line, ud_line, l5_pct, l10_pct)
            else ""
        ),
    }


def highlight_html(text, css_style: str) -> str:
    display = html.escape(str(text if text not in (None, "") else "—"))
    if not css_style:
        return display
    return (
        f'<span style="{css_style}; padding: 1px 4px; '
        f'border-radius: 3px;">{display}</span>'
    )


def format_batter_score_pick_details_html(pick: dict) -> str:
    """Caption-sized HTML for sidebar pick cards with board-matched highlights."""
    styles = pick_field_styles(pick)
    vs_pitcher = highlight_html(pick.get("vs_pitcher"), styles["vs_pitcher"])
    pp = highlight_html(pick.get("pp_fantasy_line"), styles["pp_fantasy_line"])
    ud = highlight_html(pick.get("ud_fantasy_line"), styles["ud_fantasy_line"])
    l5_l10 = highlight_html(pick.get("l5_l10_pct"), styles["l5_l10_pct"])
    caption = "margin:0 0 0.25rem 0; font-size:0.875rem; opacity:0.85;"
    return (
        f'<p style="{caption}">Vs pitcher: {vs_pitcher}</p>'
        f'<p style="margin:0; font-size:0.875rem; opacity:0.85;">'
        f"PP {pp} · UD {ud} · L5/L10 {l5_l10}</p>"
    )
