"""Scoped visual styles for the Sleeper Prop Analyzer (Sleeper Picks page only)."""

from __future__ import annotations

import streamlit as st

ANALYZER_SHELL_KEY = "sleeper_analyzer_shell"
ANALYZER_UI_SCALE = 0.75
ANALYZER_BORDER_WIDTH = "4px"
ANALYZER_FONT_BUMP = "4px"
ANALYZER_TEXT_GAP = "0.75rem"
ANALYZER_BLOCK_GAP = "0.75rem"

ANALYZER_CSS = f"""
<style>
:root {{
    --sa-border: {ANALYZER_BORDER_WIDTH} solid #111111;
    --sa-font-bump: {ANALYZER_FONT_BUMP};
    --sa-text-gap: {ANALYZER_TEXT_GAP};
    --sa-block-gap: {ANALYZER_BLOCK_GAP};
}}

/* Shrink the whole analyzer panel to 75% of its native size. */
.st-key-{ANALYZER_SHELL_KEY} {{
    zoom: {ANALYZER_UI_SCALE};
}}

@supports not (zoom: 1) {{
    .st-key-{ANALYZER_SHELL_KEY} {{
        transform: scale({ANALYZER_UI_SCALE});
        transform-origin: top center;
        width: calc(100% / {ANALYZER_UI_SCALE});
        margin-left: auto;
        margin-right: auto;
    }}
}}

/* Flat white canvas. */
.st-key-{ANALYZER_SHELL_KEY},
.st-key-{ANALYZER_SHELL_KEY} > div[data-testid="stVerticalBlockBorderWrapper"] {{
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
}}

/* Equal spacing between stacked UI blocks. */
.st-key-{ANALYZER_SHELL_KEY} [data-testid="stVerticalBlock"] {{
    gap: var(--sa-block-gap) !important;
}}

.st-key-{ANALYZER_SHELL_KEY} [data-testid="stHorizontalBlock"] {{
    gap: var(--sa-block-gap) !important;
}}

/* Bold + larger text everywhere inside the analyzer. */
.st-key-{ANALYZER_SHELL_KEY} p,
.st-key-{ANALYZER_SHELL_KEY} span,
.st-key-{ANALYZER_SHELL_KEY} label,
.st-key-{ANALYZER_SHELL_KEY} button,
.st-key-{ANALYZER_SHELL_KEY} [data-testid="stMarkdownContainer"] p,
.st-key-{ANALYZER_SHELL_KEY} [data-testid="stMarkdownContainer"] li,
.st-key-{ANALYZER_SHELL_KEY} [data-testid="stText"],
.st-key-{ANALYZER_SHELL_KEY} [data-testid="stCaptionContainer"] p,
.st-key-{ANALYZER_SHELL_KEY} [data-testid="stWidgetLabel"] p {{
    font-weight: 700 !important;
    color: #111111 !important;
}}

.st-key-{ANALYZER_SHELL_KEY} .sa-flow-caption,
.st-key-{ANALYZER_SHELL_KEY} .sa-context-line,
.st-key-{ANALYZER_SHELL_KEY} .sa-page-subtitle,
.st-key-{ANALYZER_SHELL_KEY} .sa-caption,
.st-key-{ANALYZER_SHELL_KEY} [data-testid="stCaptionContainer"] p {{
    font-size: calc(0.82rem + var(--sa-font-bump)) !important;
    line-height: 1.4 !important;
    margin: 0 0 var(--sa-text-gap) 0 !important;
}}

.st-key-{ANALYZER_SHELL_KEY} .sa-body-text,
.st-key-{ANALYZER_SHELL_KEY} [data-testid="stText"],
.st-key-{ANALYZER_SHELL_KEY} [data-testid="stMarkdownContainer"] p:not([class^="sa-"]) {{
    font-size: calc(1rem + var(--sa-font-bump)) !important;
    margin: 0 0 var(--sa-text-gap) 0 !important;
}}

.st-key-{ANALYZER_SHELL_KEY} .sa-page-title,
.st-key-{ANALYZER_SHELL_KEY} .sa-game-title,
.st-key-{ANALYZER_SHELL_KEY} .sa-section-title {{
    font-size: calc(1.55rem + var(--sa-font-bump)) !important;
    font-weight: 700 !important;
    margin: 0 0 var(--sa-text-gap) 0 !important;
    color: #111111 !important;
}}

.st-key-{ANALYZER_SHELL_KEY} .sa-game-title {{
    font-size: calc(1.2rem + var(--sa-font-bump)) !important;
}}

.st-key-{ANALYZER_SHELL_KEY} .sa-table-header {{
    font-size: calc(0.95rem + var(--sa-font-bump)) !important;
    font-weight: 700 !important;
    margin: 0 0 var(--sa-text-gap) 0 !important;
}}

.st-key-{ANALYZER_SHELL_KEY} .sa-player-name,
.st-key-{ANALYZER_SHELL_KEY} .sa-player-prop {{
    font-size: calc(1rem + var(--sa-font-bump)) !important;
    font-weight: 700 !important;
    margin: 0 !important;
    color: #111111 !important;
}}

.st-key-{ANALYZER_SHELL_KEY} .sa-player-prop {{
    text-align: center !important;
}}

/* Thick borders on cards. */
.st-key-{ANALYZER_SHELL_KEY} [class*="st-key-sleeper_analyzer_player_"] > div[data-testid="stVerticalBlockBorderWrapper"],
.st-key-{ANALYZER_SHELL_KEY} [class*="st-key-sleeper_analyzer_game_"] > div[data-testid="stVerticalBlockBorderWrapper"],
.st-key-{ANALYZER_SHELL_KEY} [class*="st-key-sleeper_analyzer_prop_"] > div[data-testid="stVerticalBlockBorderWrapper"],
.st-key-{ANALYZER_SHELL_KEY} [class*="st-key-sleeper_analyzer_player_"][data-testid="stVerticalBlockBorderWrapper"],
.st-key-{ANALYZER_SHELL_KEY} [class*="st-key-sleeper_analyzer_game_"][data-testid="stVerticalBlockBorderWrapper"],
.st-key-{ANALYZER_SHELL_KEY} [class*="st-key-sleeper_analyzer_prop_"][data-testid="stVerticalBlockBorderWrapper"] {{
    border: var(--sa-border) !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    background: #ffffff !important;
    padding: 0.45rem 0.65rem !important;
}}

/* Outlined buttons — thick border + bold text. */
.st-key-{ANALYZER_SHELL_KEY} button {{
    background: #ffffff !important;
    color: #111111 !important;
    border: var(--sa-border) !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: calc(0.95rem + var(--sa-font-bump)) !important;
    min-height: 2rem !important;
    padding: 0.25rem 0.75rem !important;
    box-shadow: none !important;
}}

.st-key-{ANALYZER_SHELL_KEY} button:hover {{
    background: #f7f7f7 !important;
    border-color: #000000 !important;
    color: #000000 !important;
}}

.st-key-{ANALYZER_SHELL_KEY} [class*="st-key-sleeper_analyzer_player_"] button {{
    min-width: 4.75rem !important;
    width: 100% !important;
}}

.st-key-{ANALYZER_SHELL_KEY} [class*="st-key-sleeper_analyzer_back_"] button {{
    width: auto !important;
}}

.st-key-{ANALYZER_SHELL_KEY} [data-testid="stDateInput"] > div,
.st-key-{ANALYZER_SHELL_KEY} [data-testid="stDateInput"] input {{
    border: var(--sa-border) !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    font-weight: 700 !important;
    font-size: calc(0.95rem + var(--sa-font-bump)) !important;
}}

.st-key-{ANALYZER_SHELL_KEY} [data-testid="stWidgetLabel"] p {{
    font-size: calc(0.95rem + var(--sa-font-bump)) !important;
    margin-bottom: var(--sa-text-gap) !important;
}}

.st-key-{ANALYZER_SHELL_KEY} [data-testid="stProgressBar"] > div {{
    border: 3px solid #111111 !important;
    border-radius: 6px !important;
    overflow: hidden;
    height: 0.65rem !important;
}}
</style>
"""


def inject_analyzer_styles() -> None:
    st.markdown(ANALYZER_CSS, unsafe_allow_html=True)
