"""Tooltip and glossary text for the MLB Prop Model UI."""

MARKET_LABELS = {
    "batter_hits": "Hits",
    "batter_home_runs": "Home Runs",
    "batter_total_bases": "Total Bases",
    "batter_rbis": "RBIs",
    "batter_runs_scored": "Runs",
    "batter_walks": "Batter Walks",
    "batter_hits_runs_rbis": "Hits + Runs + RBIs",
    "batter_stolen_bases": "Stolen Bases",
    "pitcher_strikeouts": "Strikeouts",
    "pitcher_walks": "Pitcher Walks",
    "pitcher_hits_allowed": "Hits Allowed",
    "pitcher_outs": "Pitcher Outs",
    "pitcher_earned_runs": "Earned Runs",
}

GLOSSARY = {
    "model_pct": (
        "The model's estimated probability that the listed Side (Over or "
        "Under) hits, based on recent Statcast form and (in V2) opponent, "
        "handedness, and park features. When calibrators are fitted, this "
        "matches Calibrated %; otherwise it is the raw LightGBM output."
    ),
    "calibrated_pct": (
        "Isotonic- or Platt-calibrated probability for the listed side, "
        "fit on held-out historical outcomes. Edge and EV use this value "
        "when calibrators exist in models/v2/calibrators/."
    ),
    "predicted_count": (
        "Expected stat count (μ) from the Poisson regressor head for "
        "pitcher strikeouts, walks, and outs. Shown alongside the classifier "
        "Over % / Under %; edge and EV still use the classifier path in "
        "Phase 1 (50/50 blend deferred to a later iteration)."
    ),
    "dist_over_probability": (
        "Poisson regressor P(stat > line) for pitcher strikeouts, walks, "
        "and outs. Derived from Pred # and the posted line. Visible for "
        "comparison with classifier Over %; not used for edge or EV in "
        "Phase 1."
    ),
    "over_pct": (
        "Model probability that the player exceeds the posted line "
        "(stat > line). Computed once per player, market, book, and line — "
        "the same value whether the row lists Over or Under odds."
    ),
    "under_pct": (
        "Model probability that the player stays at or below the posted line "
        "(stat ≤ line). Equals 100% minus Over % for the same prop."
    ),
    "market_pct": (
        "Implied win probability from the sportsbook's American odds. "
        "Does not remove vig (the book's built-in margin). See Devigged % "
        "for vig-adjusted fair probability when Over/Under pairs exist."
    ),
    "devigged_market_pct": (
        "Fair implied probability after removing two-way vig from the "
        "book's Over/Under pair. Edge and EV use this value when both "
        "sides are available; otherwise raw Market % is used."
    ),
    "consensus_line": (
        "Median posted line across sportsbooks for this player, market, "
        "and game. Useful for spotting books off the market number."
    ),
    "consensus_edge": (
        "Model % minus the weighted consensus devigged market probability "
        "for the listed side. Positive means the model favors this side "
        "relative to the multi-book consensus."
    ),
    "line_delta": (
        "Change in the posted line since the day's opening snapshot "
        "(current line minus opening line). Negative means the line dropped "
        "(easier Over); positive means it rose (harder Over). Requires "
        "intraday odds snapshots from fetch_data.py --props."
    ),
    "odds_delta": (
        "Change in American odds since the opening snapshot for this "
        "book, player, market, and side."
    ),
    "steam_flag": (
        "Line moved at least 0.5 points toward the side that was favored "
        "at the opening snapshot (higher implied probability). Useful for "
        "spotting sharp or heavy action before first pitch."
    ),
    "best_book": (
        "Sportsbook offering the best price (highest EV) for this player, "
        "market, line, and side among all books on the slate."
    ),
    "best_ev": (
        "Expected value at the best available price for this player, market, "
        "line, and side across all books."
    ),
    "best_price_view": (
        "The main board always shows one row per player and market — the "
        "book and line with the highest EV across all sportsbooks."
    ),
    "edge_pct": (
        "Model % minus Devigged Market % for the listed side (Over or Under). "
        "Positive edge means the model thinks that side is more likely than "
        "the fair price suggests — the bet may be underpriced. Negative edge "
        "means the market price looks generous relative to the model."
    ),
    "ev_pct": (
        "Expected profit per $1 wagered if the calibrated model probability "
        "is accurate: (model_prob × decimal_odds) − 1. Uses devigged market "
        "context for edge; EV ranks bets vig-aware. Positive EV means positive "
        "expected return over many similar bets."
    ),
    "side": (
        "Whether the line is Over or Under the posted threshold. "
        "Edge and EV are computed for this specific side."
    ),
    "line": (
        "The prop threshold set by the sportsbook (e.g. 0.5 hits, 5.5 "
        "strikeouts). Over wins if the stat exceeds the line; Under wins "
        "if it stays at or below."
    ),
    "odds": (
        "American odds from this sportsbook for the listed side. "
        "Negative odds (e.g. -110) show how much you must bet to win $100; "
        "positive odds (e.g. +150) show profit on a $100 bet."
    ),
    "book": "The sportsbook offering this line and price.",
    "game": "Today's matchup (away team @ home team).",
    "game_total_line": (
        "Consensus over/under total runs for this game from sportsbook "
        "totals markets. Used as a model context feature (Phase 5), not "
        "a standalone betting market in the UI."
    ),
    "game_run_line": (
        "Consensus run line (spread) for the player's team in this game. "
        "Negative values mean the team is favored to win by that many runs."
    ),
    "game_implied_total_over_prob": (
        "Devigged implied probability that the game goes Over the posted "
        "total runs line, aggregated across books."
    ),
    "market": "The player prop type (hits, strikeouts, total bases, etc.).",
    "commence_time": "Scheduled first pitch time for this game.",
    "min_edge": (
        "Optional filter: show only props where model edge "
        "(Model % − Market %) is at least this value. "
        "Default 0% shows all props; edge is always visible in the Edge % column."
    ),
    "min_ev": (
        "Optional filter: show only props where expected value is at least "
        "this value. Default 0% shows all props; EV is always visible in the "
        "EV % column."
    ),
    "filter_market": "Show only props in the selected markets.",
    "filter_player": (
        "Filter rows by player name. Partial matches are supported "
        "(case-insensitive)."
    ),
    "filter_game": "Show only props from the selected matchups.",
    "filter_book": "Show only lines from the selected sportsbooks.",
    "filter_side": "Show only Over or Under sides.",
    "filter_line": (
        "Filter by the sportsbook prop threshold (e.g. 0.5 hits, "
        "5.5 strikeouts)."
    ),
    "filter_odds": (
        "Filter by American odds for the listed side. Negative odds "
        "require a larger stake to win $100; positive odds pay more "
        "on a $100 bet."
    ),
    "filter_model_pct": "Show props where model probability is at least this value.",
    "filter_over_pct": "Show props where model Over % is at least this value.",
    "filter_under_pct": "Show props where model Under % is at least this value.",
    "filter_market_pct": (
        "Show props where market implied probability is at least this value."
    ),
    "filter_edge_pct": (
        "Show props where edge (Model % − Market %) is at least this value."
    ),
    "filter_ev_pct": (
        "Show props where expected value is at least this value."
    ),
    "l5_l10_pct": (
        "Share of the player's last 5 and last 10 completed games (from Statcast "
        "feature data) where the market stat strictly exceeded the posted line. "
        "Shown as L5% / L10% — e.g. 45% / 52% means the player went over this "
        "line in 45% of their last 5 games and 52% of their last 10."
    ),
    "filter_l5": (
        "Show props where the L5 over-rate is at least this value."
    ),
    "filter_l10": (
        "Show props where the L10 over-rate is at least this value."
    ),
    "filter_h2h": (
        "Show props where the player's head-to-head average is at least this value."
    ),
    "column_filters": (
        "Choose which table columns to display. Row filters live in "
        "**Filter by column**; market type is controlled by **Market type** "
        "at the top of the board."
    ),
    "header_click_help": (
        "Click a **column header** to sort (up to 3 columns — first clicked "
        "is primary, shown as subscript ₁, then ₂, ₃). Click again to reverse "
        "direction (↑↓). Active column filters show a subscript in "
        "**Filter by column** (₁ = first active filter). Use **Market type** "
        "at the top for prop categories. Open **Filters & columns** for "
        "minimum Edge / EV and column visibility."
    ),
    "header_click_sort": (
        "Click to add or toggle sort for this column. Up to 3 sort keys; "
        "first clicked has priority ₁."
    ),
    "header_click_filter": (
        "Open filter controls for this column."
    ),
    "player_link": "Click a name to view all props for that player.",
    "best_edge": (
        "Highest edge among this player's props. Edge is always for the "
        "Side shown in that row."
    ),
    "best_ev": (
        "Highest expected value among this player's props at posted odds."
    ),
    "prop_count": "Total number of sportsbook lines for this player.",
    "market_count": "Number of distinct prop markets available for this player.",
    "last_10_games": (
        "Per-game stat totals from the player's most recent 10 games in the "
        "feature dataset (Statcast game logs). Bars run oldest (left) to "
        "newest (right). These are actual outcomes, not rolling averages."
    ),
    "props": "Number of prop lines matching the current filters.",
    "players": "Number of unique players matching the current filters.",
    "top_over_list": (
        "All props ranked by model Over % (highest to lowest). Uses the same "
        "Market / Edge / EV filters as the main board. One best book per "
        "player and market — use **Market type** to narrow the list."
    ),
    "top_under_list": (
        "All props ranked by model Under % (highest to lowest). Uses the same "
        "Market / Edge / EV filters as the main board. One best book per "
        "player and market."
    ),
    "batter_score": (
        "Composite 0–100 rating for a batter's upcoming game. **Phase D** "
        "adds usage-weighted pitch-type matchup (30%) when the opposing SP "
        "and Statcast arsenal data are available. **Phase B** adds SP ERA "
        "(L5) and optional H2H vs that SP (≥10 PA) in pitcher form (15%). "
        "Season baseline (30%) and recent form (25%) always active when data "
        "exists. Label **Full** when all four components are active; "
        "**Partial** when matchup or SP data is missing. Orthogonal to "
        "LightGBM prop probabilities."
    ),
    "batter_score_season_baseline": (
        "Full-season per-game average of hits + total bases + walks (H+TB+BB), "
        "scaled to a 0–100 index (6.0 raw points = 100)."
    ),
    "batter_score_recent_form": (
        "0.7×L5 + 0.3×L10 blend of the same H+TB+BB raw points per game."
    ),
    "batter_score_matchup_grade": (
        "Usage-weighted wOBA (35%) + AVG (65%) vs Fastball / Slider / "
        "Curveball / Changeup / Other buckets, blended across the opposing "
        "SP's arsenal from Statcast. Active in Phase D when SP usage data "
        "exists; shown as — when gated."
    ),
    "batter_score_pitcher_form": (
        "Opposing starter ERA over last 5 starts (letter-graded; lower ERA "
        "= better for the batter). When the batter has ≥10 PA vs that SP, "
        "H2H H+TB+BB performance is blended in (30% of pitcher form). Gated "
        "off when SP is TBD; optional team-level opp_team_earned_runs proxy "
        "at reduced weight when enabled in code."
    ),
    "batter_score_h2h": (
        "Career plate appearances vs the listed opposing starter from "
        "Statcast. Included in pitcher form only when PA ≥ 10; below "
        "threshold the signal is omitted (not zeroed)."
    ),
    "batter_score_partial": (
        "**Form only** or **Partial · SP TBD** — matchup and/or pitcher "
        "components are excluded when the opposing starter is unknown or "
        "Phase D pitch-type data is unavailable. Active weights renormalize "
        "to 100%; partial scores are not directly comparable to full scores "
        "without context."
    ),
    "batter_score_validated": (
        "Shown when a historical backtest (`scripts/backtest_batter_score.py`) "
        "passes configured gates (minimum sample size + Spearman correlation "
        "vs same-game H+TB+BB outcomes). Indicates the composite tracks real "
        "production modestly; it does **not** mean Batter Score drives board "
        "edge or EV until explicitly wired in."
    ),
    "pp_fantasy_line": (
        "PrizePicks posted Over fantasy score line for this hitter (from the "
        "daily us_dfs fetch). Used as the threshold for the L5 / L10 % column."
    ),
    "filter_batter_score": (
        "Show props where Batter Score is at least this value (0–100)."
    ),
    "stat_history": (
        "Per-game values for any modeled stat market (batter or pitcher). "
        "Use **All** for the last 5 / 10 games overall, or **H2H** to filter "
        "to games vs today's opponent on the slate. L5 and L10 averages "
        "use the same filtered sample; the chart shows the selected window."
    ),
    "pick_builder": (
        "**Pick Builder** is your session favorites slip — like Pickfinder's "
        "builder, without export. Star props from the main board or player "
        "pages to track them in one place while you browse.\n\n"
        "Picks persist for this browser session only (`st.session_state`). "
        "The same prop cannot be added twice (matched on player, market, "
        "side, line, and book). Use **Remove** on a pick or **Clear all** "
        "to reset.\n\n"
        "Each pick shows **Player**, **Market**, **Side**, **Line**, "
        "**Over % / Under %** (your side is bold when both are shown), "
        "**Edge %**, **Game** (matchup + first-pitch time ET), and "
        "**Batter Score** (batter markets only, with partial label when "
        "applicable) — all frozen at add time."
    ),
    "pick_builder_add": (
        "Select one or more rows from the **currently filtered table** below "
        "the board, then **Add selected**. **Add top EV** adds the highest-EV "
        "row in view. Duplicates are skipped automatically."
    ),
}

EDGE_CALLOUT = (
    "**Reading edge:** Edge uses **devigged** market probability when the "
    "book posts both Over and Under prices (Model % − Devigged Market %). "
    "It is always computed for the **Side** shown in that row. A **+52% edge "
    "on Under 0.5 hits** means the model assigns 52 percentage points more "
    "probability to Under than the fair (devigged) price implies."
)
