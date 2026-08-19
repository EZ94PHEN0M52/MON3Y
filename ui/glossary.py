"""Tooltip and glossary text for the MLB Prop Model UI."""

MARKET_LABELS = {
    "batter_hits": "Hits",
    "batter_home_runs": "Home Runs",
    "batter_total_bases": "Total Bases",
    "batter_rbis": "RBIs",
    "batter_runs_scored": "Runs",
    "batter_walks": "Walks",
    "batter_hits_runs_rbis": "Hits + Runs + RBIs",
    "pitcher_strikeouts": "Strikeouts",
    "pitcher_walks": "Walks",
    "pitcher_hits_allowed": "Hits Allowed",
    "pitcher_outs": "Pitcher Outs",
    "pitcher_earned_runs": "Earned Runs",
}

GLOSSARY = {
    "model_pct": (
        "The model's estimated probability that the listed Side (Over or "
        "Under) hits, based on recent Statcast form and (in V2) opponent, "
        "handedness, and park features."
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
        "Does not remove vig (the book's built-in margin)."
    ),
    "edge_pct": (
        "Model % minus Market % for the listed side (Over or Under). "
        "Positive edge means the model thinks that side is more likely than "
        "the price suggests — the bet may be underpriced. Negative edge "
        "means the market price looks generous relative to the model."
    ),
    "ev_pct": (
        "Expected profit per $1 wagered if the model probability is accurate: "
        "(model_prob × decimal_odds) − 1. Positive EV means positive expected "
        "return over many similar bets."
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
    "market": "The player prop type (hits, strikeouts, total bases, etc.).",
    "commence_time": "Scheduled first pitch time for this game.",
    "min_edge": (
        "Filter props where model edge (Model % − Market %) is at least "
        "this value. Higher edge = stronger disagreement with the market."
    ),
    "min_ev": (
        "Filter props where expected value is at least this value. "
        "EV accounts for both probability and payout odds."
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
        "Per-column filters refine the board on top of the Market / Edge / EV "
        "controls above."
    ),
    "header_click_help": (
        "Click a **column header** to sort that column (↑ ascending, ↓ descending). "
        "Open **Filter** under a header to narrow rows for that column. "
        "A dot (•) on a header means a filter is active."
    ),
    "header_click_sort": (
        "Click to sort by this column. Click again to reverse sort direction."
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
        "Market / Edge / EV filters as the main board. Every prop row is "
        "shown — not limited to one per player."
    ),
    "top_under_list": (
        "All props ranked by model Under % (highest to lowest), excluding "
        "home run markets. Uses the same Market / Edge / EV filters as the "
        "main board. Every prop row is shown — not limited to one per player."
    ),
}

EDGE_CALLOUT = (
    "**Reading edge:** Edge is always computed for the **Side** shown in "
    "that row. A **+52% edge on Under 0.5 hits** means the model assigns "
    "52 percentage points more probability to Under than the book's odds "
    "imply — the model favors that Under bet relative to the posted price."
)
