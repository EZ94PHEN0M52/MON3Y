"""Short glossary strings for the NFL board."""

EDGE_CALLOUT = (
    "Edge = model side probability minus book implied probability (raw, not "
    "devigged). Positive edge means the model likes the posted price more than "
    "the market. Injury filter defaults to hiding Out / Doubtful / IR."
)

GLOSSARY = {
    "model_probability": (
        "Model probability for the selected side (Over or Under). "
        "Over % comes from P(actual > line)."
    ),
    "market_implied": (
        "Sportsbook implied probability from American odds (includes vig)."
    ),
    "edge": (
        "model side probability − book implied probability. "
        "Shown in percentage points."
    ),
    "ev": (
        "Expected value per $1 stake at the posted American odds using the "
        "model side probability."
    ),
    "injury": (
        "INACTIVE = Out / Doubtful / IR (excluded from Top by default). "
        "Q = Questionable. Refresh with `python fetch_data.py --injuries` "
        "before kickoff."
    ),
}
