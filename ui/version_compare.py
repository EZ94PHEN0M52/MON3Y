"""Side-by-side Over/Under % comparison across project generations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from odds_aggregation import dedupe_best_prop
from predict import generate_predictions
from ui.formatting import (
    enrich_with_over_under_probs,
    format_pct,
    market_label,
)
from ui.market_filters import exclude_ui_markets
from ui.player import render_back_to_board
from utils import (
    VERSION_COMPARE_SLOTS,
    compare_predictions_path,
    version_has_models,
)

TOP_N = 30
PREDICT_START = "2026-03-25"
PREDICT_END = "2026-08-16"
_GEN_BUTTON_KEY = "version_compare_generate_missing"
_GEN_FEEDBACK_KEY = "version_compare_gen_feedback"

MERGE_KEYS = ("player", "market")


def _slot_sort_column(df: pd.DataFrame) -> str:
    for column in ("ev", "edge", "over_probability"):
        if column in df.columns:
            return column
    return "over_probability"


def _dedupe_slot(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    working = enrich_with_over_under_probs(df)
    return dedupe_best_prop(
        working,
        sort_col=_slot_sort_column(working),
    )


def _load_slot_csv(slot_key: str) -> pd.DataFrame | None:
    path = compare_predictions_path(slot_key)
    if not path.exists():
        return None

    df = exclude_ui_markets(pd.read_csv(path))
    if df.empty:
        return None

    return _dedupe_slot(df)


def _slot_frame(slot: dict) -> pd.DataFrame | None:
    key = slot["key"]
    df = _load_slot_csv(key)
    if df is None or df.empty:
        return None

    keep = [
        column
        for column in (
            "player",
            "market",
            "line",
            "over_probability",
            "under_probability",
            "edge",
        )
        if column in df.columns
    ]

    renamed = df[keep].rename(
        columns={
            "line": f"{key}_line",
            "over_probability": f"{key}_over",
            "under_probability": f"{key}_under",
            "edge": f"{key}_edge",
        }
    )
    return renamed


def _ensure_slot_columns(merged: pd.DataFrame) -> pd.DataFrame:
    """Always expose every version column; missing slots stay NaN → display as —."""
    result = merged.copy()

    for slot in VERSION_COMPARE_SLOTS:
        key = slot["key"]
        for suffix in ("line", "over", "under", "edge"):
            column = f"{key}_{suffix}"
            if column not in result.columns:
                result[column] = pd.NA

    return result


def _merge_compare_frames(
    slot_frames: dict[str, pd.DataFrame | None],
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None

    for slot in VERSION_COMPARE_SLOTS:
        key = slot["key"]
        frame = slot_frames.get(key)
        if frame is None or frame.empty:
            continue

        if merged is None:
            merged = frame.copy()
            continue

        merged = merged.merge(
            frame,
            on=list(MERGE_KEYS),
            how="outer",
        )

    if merged is None:
        return pd.DataFrame()

    merged = _ensure_slot_columns(merged)

    line_cols = [f"{slot['key']}_line" for slot in VERSION_COMPARE_SLOTS]
    present_line_cols = [
        column for column in line_cols if column in merged.columns
    ]
    if "main_line" in merged.columns:
        merged["line"] = merged["main_line"]
    elif present_line_cols:
        merged["line"] = merged[present_line_cols[0]]
        for column in present_line_cols[1:]:
            merged["line"] = merged["line"].fillna(merged[column])

    return merged


def _rank_score(row: pd.Series) -> float:
    if "main_edge" in row.index and pd.notna(row["main_edge"]):
        return abs(float(row["main_edge"]))

    edge_cols = [
        column
        for column in row.index
        if column.endswith("_edge")
    ]
    edges = pd.to_numeric(row[edge_cols], errors="coerce")
    if edges.notna().any():
        return float(edges.abs().max())

    over_cols = [
        column
        for column in row.index
        if column.endswith("_over")
    ]
    overs = pd.to_numeric(row[over_cols], errors="coerce")
    if overs.notna().any():
        return float(overs.max())

    return 0.0


def _top_compare_rows(merged: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    if merged.empty:
        return merged

    ranked = merged.copy()
    ranked["_rank"] = ranked.apply(_rank_score, axis=1)
    ranked = ranked.sort_values("_rank", ascending=False).head(n)
    return ranked.drop(columns="_rank").reset_index(drop=True)


def _format_compare_display(merged: pd.DataFrame) -> pd.DataFrame:
    display = merged.copy()

    if "market" in display.columns:
        display["market"] = display["market"].map(
            lambda value: market_label(value) if pd.notna(value) else value
        )

    prob_cols = [
        column
        for column in display.columns
        if column.endswith("_over") or column.endswith("_under")
    ]
    for column in prob_cols:
        display[column] = display[column].apply(format_pct)

    if "line" in display.columns:
        display["line"] = pd.to_numeric(
            display["line"],
            errors="coerce",
        ).round(1)

    return display


def _compare_column_config() -> dict:
    config = {
        "player": st.column_config.TextColumn("Player", width="medium"),
        "market": st.column_config.TextColumn("Market", width="medium"),
        "line": st.column_config.NumberColumn("Line", format="%.1f"),
    }

    for slot in VERSION_COMPARE_SLOTS:
        key = slot["key"]
        label = slot["label"]
        config[f"{key}_over"] = st.column_config.TextColumn(
            f"{label} Over",
            width="small",
        )
        config[f"{key}_under"] = st.column_config.TextColumn(
            f"{label} Under",
            width="small",
        )

    return config


def _display_column_order() -> list[str]:
    columns = ["player", "market", "line"]
    for slot in VERSION_COMPARE_SLOTS:
        key = slot["key"]
        columns.extend([f"{key}_over", f"{key}_under"])
    return columns


def _slot_availability() -> list[dict]:
    rows = []
    for slot in VERSION_COMPARE_SLOTS:
        path = compare_predictions_path(slot["key"])
        models_ok = version_has_models(slot["model_version"])
        rows.append(
            {
                "slot": slot["label"],
                "predictions": "yes" if path.exists() else "missing",
                "models": "yes" if models_ok else "missing",
                "path": str(path.name),
            }
        )
    return rows


def _generate_missing_predictions() -> list[str]:
    """Run predict for missing CSVs. V3 is manual; Main shares V2's file."""
    generated = []
    seen_paths: set[str] = set()

    for slot in VERSION_COMPARE_SLOTS:
        key = slot["key"]
        if key in {"v3", "main"}:
            continue

        path = compare_predictions_path(key)
        path_key = str(path.resolve())
        if path.exists() or path_key in seen_paths:
            continue

        model_version = slot["model_version"]
        if not version_has_models(model_version):
            continue

        generate_predictions(
            PREDICT_START,
            PREDICT_END,
            version=model_version,
        )
        generated.append(slot["label"])
        seen_paths.add(path_key)

    return generated


def _set_generate_feedback(level: str, message: str) -> None:
    st.session_state[_GEN_FEEDBACK_KEY] = (level, message)


def _render_generate_feedback() -> None:
    feedback = st.session_state.pop(_GEN_FEEDBACK_KEY, None)
    if not feedback:
        return

    level, message = feedback
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)


def _on_generate_missing_predictions() -> None:
    try:
        generated = _generate_missing_predictions()
        load_version_compare_table.clear()
        if generated:
            _set_generate_feedback(
                "success",
                "Generated predictions for: " + ", ".join(generated),
            )
        else:
            _set_generate_feedback(
                "warning",
                "Nothing generated — files may already exist or models "
                "are missing (see Version sources).",
            )
    except Exception as exc:
        load_version_compare_table.clear()
        _set_generate_feedback(
            "error",
            f"Failed to generate predictions: {exc}",
        )


@st.cache_data(show_spinner="Loading version comparison…")
def load_version_compare_table(cache_key: tuple[tuple[str, float], ...]):
    _ = cache_key
    slot_frames = {
        slot["key"]: _slot_frame(slot)
        for slot in VERSION_COMPARE_SLOTS
    }
    merged = _merge_compare_frames(slot_frames)
    return _top_compare_rows(merged)


def _compare_cache_key() -> tuple[tuple[str, float], ...]:
    key = []
    for slot in VERSION_COMPARE_SLOTS:
        path = compare_predictions_path(slot["key"])
        mtime = path.stat().st_mtime if path.exists() else 0.0
        key.append((slot["key"], mtime))
    return tuple(key)


def render_version_compare_page():
    render_back_to_board("view")

    st.title("Version compare")
    st.caption(
        "Top **30** unique player props (one best book per player and market), "
        f"with **Over %** and **Under %** from each project generation side-by-side. "
        f"Feature window: **{PREDICT_START}** → **{PREDICT_END}**."
    )

    availability = _slot_availability()
    available_labels = [
        row["slot"]
        for row in availability
        if row["predictions"] == "yes"
    ]
    missing_labels = [
        row["slot"]
        for row in availability
        if row["predictions"] == "missing"
    ]

    with st.expander("Version sources", expanded=not available_labels):
        for slot, row in zip(VERSION_COMPARE_SLOTS, availability):
            status = (
                "loaded"
                if row["predictions"] == "yes"
                else "— (file missing)"
            )
            st.markdown(
                f"**{slot['label']}** — {slot['description']}  \n"
                f"`{row['path']}` · predictions: **{status}** · "
                f"models/{slot['model_version']}/: "
                f"**{'yes' if row['models'] == 'yes' else 'missing'}**"
            )

        if missing_labels:
            st.info(
                "**V3** predictions are not generated in this workspace — copy "
                "`predictions_v2.csv` from the frozen "
                "[mlb-prop-model-v3](../mlb-prop-model-v3/) snapshot to "
                "`data/predictions/predictions_v3.csv` to populate that column. "
                "**V2** and **Main** share `predictions_v2.csv` from the current "
                "daily pipeline."
            )

    action_row = st.container(horizontal=True)
    with action_row:
        generate_clicked = st.button(
            "Generate missing predictions",
            type="primary",
            key=_GEN_BUTTON_KEY,
            help=(
                "Run predict.py for V1 and/or V2 when CSVs are missing and "
                "models exist. V3 is not auto-generated."
            ),
        )

    if generate_clicked:
        with st.spinner("Running predict for missing versions…"):
            _on_generate_missing_predictions()

    _render_generate_feedback()

    compare_df = load_version_compare_table(_compare_cache_key())

    if compare_df.empty:
        st.warning(
            "No predictions loaded for comparison. Run the daily pipeline or "
            "click **Generate missing predictions** above."
        )
        missing_models = [
            row["slot"]
            for row in availability
            if row["models"] == "missing"
        ]
        if missing_models:
            st.caption(
                "Models missing for: "
                + ", ".join(missing_models)
                + ". Those columns will show **—** until models and CSVs exist."
            )
        return

    compare_df = _ensure_slot_columns(compare_df)
    display = _format_compare_display(compare_df)
    display_columns = _display_column_order()

    loaded_count = len(available_labels)
    st.caption(
        f"Showing **{len(display)}** props ranked by **Main edge** (or max "
        f"|edge| / max Over % across loaded versions). "
        f"**{loaded_count}** of **{len(VERSION_COMPARE_SLOTS)}** "
        f"version columns loaded ({', '.join(available_labels) or 'none'})."
    )

    st.dataframe(
        display[display_columns],
        hide_index=True,
        column_config=_compare_column_config(),
    )

    st.divider()
    st.caption(
        "Over % and Under % are model estimates from each generation's "
        "predict.py output (best book per player and market). "
        "This is a statistical research tool, not a guarantee of betting outcomes."
    )
