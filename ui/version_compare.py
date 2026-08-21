"""Side-by-side Over/Under % comparison across project generations."""

from __future__ import annotations

import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from odds_aggregation import dedupe_best_prop
from predict import generate_predictions
from ui.formatting import (
    enrich_with_over_under_probs,
    format_pct,
    market_label,
)
from ui.market_filters import exclude_ui_markets, render_market_multiselect
from ui.player import render_back_to_board
from utils import (
    VERSION_COMPARE_SLOTS,
    compare_predictions_path,
    version_has_models,
    version_models_dir,
)

TOP_N = 30
PREDICT_START = "2026-03-25"
# Opening-day anchor; effective end rolls forward through today so daily CSVs
# stay in-window after the season moves past a fixed research cutoff.
PREDICT_END = "2026-08-16"
_GEN_BUTTON_KEY = "version_compare_generate_missing"
_GEN_FEEDBACK_KEY = "version_compare_gen_feedback"
_FORCE_REGEN_KEY = "version_compare_force_regenerate"

MERGE_KEYS = ("player", "market")
ROOT = Path(__file__).resolve().parent.parent


def _compare_window_end() -> str:
    """Last inclusive calendar date for compare filtering."""
    anchor = date.fromisoformat(PREDICT_END)
    return max(anchor, date.today()).isoformat()


def _predict_feature_end() -> str:
    """Feature parquet end date when running predict from version compare."""
    anchor = date.fromisoformat(PREDICT_END)
    yesterday = date.today() - timedelta(days=1)
    return max(anchor, yesterday).isoformat()


def _compare_window_label() -> str:
    end = _compare_window_end()
    if end == PREDICT_END:
        return f"{PREDICT_START} → {PREDICT_END}"
    return f"{PREDICT_START} → {PREDICT_END} (extended through **{end}**)"


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


def _filter_compare_window(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows whose commence_time falls in the compare feature window."""
    if df.empty or "commence_time" not in df.columns:
        return df

    times = pd.to_datetime(
        df["commence_time"],
        errors="coerce",
        utc=True,
    )
    start = pd.Timestamp(PREDICT_START, tz="UTC")
    end = pd.Timestamp(f"{_compare_window_end()} 23:59:59", tz="UTC")
    mask = times.notna() & (times >= start) & (times <= end)
    return df.loc[mask].copy()


def _has_usable_probabilities(df: pd.DataFrame) -> bool:
    if df.empty:
        return False

    if "over_probability" in df.columns:
        if pd.to_numeric(df["over_probability"], errors="coerce").notna().any():
            return True

    if "model_probability" in df.columns:
        return pd.to_numeric(
            df["model_probability"],
            errors="coerce",
        ).notna().any()

    return False


def _read_slot_raw_csv(slot_key: str) -> pd.DataFrame | None:
    path = compare_predictions_path(slot_key)
    if not path.exists():
        return None

    df = exclude_ui_markets(pd.read_csv(path))
    if df.empty:
        return None

    return df


def _slot_window_rows(slot_key: str) -> pd.DataFrame | None:
    raw = _read_slot_raw_csv(slot_key)
    if raw is None:
        return None

    windowed = _filter_compare_window(raw)
    if windowed.empty or not _has_usable_probabilities(windowed):
        return None

    return windowed


def _slot_diagnostics(slot: dict) -> dict:
    key = slot["key"]
    path = compare_predictions_path(key)
    models_dir = version_models_dir(slot["model_version"])
    model_count = (
        len(list(models_dir.glob("*.pkl")))
        if models_dir.exists()
        else 0
    )
    models_ok = model_count > 0

    raw = _read_slot_raw_csv(key)
    raw_rows = len(raw) if raw is not None else 0
    window_df = _slot_window_rows(key)
    window_rows = len(window_df) if window_df is not None else 0

    date_min = date_max = None
    if raw is not None and not raw.empty and "commence_time" in raw.columns:
        times = pd.to_datetime(
            raw["commence_time"],
            errors="coerce",
            utc=True,
        ).dropna()
        if not times.empty:
            date_min = times.min().date().isoformat()
            date_max = times.max().date().isoformat()

    if not path.exists():
        status = "missing"
    elif window_rows > 0:
        status = "ready"
    elif raw_rows > 0:
        status = "stale"
    else:
        status = "empty"

    return {
        "slot": slot["label"],
        "key": key,
        "status": status,
        "models": "yes" if models_ok else "missing",
        "model_count": model_count,
        "path": str(path.name),
        "file_exists": path.exists(),
        "raw_rows": raw_rows,
        "window_rows": window_rows,
        "date_min": date_min,
        "date_max": date_max,
    }


def _load_slot_csv(slot_key: str) -> pd.DataFrame | None:
    windowed = _slot_window_rows(slot_key)
    if windowed is None:
        return None

    return _dedupe_slot(windowed)


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


def _version_overlap_count(row: pd.Series) -> int:
    over_cols = [
        column
        for column in row.index
        if column.endswith("_over")
    ]
    values = pd.to_numeric(row[over_cols], errors="coerce")
    return int(values.notna().sum())


def _top_compare_rows(merged: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    if merged.empty:
        return merged

    ranked = merged.copy()
    ranked["_rank"] = ranked.apply(_rank_score, axis=1)
    ranked["_overlap"] = ranked.apply(_version_overlap_count, axis=1)

    # Prefer props scored by multiple versions on the same player+market.
    overlap = ranked[ranked["_overlap"] >= 2].copy()
    if len(overlap) >= n:
        ranked = overlap

    ranked = ranked.sort_values(
        ["_overlap", "_rank"],
        ascending=[False, False],
    ).head(n)
    return ranked.drop(columns=["_rank", "_overlap"]).reset_index(drop=True)


def _slot_date_ranges(availability: list[dict]) -> dict[str, tuple[str, str] | None]:
    ranges = {}
    for row in availability:
        if row["date_min"] and row["date_max"]:
            ranges[row["key"]] = (row["date_min"], row["date_max"])
        else:
            ranges[row["key"]] = None
    return ranges


def _dates_overlap(
    left: tuple[str, str] | None,
    right: tuple[str, str] | None,
) -> bool:
    if left is None or right is None:
        return False

    left_start, left_end = left
    right_start, right_end = right
    return left_start <= right_end and right_start <= left_end


def _ensure_features_for_version(version: str, start: str, end: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ensure_features.py"),
            "--start",
            start,
            "--end",
            end,
            "--version",
            version,
            "--fix",
        ],
        check=True,
        cwd=str(ROOT),
    )


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
    return [_slot_diagnostics(slot) for slot in VERSION_COMPARE_SLOTS]


def _slot_needs_generation(slot: dict, *, force: bool = False) -> bool:
    key = slot["key"]
    if key in {"v3", "main"}:
        return False

    if force:
        return True

    return _slot_window_rows(key) is None


def _generate_missing_predictions(*, force: bool = False) -> dict:
    """Run predict for compare-window CSVs. V3 is manual; Main shares V2's file."""
    generated: list[str] = []
    skipped_models: list[str] = []
    skipped_ready: list[str] = []
    seen_paths: set[str] = set()

    for slot in VERSION_COMPARE_SLOTS:
        key = slot["key"]
        if key in {"v3", "main"}:
            continue

        path = compare_predictions_path(key)
        path_key = str(path.resolve())
        if path_key in seen_paths:
            continue

        model_version = slot["model_version"]
        if not version_has_models(model_version):
            skipped_models.append(slot["label"])
            continue

        if not force and not _slot_needs_generation(slot):
            skipped_ready.append(slot["label"])
            continue

        feature_end = _predict_feature_end()
        _ensure_features_for_version(model_version, PREDICT_START, feature_end)

        generate_predictions(
            PREDICT_START,
            feature_end,
            version=model_version,
        )
        generated.append(slot["label"])
        seen_paths.add(path_key)

    return {
        "generated": generated,
        "skipped_models": skipped_models,
        "skipped_ready": skipped_ready,
    }


def _format_generate_feedback(result: dict) -> tuple[str, str]:
    generated = result["generated"]
    skipped_models = result["skipped_models"]
    skipped_ready = result["skipped_ready"]

    if generated:
        return (
            "success",
            "Generated compare-window predictions for: "
            + ", ".join(generated),
        )

    parts: list[str] = []
    if skipped_ready:
        parts.append(
            "Compare-window predictions already loaded for: "
            + ", ".join(skipped_ready)
            + ". Enable **Force regenerate** to overwrite."
        )
    if skipped_models:
        parts.append(
            "Models missing for: "
            + ", ".join(skipped_models)
            + " — train models before generating."
        )

    stale = [
        row["slot"]
        for row in _slot_availability()
        if row["status"] == "stale"
    ]
    if stale and not skipped_ready:
        parts.append(
            "Daily pipeline CSVs exist for "
            + ", ".join(stale)
            + f" but have **0 rows** in the compare window "
            f"({_compare_window_label()}). "
            "Click **Generate missing predictions** to build window CSVs."
        )

    if not parts:
        availability = _slot_availability()
        stale = [row["slot"] for row in availability if row["status"] == "stale"]
        missing_models = [
            row["slot"]
            for row in availability
            if row["models"] == "missing"
        ]
        if stale:
            parts.append(
                "Daily pipeline CSVs exist but have "
                f"**0 rows** in the compare window ({_compare_window_label()}). "
                "Enable **Force regenerate V1/V2** or check **Version sources**."
            )
        elif missing_models:
            parts.append(
                "Models missing for: "
                + ", ".join(missing_models)
                + " — train models before generating."
            )
        else:
            parts.append(
                "Nothing to generate — see **Version sources** for details."
            )

    return ("warning", " ".join(parts))


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


def _on_generate_missing_predictions(*, force: bool = False) -> None:
    try:
        result = _generate_missing_predictions(force=force)
        load_version_compare_merged.clear()
        level, message = _format_generate_feedback(result)
        _set_generate_feedback(level, message)
    except subprocess.CalledProcessError as exc:
        load_version_compare_merged.clear()
        _set_generate_feedback(
            "error",
            "Feature rebuild failed during generate. Ensure Statcast raw "
            "data exists and `DISABLE_LIVE_FETCH` is unset if features "
            "need refreshing. "
            f"Details: {exc}",
        )
    except Exception as exc:
        load_version_compare_merged.clear()
        _set_generate_feedback(
            "error",
            f"Failed to generate predictions: {exc}",
        )


@st.cache_data(show_spinner="Loading version comparison…")
def load_version_compare_merged(cache_key: tuple[tuple[str, float], ...]):
    _ = cache_key
    slot_frames = {
        slot["key"]: _slot_frame(slot)
        for slot in VERSION_COMPARE_SLOTS
    }
    return _merge_compare_frames(slot_frames)


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
        f"Feature window: {_compare_window_label()}."
    )

    availability = _slot_availability()
    ready_labels = [
        row["slot"]
        for row in availability
        if row["status"] == "ready"
    ]
    stale_labels = [
        row["slot"]
        for row in availability
        if row["status"] == "stale"
    ]
    missing_labels = [
        row["slot"]
        for row in availability
        if row["status"] in {"missing", "empty"}
    ]

    with st.expander("Version sources", expanded=not ready_labels):
        for slot, row in zip(VERSION_COMPARE_SLOTS, availability):
            if row["status"] == "ready":
                pred_status = (
                    f"ready ({row['window_rows']:,} rows in compare window)"
                )
            elif row["status"] == "stale":
                pred_status = (
                    f"stale — file has {row['raw_rows']:,} rows but "
                    f"**0 in compare window** ({_compare_window_label()})"
                )
            elif row["status"] == "empty":
                pred_status = "empty file"
            else:
                pred_status = "missing"

            date_range = "—"
            if row["date_min"] and row["date_max"]:
                date_range = f"{row['date_min']} → {row['date_max']}"

            st.markdown(
                f"**{slot['label']}** — {slot['description']}  \n"
                f"`{row['path']}` · predictions: **{pred_status}** · "
                f"file dates: **{date_range}** · "
                f"models/{slot['model_version']}/: "
                f"**{row['model_count']}** "
                f"({'ok' if row['models'] == 'yes' else 'missing'})"
            )

        if stale_labels:
            st.warning(
                "**Daily pipeline CSVs** for "
                + ", ".join(stale_labels)
                + f" fall outside the compare window ({_compare_window_label()}). "
                "Use **Generate missing predictions** to build window CSVs "
                "for side-by-side comparison."
            )

        date_ranges = _slot_date_ranges(availability)
        v1_range = date_ranges.get("v1")
        v2_range = date_ranges.get("v2")
        if (
            v1_range
            and v2_range
            and not _dates_overlap(v1_range, v2_range)
        ):
            st.error(
                f"**V1** props are from **{v1_range[0]} → {v1_range[1]}** but "
                f"**V2/Main** are from **{v2_range[0]} → {v2_range[1]}**. "
                "Columns will not fill on the same rows until both versions "
                "score the **same current slate**. Check **Force regenerate "
                "V1/V2**, then **Generate missing predictions** "
                "(unset `DISABLE_LIVE_FETCH` if V1 features must rebuild)."
            )

        v3_row = next(row for row in availability if row["key"] == "v3")
        if missing_labels or v3_row["status"] != "ready":
            st.info(
                "**V3** predictions are not generated in this workspace — copy "
                "`predictions_v2.csv` from the frozen "
                "[mlb-prop-model-v3](../mlb-prop-model-v3/) snapshot to "
                "`data/predictions/predictions_v3.csv` to populate that column. "
                "**V2** and **Main** share `predictions_v2.csv`."
            )

    action_row = st.container(horizontal=True)
    with action_row:
        generate_clicked = st.button(
            "Generate missing predictions",
            type="primary",
            key=_GEN_BUTTON_KEY,
            help=(
                "Run predict for V1 and/or V2 when compare-window CSVs are "
                "missing or stale. V3 is not auto-generated."
            ),
        )
        force_regenerate = st.checkbox(
            "Force regenerate V1/V2",
            key=_FORCE_REGEN_KEY,
            help=(
                "Overwrite existing CSVs and rebuild predictions for the "
                f"compare window ({_compare_window_label()})."
            ),
        )

    if generate_clicked:
        with st.spinner("Running predict for missing versions…"):
            _on_generate_missing_predictions(force=force_regenerate)

    _render_generate_feedback()

    merged_df = load_version_compare_merged(_compare_cache_key())

    if merged_df.empty:
        st.warning(
            "No predictions loaded for the compare window "
            f"({_compare_window_label()}). "
            "Click **Generate missing predictions** above."
        )
        if stale_labels:
            st.caption(
                "Daily pipeline files exist for "
                + ", ".join(stale_labels)
                + " but their game dates fall outside this window."
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

    filter_row = st.container(horizontal=True, vertical_alignment="bottom")
    with filter_row:
        selected_markets = render_market_multiselect(
            merged_df,
            key="version_compare_markets",
            label="Market type",
        )

    compare_df = merged_df.copy()
    if selected_markets:
        compare_df = compare_df[
            compare_df["market"].isin(selected_markets)
        ]

    compare_df = _top_compare_rows(compare_df)

    if compare_df.empty:
        st.info(
            "No props in the compare window for the selected market type. "
            "Clear the filter or pick another market."
        )
        return

    compare_df = _ensure_slot_columns(compare_df)
    display = _format_compare_display(compare_df)
    display_columns = _display_column_order()

    loaded_count = len(ready_labels)
    overlap_rows = int(
        compare_df.apply(_version_overlap_count, axis=1).ge(2).sum()
    )
    st.caption(
        f"Showing **{len(display)}** props ranked by **Main edge** (or max "
        f"|edge| / max Over % across loaded versions). "
        f"**{loaded_count}** of **{len(VERSION_COMPARE_SLOTS)}** "
        f"version columns loaded ({', '.join(ready_labels) or 'none'}). "
        f"**{overlap_rows}** rows have Over/Under % in **2+** versions."
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
