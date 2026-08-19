import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prop_scoring import (  # noqa: E402
    MODEL_MAP,
    MARKET_STAT_MAP,
    fuzzy_player_match,
    load_model,
    score_prop,
)
from clv import attach_clv  # noqa: E402
from calibration import load_calibrator  # noqa: E402
from training_odds import load_historical_props  # noqa: E402
from odds_aggregation import enrich_predictions  # noqa: E402
from utils import (  # noqa: E402
    batter_features_path,
    pitcher_features_path,
    normalize_version,
    backtest_output_path,
)


def find_covering_feature_path(
    start_date,
    end_date,
    version,
    role,
):
    processed = (
        ROOT / "data" / "processed"
    )

    if version == "v1":
        prefix = (
            "batter_features_"
            if role == "batter"
            else "pitcher_features_"
        )
    else:
        prefix = (
            "batter_features_v2_"
            if role == "batter"
            else "pitcher_features_v2_"
        )

    candidates = []

    for path in processed.glob(
        f"{prefix}*.parquet"
    ):
        stem = path.stem.replace(
            prefix,
            "",
        )

        if "_" not in stem:
            continue

        file_start, file_end = stem.split(
            "_",
            1,
        )

        if (
            file_start <= start_date
            and file_end >= end_date
        ):
            span = (
                pd.to_datetime(file_end)
                - pd.to_datetime(file_start)
            ).days

            candidates.append(
                (span, path)
            )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def load_outcome_tables(
    start_date,
    end_date,
    version="v2",
):
    batter_path = batter_features_path(
        start_date,
        end_date,
        version,
    )

    pitcher_path = pitcher_features_path(
        start_date,
        end_date,
        version,
    )

    if not batter_path.exists():
        batter_path = find_covering_feature_path(
            start_date,
            end_date,
            version,
            "batter",
        )

    if not pitcher_path.exists():
        pitcher_path = find_covering_feature_path(
            start_date,
            end_date,
            version,
            "pitcher",
        )

    if batter_path is None or not batter_path.exists():
        raise FileNotFoundError(
            "No batter feature parquet "
            f"covers {start_date} → "
            f"{end_date}.\n"
            "Run build_features.py for "
            "a range that includes these "
            "dates."
        )

    if pitcher_path is None or not pitcher_path.exists():
        raise FileNotFoundError(
            "No pitcher feature parquet "
            f"covers {start_date} → "
            f"{end_date}.\n"
            "Run build_features.py for "
            "a range that includes these "
            "dates."
        )

    batters = pd.read_parquet(
        batter_path
    )

    pitchers = pd.read_parquet(
        pitcher_path
    )

    batters = batters[
        batters["game_date"].ge(
            start_date
        )
        & batters["game_date"].le(
            end_date
        )
    ].copy()

    pitchers = pitchers[
        pitchers["game_date"].ge(
            start_date
        )
        & pitchers["game_date"].le(
            end_date
        )
    ].copy()

    batters["game_date"] = (
        pd.to_datetime(
            batters["game_date"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    pitchers["game_date"] = (
        pd.to_datetime(
            pitchers["game_date"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    batters = batters.drop_duplicates(
        subset=["game_date", "player_name"],
        keep="first",
    )

    pitchers = pitchers.drop_duplicates(
        subset=["game_date", "player_name"],
        keep="first",
    )

    batter_lookup = (
        batters
        .set_index(
            ["game_date", "player_name"]
        )
        .sort_index()
    )

    pitcher_lookup = (
        pitchers
        .set_index(
            ["game_date", "player_name"]
        )
        .sort_index()
    )

    batter_names_by_date = (
        batters
        .groupby("game_date")[
            "player_name"
        ]
        .apply(
            lambda s: s.dropna().drop_duplicates()
        )
    )

    pitcher_names_by_date = (
        pitchers
        .groupby("game_date")[
            "player_name"
        ]
        .apply(
            lambda s: s.dropna().drop_duplicates()
        )
    )

    return (
        batter_lookup,
        pitcher_lookup,
        batter_names_by_date,
        pitcher_names_by_date,
    )


def _lookup_feature_row(lookup, key):
    """Return a single feature row Series for a MultiIndex key."""
    try:
        row = lookup.loc[key]
    except KeyError:
        return None

    if isinstance(row, pd.DataFrame):
        if row.empty:
            return None
        return row.iloc[0]

    return row


def actual_over_hit(
    actual_stat,
    line,
):
    if pd.isna(actual_stat) or pd.isna(line):
        return np.nan

    return int(float(actual_stat) > float(line))


def bet_won(
    side,
    actual_stat,
    line,
):
    over_hit = actual_over_hit(
        actual_stat,
        line,
    )

    if pd.isna(over_hit):
        return np.nan

    side = str(side).strip().lower()

    if side == "over":
        return int(over_hit == 1)

    if side == "under":
        return int(over_hit == 0)

    return np.nan


def flat_bet_profit(
    won,
    american_odds,
):
    if pd.isna(won):
        return np.nan

    odds = float(american_odds)

    if odds > 0:
        payout = odds / 100
    else:
        payout = 100 / abs(odds)

    if won:
        return payout

    return -1.0


def run_backtest(
    start_date,
    end_date,
    version="v2",
    min_edge=0.0,
    min_ev=0.0,
    market_filter=None,
):
    version = normalize_version(version)

    props = load_historical_props(
        start_date,
        end_date,
        market_filter=market_filter,
    )

    if props.empty:
        print(
            "No historical props found "
            f"for {start_date} → {end_date}."
        )
        print(
            "Run fetch_historical_odds.py "
            "first."
        )
        return pd.DataFrame()

    (
        batter_lookup,
        pitcher_lookup,
        batter_names_by_date,
        pitcher_names_by_date,
    ) = load_outcome_tables(
        start_date,
        end_date,
        version,
    )

    models = {}
    calibrators = {}
    missing_models = set()

    results = []
    unmatched_players = 0
    missing_features = 0
    missing_stats = 0

    props = props.copy()
    props["game_date"] = (
        pd.to_datetime(
            props["commence_time"],
            utc=True,
        )
        .dt.strftime("%Y-%m-%d")
    )

    for _, prop in props.iterrows():
        market = prop["market"]

        model_filename = MODEL_MAP.get(
            market
        )

        stat_col = MARKET_STAT_MAP.get(
            market
        )

        if not model_filename or not stat_col:
            continue

        if market not in models:
            package = load_model(
                model_filename,
                version,
            )

            if package is None:
                missing_models.add(market)
                continue

            models[market] = package

        if market not in calibrators:
            calibrators[market] = load_calibrator(
                market,
                version,
            )

        package = models[market]
        game_date = prop["game_date"]
        player_name = prop["player"]

        if market.startswith("batter_"):
            if game_date not in batter_names_by_date.index:
                missing_features += 1
                continue

            candidates = (
                batter_names_by_date.loc[
                    game_date
                ]
            )

            match = fuzzy_player_match(
                player_name,
                candidates,
            )

            if match is None:
                unmatched_players += 1
                continue

            key = (game_date, match)
            feature_row = _lookup_feature_row(
                batter_lookup,
                key,
            )

            if feature_row is None:
                missing_features += 1
                continue

        else:
            if game_date not in pitcher_names_by_date.index:
                missing_features += 1
                continue

            candidates = (
                pitcher_names_by_date.loc[
                    game_date
                ]
            )

            match = fuzzy_player_match(
                player_name,
                candidates,
            )

            if match is None:
                unmatched_players += 1
                continue

            key = (game_date, match)
            feature_row = _lookup_feature_row(
                pitcher_lookup,
                key,
            )

            if feature_row is None:
                missing_features += 1
                continue

        scores = score_prop(
            prop,
            feature_row,
            package,
            version=version,
            calibrator=calibrators.get(market),
        )

        actual_stat = feature_row.get(
            stat_col,
            np.nan,
        )

        if pd.isna(actual_stat):
            missing_stats += 1
            continue

        actual_over = actual_over_hit(
            actual_stat,
            prop["line"],
        )

        won = bet_won(
            prop["side"],
            actual_stat,
            prop["line"],
        )

        profit = flat_bet_profit(
            won,
            prop["odds"],
        )

        results.append({
            "snapshot_date": prop.get(
                "snapshot_date"
            ),
            "event_id": prop.get(
                "event_id"
            ),
            "game_date": game_date,
            "player": player_name,
            "matched_player": match,
            "market": market,
            "bookmaker": prop["bookmaker"],
            "bookmaker_key": prop.get(
                "bookmaker_key"
            ),
            "side": prop["side"],
            "line": prop["line"],
            "odds": prop["odds"],
            "actual_stat": actual_stat,
            "actual_over": actual_over,
            "won": won,
            "profit": profit,
            **scores,
        })

    detail = enrich_predictions(
        pd.DataFrame(results)
    )

    detail = attach_clv(
        detail,
        props,
    )

    print()
    print("=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)
    print(
        f"Props loaded: {len(props):,}"
    )
    print(
        f"Scored rows: {len(detail):,}"
    )
    print(
        f"Unmatched players: "
        f"{unmatched_players:,}"
    )
    print(
        f"Missing feature rows: "
        f"{missing_features:,}"
    )
    print(
        f"Missing stat values: "
        f"{missing_stats:,}"
    )

    if missing_models:
        print(
            "Missing models: "
            + ", ".join(
                sorted(missing_models)
            )
        )

    if detail.empty:
        print()
        print("No scored rows to summarize.")
        return detail

    bets = detail[
        detail["edge"].ge(min_edge)
        & detail["ev"].ge(min_ev)
    ].copy()

    print()
    print(
        f"Flat bets (edge ≥ {min_edge:.3f}, "
        f"EV ≥ {min_ev:.3f}): "
        f"{len(bets):,}"
    )

    if len(bets) > 0:
        total_profit = bets["profit"].sum()
        roi = total_profit / len(bets)
        win_rate = bets["won"].mean()

        print(
            f"Win rate: {win_rate:.1%}"
        )
        print(
            f"ROI (flat $1): {roi:.2%}"
        )
        print(
            f"Total profit: "
            f"{total_profit:.2f} units"
        )

    clv_rows = detail["clv"].dropna()

    if len(clv_rows) > 0:
        print()
        print(
            f"Avg CLV (beat close): "
            f"{clv_rows.mean():.4f} "
            f"({len(clv_rows):,} rows with "
            "multiple snapshots)"
        )

        model_clv = detail[
            "model_clv"
        ].dropna()

        if len(model_clv) > 0:
            print(
                f"Avg model CLV vs close: "
                f"{model_clv.mean():.4f}"
            )

    print()
    print("Per-market metrics:")
    print("-" * 80)

    summary_rows = []

    for market, group in detail.groupby(
        "market"
    ):
        brier = (
            (
                group["over_probability"]
                - group["actual_over"]
            ) ** 2
        ).mean()

        raw_brier = np.nan

        if "raw_over_probability" in group.columns:
            raw_brier = (
                (
                    group["raw_over_probability"]
                    - group["actual_over"]
                ) ** 2
            ).mean()

        market_bets = bets[
            bets["market"].eq(market)
        ]

        market_roi = np.nan
        market_win = np.nan

        if len(market_bets) > 0:
            market_roi = (
                market_bets["profit"].sum()
                / len(market_bets)
            )
            market_win = (
                market_bets["won"].mean()
            )

        summary_rows.append({
            "market": market,
            "rows": len(group),
            "avg_edge": group["edge"].mean(),
            "brier_over": brier,
            "brier_raw_over": raw_brier,
            "avg_clv": group["clv"].mean(),
            "bets": len(market_bets),
            "win_rate": market_win,
            "roi": market_roi,
        })

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        "market"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    output = backtest_output_path(
        start_date,
        end_date,
    )

    detail.to_csv(
        output,
        index=False,
    )

    print()
    print("Saved:", output)

    return detail


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Backtest model edges against "
            "historical sportsbook props"
        ),
    )

    parser.add_argument(
        "--start",
        required=True,
        help="Start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end",
        required=True,
        help="End date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--version",
        default="v2",
        choices=["v1", "v2"],
        help="Model version (default: v2)",
    )

    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.0,
        help=(
            "Minimum model edge for flat "
            "bet ROI filter (default: 0)"
        ),
    )

    parser.add_argument(
        "--min-ev",
        type=float,
        default=0.0,
        help=(
            "Minimum EV for flat bet ROI "
            "filter (default: 0)"
        ),
    )

    parser.add_argument(
        "--market",
        default=None,
        help="Optional single market filter",
    )

    args = parser.parse_args()

    run_backtest(
        args.start,
        args.end,
        version=args.version,
        min_edge=args.min_edge,
        min_ev=args.min_ev,
        market_filter=args.market,
    )


if __name__ == "__main__":
    main()
