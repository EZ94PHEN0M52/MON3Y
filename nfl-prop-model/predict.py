"""Score live NFL props with trained models + injury gates."""

from __future__ import annotations

import argparse
import re

import joblib
import numpy as np
import pandas as pd

from features_v2 import (
    build_opponent_defense_table,
    load_team_stats,
    v2_feature_column_names,
)
from train import MARKET_STAT
from utils import (
    current_injuries_path,
    current_props_path,
    expected_value,
    american_to_implied_probability,
    normalize_player_name,
    normalize_version,
    player_features_path,
    predictions_path,
    version_models_dir,
)

TEAM_NAME_TO_ABBR = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "los angeles chargers": "LAC",
    "los angeles rams": "LA",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "new york jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
}


def team_to_abbr(name: str | None) -> str | None:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    text = str(name).strip()
    if re.fullmatch(r"[A-Z]{2,3}", text):
        return text
    return TEAM_NAME_TO_ABBR.get(text.lower())


def load_model(market: str, version: str = "v1") -> dict | None:
    path = version_models_dir(version) / f"{market}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def latest_feature_rows(features: pd.DataFrame) -> pd.DataFrame:
    ordered = features.sort_values(["season", "week"])
    return (
        ordered.groupby("player_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def latest_defense_lookup(seasons: list[int]) -> pd.DataFrame:
    team_stats = load_team_stats(seasons)
    defense = build_opponent_defense_table(team_stats)
    if defense.empty:
        return pd.DataFrame()
    cols = ["team"] + [
        c
        for c in v2_feature_column_names()
        if c.startswith("opp_") and c in defense.columns
    ]
    ordered = defense.sort_values(["season", "week"])
    return ordered.groupby("team", as_index=False).tail(1)[cols]


def overlay_slate_opponent(
    feat: pd.Series,
    prop: pd.Series,
    defense_lookup: pd.DataFrame,
) -> pd.Series:
    out = feat.copy()
    if defense_lookup.empty:
        return out

    player_team = str(feat.get("team", "") or "")
    home = team_to_abbr(prop.get("home_team"))
    away = team_to_abbr(prop.get("away_team"))
    if not player_team or not home or not away:
        return out

    if player_team == home:
        opponent = away
        out["is_home"] = 1.0
    elif player_team == away:
        opponent = home
        out["is_home"] = 0.0
    else:
        return out

    out["opponent_team"] = opponent
    hit = defense_lookup.loc[defense_lookup["team"].eq(opponent)]
    if hit.empty:
        return out
    row = hit.iloc[0]
    for col in row.index:
        if col == "team":
            continue
        out[col] = row[col]
    return out


def fuzzy_match_features(
    player_name: str,
    features: pd.DataFrame,
) -> pd.Series | None:
    needle = normalize_player_name(player_name)
    if not needle or features.empty:
        return None

    names = features["player_name"].map(normalize_player_name)
    exact = features.loc[names.eq(needle)]
    if len(exact) == 1:
        return exact.iloc[0]
    if len(exact) > 1:
        return exact.iloc[-1]

    parts = needle.split()
    if len(parts) >= 2:
        last = parts[-1]
        initial = parts[0][0]
        mask = names.map(
            lambda n: bool(n)
            and n.split()[-1] == last
            and n.split()[0][:1] == initial
        )
        hits = features.loc[mask]
        if len(hits) >= 1:
            return hits.iloc[-1]
    return None


def attach_live_injuries(
    row: pd.Series,
    injuries: pd.DataFrame,
    player_name: str,
) -> pd.Series:
    out = row.copy()
    if injuries.empty:
        out["injury_status_norm"] = out.get("injury_status_norm", "active")
        out["is_inactive"] = int(out.get("is_inactive", 0) or 0)
        out["is_questionable"] = int(out.get("is_questionable", 0) or 0)
        out["report_status"] = None
        return out

    needle = normalize_player_name(player_name)
    inj_names = injuries["full_name"].map(normalize_player_name)
    hits = injuries.loc[inj_names.eq(needle)]
    if hits.empty:
        out["report_status"] = None
        return out

    hit = hits.iloc[-1]
    out["report_status"] = hit.get("report_status")
    out["injury_status_norm"] = hit.get(
        "report_status_norm",
        out.get("injury_status_norm", "active"),
    )
    out["is_inactive"] = int(bool(hit.get("is_inactive", False)))
    out["is_questionable"] = int(bool(hit.get("is_questionable", False)))
    if "depth_chart_position" in hit.index:
        out["depth_chart_position"] = hit.get("depth_chart_position")
    return out


def score_prop(
    model_bundle: dict,
    feature_row: pd.Series,
    line: float,
) -> float:
    model = model_bundle["model"]
    cols = model_bundle["feature_columns"]
    target = model_bundle["target_stat"]

    payload = feature_row.copy()
    payload["line"] = float(line)
    season_avg = feature_row.get(f"{target}_season", np.nan)
    payload["line_vs_season_avg"] = float(line) - (
        float(season_avg) if pd.notna(season_avg) else float(line)
    )

    x = pd.DataFrame([{c: payload.get(c, np.nan) for c in cols}])
    x = x.astype(float)
    return float(model.predict_proba(x)[0, 1])


def generate_predictions(version: str = "v1") -> pd.DataFrame:
    version = normalize_version(version)

    props_path = current_props_path()
    feat_path = player_features_path(version)
    if not props_path.exists():
        raise FileNotFoundError(
            f"Missing {props_path}. Run: python fetch_data.py --props"
        )
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Missing {feat_path}. Run: python build_features.py --version {version}"
        )

    props = pd.read_parquet(props_path)
    full_features = pd.read_parquet(feat_path)
    features = latest_feature_rows(full_features)

    seasons = sorted(
        full_features["season"].dropna().astype(int).unique().tolist()
    )
    defense_lookup = (
        latest_defense_lookup(seasons) if version == "v2" else pd.DataFrame()
    )

    injuries = pd.DataFrame()
    inj_path = current_injuries_path()
    if inj_path.exists():
        injuries = pd.read_parquet(inj_path)

    models = {}
    for market in MARKET_STAT:
        bundle = load_model(market, version)
        if bundle is None:
            print(f"WARNING: no model for {market}")
        else:
            models[market] = bundle

    rows = []
    unmatched = 0

    for _, prop in props.iterrows():
        market = prop.get("market")
        if market not in models:
            continue

        player = prop.get("player")
        feat = fuzzy_match_features(player, features)
        if feat is None:
            unmatched += 1
            continue

        feat = attach_live_injuries(feat, injuries, player)
        if version == "v2":
            feat = overlay_slate_opponent(feat, prop, defense_lookup)

        line = prop.get("line")
        if pd.isna(line):
            continue

        model_prob = score_prop(models[market], feat, float(line))
        side = str(prop.get("side", "")).strip().lower()
        odds = prop.get("odds")

        if side == "over":
            side_prob = model_prob
        elif side == "under":
            side_prob = 1.0 - model_prob
        else:
            continue

        market_implied = (
            american_to_implied_probability(odds)
            if pd.notna(odds)
            else np.nan
        )
        edge = (
            side_prob - market_implied
            if pd.notna(market_implied)
            else np.nan
        )
        ev = (
            expected_value(side_prob, odds)
            if pd.notna(odds)
            else np.nan
        )

        inactive = bool(int(feat.get("is_inactive", 0) or 0))
        questionable = bool(int(feat.get("is_questionable", 0) or 0))

        rows.append({
            "player": player,
            "market": market,
            "side": side.title(),
            "line": float(line),
            "odds": odds,
            "bookmaker": prop.get("bookmaker"),
            "home_team": prop.get("home_team"),
            "away_team": prop.get("away_team"),
            "commence_time": prop.get("commence_time"),
            "model_probability": side_prob,
            "model_over_probability": model_prob,
            "market_implied_probability": market_implied,
            "edge": edge,
            "ev": ev,
            "position": feat.get("position"),
            "team": feat.get("team"),
            "opponent_team": feat.get("opponent_team"),
            "injury_status": feat.get("report_status"),
            "injury_status_norm": feat.get("injury_status_norm"),
            "is_inactive": inactive,
            "is_questionable": questionable,
            "exclude_from_top": inactive,
            "passing_yards_l5": feat.get("passing_yards_l5"),
            "rushing_yards_l5": feat.get("rushing_yards_l5"),
            "receptions_l5": feat.get("receptions_l5"),
            "receiving_yards_l5": feat.get("receiving_yards_l5"),
            "player_id": feat.get("player_id"),
            "feature_season": feat.get("season"),
            "feature_week": feat.get("week"),
            "model_version": version,
        })

    if not rows:
        raise RuntimeError(
            "No scored props — check name matching / models / current_props."
        )

    out = pd.DataFrame(rows)
    out = (
        out.sort_values("ev", ascending=False)
        .drop_duplicates(subset=["player", "market", "side"], keep="first")
        .reset_index(drop=True)
    )

    path = predictions_path(version)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    active = out[~out["exclude_from_top"]]
    print(
        f"Saved {len(out):,} prediction rows "
        f"({len(active):,} active / "
        f"{int(out['exclude_from_top'].sum())} inactive-gated) → {path}"
    )
    print(f"Unmatched prop rows skipped: {unmatched:,}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict NFL props.")
    parser.add_argument("--version", default="v2", choices=["v1", "v2"])
    args = parser.parse_args()
    generate_predictions(version=args.version)


if __name__ == "__main__":
    main()
