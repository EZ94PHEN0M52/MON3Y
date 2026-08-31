"""
Batter Score model
===================
A composite 0-100 rating for a batter's upcoming game, blending:
  1. Season baseline        (H + TB + BB per game)              - 20%
  2. Recent form             L5/L10 blend of the same stat       - 30%
  3. Matchup grade            wOBA-vs-pitch (35%) + AVG-vs-pitch (65%),
                               usage-weighted across the opposing
                               pitcher's arsenal                  - 35%
  4. Pitcher recent form      opposing starter's ERA, last 5      - 15%

Phase A uses season baseline + recent form only; matchup is gated off
until Phase D. Phase B adds opposing SP ERA (L5) and optional H2H blend
into pitcher form when the starter is known; weights renormalize among
active components. Phase D enables usage-weighted pitch-type matchup when
Statcast arsenal data is available for the opposing SP.

All weights and thresholds are configurable via the dataclasses below -
nothing is hardcoded in the scoring functions themselves.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Minimum PA vs a specific SP before H2H stats are blended (Phase B).
MIN_PA_H2H = 10

# Lower bar when the user supplies career H2H H/AB manually (pre-2024 history).
MIN_PA_H2H_MANUAL = 3

# Share of pitcher-form index from H2H when PA threshold is met.
H2H_PITCHER_FORM_BLEND = 0.30

# Stronger blend when the user supplies career H2H H/AB (manual calculator).
H2H_PITCHER_FORM_BLEND_MANUAL = 0.55

# Optional soft fallback when SP is TBD: team opp_team_earned_runs proxy at
# reduced weight (disabled by default — see batter_score_data.py).
USE_TEAM_PITCHING_PROXY = False
TEAM_PITCHING_PROXY_WEIGHT = 0.05


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Weights:
    season_baseline: float = 0.30
    recent_form: float = 0.25
    matchup_grade: float = 0.30
    pitcher_form: float = 0.15

    def validate(self):
        total = (
            self.season_baseline
            + self.recent_form
            + self.matchup_grade
            + self.pitcher_form
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Component weights must sum to 1.0, got {total:.4f}"
            )


WEIGHTS_V1 = Weights(
    season_baseline=0.30,
    recent_form=0.25,
    matchup_grade=0.30,
    pitcher_form=0.15,
)

WEIGHTS_V2 = Weights(
    season_baseline=0.20,
    recent_form=0.30,
    matchup_grade=0.35,
    pitcher_form=0.15,
)


@dataclass
class ComponentGates:
    """Toggle which composite components participate in the score."""

    season_baseline: bool = True
    recent_form: bool = True
    matchup_grade: bool = False
    pitcher_form: bool = False

    def as_dict(self) -> Dict[str, bool]:
        return {
            "season_baseline": self.season_baseline,
            "recent_form": self.recent_form,
            "matchup_grade": self.matchup_grade,
            "pitcher_form": self.pitcher_form,
        }


# Phase A: form-only (matchup + pitcher gated off).
PHASE_A_GATES = ComponentGates(
    season_baseline=True,
    recent_form=True,
    matchup_grade=False,
    pitcher_form=False,
)

# Phase B: season + form + pitcher form; matchup gated until Phase D.
PHASE_B_GATES = ComponentGates(
    season_baseline=True,
    recent_form=True,
    matchup_grade=False,
    pitcher_form=True,
)

# Phase D: full composite when SP + pitch-type arsenal are available.
PHASE_D_GATES = ComponentGates(
    season_baseline=True,
    recent_form=True,
    matchup_grade=True,
    pitcher_form=True,
)


@dataclass
class RecentFormWeights:
    l5: float = 0.70
    l10: float = 0.30

    def validate(self):
        total = self.l5 + self.l10
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"L5/L10 weights must sum to 1.0, got {total:.4f}"
            )


@dataclass
class MatchupWeights:
    woba: float = 0.35
    avg: float = 0.65

    def validate(self):
        total = self.woba + self.avg
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"wOBA/AVG weights must sum to 1.0, got {total:.4f}"
            )


# Grade thresholds: list of (min_value, letter, points), checked highest-first.
WOBA_THRESHOLDS = [
    (0.410, "A", 4),
    (0.350, "B", 3),
    (0.300, "C", 2),
    (0.250, "D", 1),
    (0.000, "F", 0),
]

AVG_THRESHOLDS = [
    (0.284, "A", 4),
    (0.248, "B", 3),
    (0.230, "C", 2),
    (0.200, "D", 1),
    (0.000, "F", 0),
]

# ERA thresholds are inverted: lower ERA = better grade, so this is a max-ERA table.
ERA_THRESHOLDS = [
    (2.50, "A", 4),
    (3.50, "B", 3),
    (4.50, "C", 2),
    (5.50, "D", 1),
    (float("inf"), "F", 0),
]

# FIP thresholds (Batter Score v2): lower FIP = better for the batter.
FIP_THRESHOLDS = [
    (3.00, "A", 4),   # Excellent / Ace
    (3.50, "B", 3),   # fills 3.00–3.50
    (4.00, "B", 3),   # Good / above average
    (4.50, "C", 2),   # Average
    (5.00, "D", 1),   # fills 4.50–5.00
    (float("inf"), "F", 0),  # Below average
]

MAX_GRADE_POINTS = 4  # A = 4 points, used to normalize grade blends to a 0-100 index


def grade_min_threshold(value: float, thresholds) -> tuple:
    """Return (letter, points) for a stat where higher is better (wOBA, AVG)."""
    for min_val, letter, points in thresholds:
        if value >= min_val:
            return letter, points
    return thresholds[-1][1], thresholds[-1][2]


def grade_max_threshold(value: float, thresholds) -> tuple:
    """Return (letter, points) for a stat where lower is better (ERA)."""
    for max_val, letter, points in thresholds:
        if value <= max_val:
            return letter, points
    return thresholds[-1][1], thresholds[-1][2]


def renormalize_weights(
    weights: Weights,
    gates: ComponentGates,
) -> Dict[str, float]:
    """
    Return active component weights scaled to sum to 1.0.

    Excluded (gated-off) components are omitted entirely.
    """
    gate_map = gates.as_dict()
    active = {}

    for key, enabled in gate_map.items():
        if enabled:
            active[key] = getattr(weights, key)

    total = sum(active.values())
    if total <= 0:
        raise ValueError("At least one component must be active")

    return {
        key: value / total
        for key, value in active.items()
    }


def partial_score_label(
    gates: ComponentGates,
    *,
    sp_tbd: bool = False,
    team_proxy: bool = False,
) -> Optional[str]:
    """Human-readable label when some components are gated off."""
    gate_map = gates.as_dict()
    active = [key for key, enabled in gate_map.items() if enabled]
    inactive = [key for key, enabled in gate_map.items() if not enabled]

    if not inactive:
        return "Full"

    if sp_tbd and active == ["season_baseline", "recent_form"]:
        return "Partial · SP TBD"

    if team_proxy and "pitcher_form" in active:
        return "Partial · SP TBD (team proxy)"

    if active == ["season_baseline", "recent_form"]:
        return "Form only"

    if inactive == ["matchup_grade", "pitcher_form"]:
        return "Form only"

    if inactive == ["matchup_grade"]:
        return "Partial"

    return "Partial"


# ---------------------------------------------------------------------------
# Data inputs
# ---------------------------------------------------------------------------

@dataclass
class GameLine:
    date: str
    opponent: str
    hits: int
    total_bases: int
    walks: int

    @property
    def raw_points(self) -> float:
        return self.hits + self.total_bases + self.walks


@dataclass
class PitchTypeMatchup:
    pitch_type: str
    usage_pct: float          # opposing pitcher's usage rate for this pitch, 0-1
    batter_woba: float        # batter's wOBA against this pitch type
    batter_avg: float         # batter's batting average against this pitch type


@dataclass
class BatterInputs:
    name: str
    season_avg_raw_points: float          # full-season per-game avg of H+TB+BB
    game_log: List[GameLine]              # most recent games, chronological desc
    opponent_pitcher_arsenal: List[PitchTypeMatchup] = field(default_factory=list)
    opponent_pitcher_arsenal_v2: List[PitchTypeMatchup] = field(
        default_factory=list,
    )
    opponent_pitcher_era_l5: Optional[float] = None
    opponent_pitcher_fip_l5: Optional[float] = None
    opposing_sp_name: Optional[str] = None
    h2h_pa: Optional[int] = None
    h2h_avg_raw_points: Optional[float] = None
    h2h_hits: Optional[int] = None
    h2h_ab: Optional[int] = None
    h2h_manual_override: bool = False
    team_opp_earned_runs_proxy: Optional[float] = None
    max_raw_points_for_100: float = 6.0   # scaling benchmark for the 0-100 index


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------

def season_baseline_index(batter: BatterInputs) -> float:
    return min(
        100.0,
        batter.season_avg_raw_points
        / batter.max_raw_points_for_100
        * 100,
    )


def recent_form_index(
    batter: BatterInputs,
    rf_weights: RecentFormWeights,
) -> float:
    games = batter.game_log
    if len(games) < 10:
        raise ValueError(
            "Need at least 10 games in game_log for an L5/L10 blend"
        )
    l5 = sum(g.raw_points for g in games[:5]) / 5
    l10 = sum(g.raw_points for g in games[:10]) / 10
    blended = rf_weights.l5 * l5 + rf_weights.l10 * l10
    return min(
        100.0,
        blended / batter.max_raw_points_for_100 * 100,
    )


def matchup_grade_index(
    batter: BatterInputs,
    m_weights: MatchupWeights,
) -> float:
    arsenal = batter.opponent_pitcher_arsenal
    usage_total = sum(p.usage_pct for p in arsenal)
    if abs(usage_total - 1.0) > 0.01:
        raise ValueError(
            f"Pitch usage percentages should sum to ~1.0, got {usage_total:.3f}"
        )

    weighted_points = 0.0
    for pitch in arsenal:
        _, woba_pts = grade_min_threshold(
            pitch.batter_woba,
            WOBA_THRESHOLDS,
        )
        _, avg_pts = grade_min_threshold(
            pitch.batter_avg,
            AVG_THRESHOLDS,
        )
        blended_pts = (
            m_weights.woba * woba_pts
            + m_weights.avg * avg_pts
        )
        weighted_points += pitch.usage_pct * blended_pts

    return weighted_points / MAX_GRADE_POINTS * 100


def _h2h_form_index(batter: BatterInputs) -> Optional[float]:
    min_pa = MIN_PA_H2H_MANUAL if batter.h2h_manual_override else MIN_PA_H2H
    if (
        batter.h2h_pa is None
        or batter.h2h_pa < min_pa
        or batter.h2h_avg_raw_points is None
    ):
        return None

    return min(
        100.0,
        batter.h2h_avg_raw_points
        / batter.max_raw_points_for_100
        * 100,
    )


def _team_pitching_proxy_index(batter: BatterInputs) -> Optional[float]:
    """
    Soft fallback when SP is unknown: higher opp-team runs allowed → easier
    environment for the batter (not a substitute for SP ERA L5).
    """
    proxy = batter.team_opp_earned_runs_proxy
    if proxy is None:
        return None

    try:
        proxy_value = float(proxy)
    except (TypeError, ValueError):
        return None

    if proxy_value != proxy_value:
        return None

    return min(
        100.0,
        proxy_value / batter.max_raw_points_for_100 * 100,
    )


def pitcher_form_index(
    batter: BatterInputs,
    *,
    h2h_blend: float = H2H_PITCHER_FORM_BLEND,
    use_team_proxy: bool = False,
    use_fip: bool = False,
) -> float:
    if use_team_proxy:
        proxy_index = _team_pitching_proxy_index(batter)
        if proxy_index is not None:
            return proxy_index
        raise ValueError(
            "team_opp_earned_runs_proxy is required for team proxy pitcher_form"
        )

    if use_fip:
        if batter.opponent_pitcher_fip_l5 is None:
            raise ValueError(
                "opponent_pitcher_fip_l5 is required for FIP pitcher_form"
            )
        _, points = grade_max_threshold(
            batter.opponent_pitcher_fip_l5,
            FIP_THRESHOLDS,
        )
        pitcher_metric_index = points / MAX_GRADE_POINTS * 100
    else:
        if batter.opponent_pitcher_era_l5 is None:
            raise ValueError(
                "opponent_pitcher_era_l5 is required for pitcher_form"
            )

        _, points = grade_max_threshold(
            batter.opponent_pitcher_era_l5,
            ERA_THRESHOLDS,
        )
        pitcher_metric_index = points / MAX_GRADE_POINTS * 100

    h2h_index = _h2h_form_index(batter)
    if h2h_index is None:
        return pitcher_metric_index

    effective_blend = (
        H2H_PITCHER_FORM_BLEND_MANUAL
        if batter.h2h_manual_override
        else h2h_blend
    )
    blend = max(0.0, min(1.0, effective_blend))
    return (1.0 - blend) * pitcher_metric_index + blend * h2h_index


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

@dataclass
class BatterScoreResult:
    batter_name: str
    season_baseline: float
    recent_form: float
    matchup_grade: Optional[float]
    pitcher_form: Optional[float]
    batter_score: float
    is_partial: bool = False
    partial_label: Optional[str] = None
    active_weights: Dict[str, float] = field(default_factory=dict)
    gated_components: List[str] = field(default_factory=list)
    opposing_sp_name: Optional[str] = None
    opposing_sp_era_l5: Optional[float] = None
    opposing_sp_fip_l5: Optional[float] = None
    h2h_pa: Optional[int] = None
    h2h_avg_raw_points: Optional[float] = None
    h2h_hits: Optional[int] = None
    h2h_ab: Optional[int] = None
    sp_tbd: bool = False
    team_proxy_used: bool = False

    def __str__(self):
        lines = [
            f"Batter score for {self.batter_name}: {self.batter_score:.1f}/100",
            f"  Season baseline   : {self.season_baseline:.1f}",
            f"  Recent form       : {self.recent_form:.1f}",
        ]
        if self.matchup_grade is not None:
            lines.append(
                f"  Matchup grade     : {self.matchup_grade:.1f}"
            )
        else:
            lines.append("  Matchup grade     : — (gated)")
        if self.pitcher_form is not None:
            lines.append(
                f"  Pitcher form      : {self.pitcher_form:.1f}"
            )
        else:
            lines.append("  Pitcher form      : — (gated)")
        if self.opposing_sp_name:
            lines.append(
                f"  Opposing SP       : {self.opposing_sp_name}"
            )
        if self.opposing_sp_era_l5 is not None:
            lines.append(
                f"  SP ERA (L5)       : {self.opposing_sp_era_l5:.2f}"
            )
        if self.opposing_sp_fip_l5 is not None:
            lines.append(
                f"  SP FIP (L5)       : {self.opposing_sp_fip_l5:.2f}"
            )
        if self.h2h_pa is not None and self.h2h_pa > 0:
            lines.append(
                f"  H2H vs SP         : {self.h2h_pa} PA"
            )
        if self.partial_label:
            lines.append(f"  Label             : {self.partial_label}")
        return "\n".join(lines)


def compute_batter_score(
    batter: BatterInputs,
    weights: Weights = None,
    rf_weights: RecentFormWeights = None,
    m_weights: MatchupWeights = None,
    gates: ComponentGates = None,
    *,
    sp_tbd: bool = False,
    team_proxy: bool = False,
    pitcher_form_use_fip: bool = False,
) -> BatterScoreResult:
    weights = weights or Weights()
    rf_weights = rf_weights or RecentFormWeights()
    m_weights = m_weights or MatchupWeights()
    gates = gates or ComponentGates(
        season_baseline=True,
        recent_form=True,
        matchup_grade=True,
        pitcher_form=True,
    )
    weights.validate()
    rf_weights.validate()
    m_weights.validate()

    return _compute_with_gates(
        batter,
        weights,
        rf_weights,
        m_weights,
        gates,
        sp_tbd=sp_tbd,
        team_proxy=team_proxy,
        pitcher_form_use_fip=pitcher_form_use_fip,
    )


def compute_batter_score_partial(
    batter: BatterInputs,
    weights: Weights = None,
    rf_weights: RecentFormWeights = None,
    gates: ComponentGates = None,
    *,
    sp_tbd: bool = False,
    pitcher_form_use_fip: bool = False,
) -> BatterScoreResult:
    """
    Phase A entry point: season baseline + recent form only.

    Matchup and pitcher components are gated off; active weights are
    renormalized to 1.0.
    """
    gates = gates or PHASE_A_GATES
    return compute_batter_score(
        batter,
        weights=weights,
        rf_weights=rf_weights,
        gates=gates,
        sp_tbd=sp_tbd,
        pitcher_form_use_fip=pitcher_form_use_fip,
    )


def compute_batter_score_phase_b(
    batter: BatterInputs,
    weights: Weights = None,
    rf_weights: RecentFormWeights = None,
    gates: ComponentGates = None,
    *,
    sp_tbd: bool = False,
    team_proxy: bool = False,
    proxy_weights: Weights = None,
    pitcher_form_use_fip: bool = False,
) -> BatterScoreResult:
    """
    Phase B entry point: season + form + pitcher form (ERA/FIP L5 + optional H2H).

    Matchup grade remains gated until Phase D arsenal data is available.
    """
    gates = gates or PHASE_B_GATES
    score_weights = weights or Weights()

    if team_proxy:
        score_weights = proxy_weights or Weights(
            season_baseline=score_weights.season_baseline,
            recent_form=score_weights.recent_form,
            matchup_grade=score_weights.matchup_grade,
            pitcher_form=TEAM_PITCHING_PROXY_WEIGHT,
        )

    return compute_batter_score(
        batter,
        weights=score_weights,
        rf_weights=rf_weights,
        gates=gates,
        sp_tbd=sp_tbd,
        team_proxy=team_proxy,
        pitcher_form_use_fip=pitcher_form_use_fip,
    )


def compute_batter_score_phase_d(
    batter: BatterInputs,
    weights: Weights = None,
    rf_weights: RecentFormWeights = None,
    m_weights: MatchupWeights = None,
    gates: ComponentGates = None,
    *,
    sp_tbd: bool = False,
    pitcher_form_use_fip: bool = False,
) -> BatterScoreResult:
    """
    Phase D entry point: full composite with usage-weighted pitch-type matchup.
    """
    gates = gates or PHASE_D_GATES

    return compute_batter_score(
        batter,
        weights=weights,
        rf_weights=rf_weights,
        m_weights=m_weights,
        gates=gates,
        sp_tbd=sp_tbd,
        pitcher_form_use_fip=pitcher_form_use_fip,
    )


def _compute_with_gates(
    batter: BatterInputs,
    weights: Weights,
    rf_weights: RecentFormWeights,
    m_weights: MatchupWeights,
    gates: ComponentGates,
    *,
    sp_tbd: bool = False,
    team_proxy: bool = False,
    pitcher_form_use_fip: bool = False,
) -> BatterScoreResult:
    gate_map = gates.as_dict()
    active_weight_map = renormalize_weights(weights, gates)
    component_values = {}
    gated_off = []

    if gate_map["season_baseline"]:
        component_values["season_baseline"] = season_baseline_index(
            batter
        )
    else:
        gated_off.append("season_baseline")

    if gate_map["recent_form"]:
        component_values["recent_form"] = recent_form_index(
            batter,
            rf_weights,
        )
    else:
        gated_off.append("recent_form")

    matchup_grade = None
    if gate_map["matchup_grade"]:
        component_values["matchup_grade"] = matchup_grade_index(
            batter,
            m_weights,
        )
        matchup_grade = component_values["matchup_grade"]
    else:
        gated_off.append("matchup_grade")

    pitcher_form = None
    if gate_map["pitcher_form"]:
        component_values["pitcher_form"] = pitcher_form_index(
            batter,
            use_team_proxy=team_proxy,
            use_fip=pitcher_form_use_fip,
        )
        pitcher_form = component_values["pitcher_form"]
    else:
        gated_off.append("pitcher_form")

    score = sum(
        active_weight_map[key] * component_values[key]
        for key in component_values
    )

    is_partial = len(gated_off) > 0
    label = partial_score_label(
        gates,
        sp_tbd=sp_tbd,
        team_proxy=team_proxy,
    )
    if not is_partial and label is None:
        label = "Full"

    return BatterScoreResult(
        batter_name=batter.name,
        season_baseline=component_values.get("season_baseline", 0.0),
        recent_form=component_values.get("recent_form", 0.0),
        matchup_grade=matchup_grade,
        pitcher_form=pitcher_form,
        batter_score=score,
        is_partial=is_partial,
        partial_label=label,
        active_weights=active_weight_map,
        gated_components=gated_off,
        opposing_sp_name=batter.opposing_sp_name,
        opposing_sp_era_l5=batter.opponent_pitcher_era_l5,
        opposing_sp_fip_l5=batter.opponent_pitcher_fip_l5,
        h2h_pa=batter.h2h_pa,
        h2h_avg_raw_points=batter.h2h_avg_raw_points,
        h2h_hits=batter.h2h_hits,
        h2h_ab=batter.h2h_ab,
        sp_tbd=sp_tbd,
        team_proxy_used=team_proxy,
    )


if __name__ == "__main__":
    sample_games = [
        GameLine("2026-08-18", "SEA", 2, 3, 1),
        GameLine("2026-08-17", "SEA", 1, 1, 0),
        GameLine("2026-08-16", "SEA", 3, 5, 1),
        GameLine("2026-08-15", "HOU", 0, 0, 1),
        GameLine("2026-08-14", "HOU", 1, 2, 0),
        GameLine("2026-08-13", "HOU", 2, 2, 2),
        GameLine("2026-08-11", "TEX", 1, 1, 0),
        GameLine("2026-08-10", "TEX", 0, 0, 0),
        GameLine("2026-08-09", "TEX", 2, 4, 1),
        GameLine("2026-08-08", "LAA", 1, 1, 1),
    ]

    batter = BatterInputs(
        name="Sample Batter",
        season_avg_raw_points=3.8,
        game_log=sample_games,
    )

    result = compute_batter_score_partial(batter)
    print(result)
