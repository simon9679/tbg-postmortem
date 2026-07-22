"""
Intervention Engine v1.0

Selects optimal response strategy via mathematical scoring simulation.
Pure Python — no IO, no LLM. Fast (<1ms).

8 strategies, 6 scoring metrics, safety gate per mode.
Sensitivity profile persisted in user_sensitivity table.

Output: InterventionDirective — tells the LLM HOW to respond,
        not WHAT to say.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

STRATEGIES = [
    "validate",         # affirm feelings, show understanding without judgment
    "structure",        # give clarity, break into manageable steps
    "reflect",          # mirror what you hear, ask one deep question
    "gentle_challenge", # softly question a limiting belief or assumption
    "reframe",          # offer alternative perspective on the situation
    "ground",           # bring to present moment, reduce overwhelm
    "amplify_agency",   # reinforce sense of choice, capability, control
    "neutral_explore",  # open-ended, non-directive exploration
]

STRATEGY_DESCRIPTIONS = {
    "validate":         "Affirm the person's feelings and experience without judgment",
    "structure":        "Provide clarity and structure, break the problem into manageable steps",
    "reflect":          "Mirror what you hear, ask one deep question to deepen self-awareness",
    "gentle_challenge": "Softly question a limiting assumption or belief",
    "reframe":          "Offer a genuinely different perspective on the situation",
    "ground":           "Bring the person to the present moment, reduce overwhelm",
    "amplify_agency":   "Reinforce their sense of choice, capability, and control",
    "neutral_explore":  "Explore openly without pushing any direction",
}

STRATEGY_TONES = {
    "validate":         "warm, empathetic, non-judgmental",
    "structure":        "clear, calm, organized",
    "reflect":          "curious, open, gentle",
    "gentle_challenge": "warm but direct, questioning",
    "reframe":          "thoughtful, offering new angle",
    "ground":           "calm, slow, present-focused",
    "amplify_agency":   "affirming, direct, energizing",
    "neutral_explore":  "open, curious, non-directive",
}

STRATEGY_FORBIDDEN_MOVES = {
    "validate":         ["advice giving", "challenging beliefs", "problem-solving immediately"],
    "structure":        ["emotional probing", "challenging core beliefs"],
    "reflect":          ["giving advice", "evaluation", "challenge"],
    "gentle_challenge": ["harsh directness", "multiple challenges at once", "invalidating"],
    "reframe":          ["invalidating current view", "lecturing", "pushing too hard"],
    "ground":           ["complex analysis", "future planning", "any challenge"],
    "amplify_agency":   ["taking over", "giving direct answers", "minimizing difficulty"],
    "neutral_explore":  ["leading questions", "drawing conclusions", "unsolicited advice"],
}

# ---------------------------------------------------------------------------
# Safety gate — forbidden strategies per mode
# Applied BEFORE scoring. Not penalized — eliminated.
# ---------------------------------------------------------------------------

SAFETY_GATE: Dict[str, List[str]] = {
    "overload":     ["gentle_challenge", "reframe"],
    "shame":        ["gentle_challenge", "structure"],
    "avoidance":    ["gentle_challenge"],
    "defense":      [],  # challenge allowed but will score low naturally
    "exploration":  [],
    "commitment":   [],
}

# ---------------------------------------------------------------------------
# Prior probability: strategy → mode fit (heuristic baseline)
# ---------------------------------------------------------------------------

MODE_STRATEGY_PRIOR: Dict[str, Dict[str, float]] = {
    "exploration":  {
        "reflect": 0.80, "neutral_explore": 0.70,
        "reframe": 0.60, "gentle_challenge": 0.50,
    },
    "defense":      {
        "validate": 0.70, "reflect": 0.65,
        "neutral_explore": 0.60, "ground": 0.45,
    },
    "overload":     {
        "validate": 0.90, "ground": 0.80, "structure": 0.55,
    },
    "shame":        {
        "validate": 0.90, "amplify_agency": 0.70, "reframe": 0.55,
    },
    "avoidance":    {
        "reflect": 0.70, "neutral_explore": 0.70, "validate": 0.60,
    },
    "commitment":   {
        "amplify_agency": 0.90, "structure": 0.75, "reflect": 0.50,
    },
}

# Mode transitions expected from strategy (positive = toward better modes)
MODE_SHIFT_QUALITY: Dict[str, Dict[str, float]] = {
    "overload":     {"validate": 0.75, "ground": 0.80, "structure": 0.50},
    "shame":        {"validate": 0.80, "amplify_agency": 0.75, "reframe": 0.55},
    "defense":      {"validate": 0.65, "reflect": 0.60, "neutral_explore": 0.55},
    "avoidance":    {"reflect": 0.65, "neutral_explore": 0.55, "validate": 0.45},
    "exploration":  {"reflect": 0.70, "gentle_challenge": 0.60, "reframe": 0.55},
    "commitment":   {"amplify_agency": 0.90, "structure": 0.75},
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SensitivityProfile:
    challenge_tolerance: float = 0.5   # 0=avoids challenge, 1=welcomes it
    validation_need: float = 0.5        # 0=doesn't need, 1=needs heavily
    shame_sensitivity: float = 0.5      # 0=robust, 1=very sensitive to shame
    autonomy_sensitivity: float = 0.5   # 0=likes guidance, 1=wants full control
    structure_preference: float = 0.5   # 0=free exploration, 1=wants steps


@dataclass
class StrategyScore:
    strategy: str
    rupture_risk: float = 0.0        # risk person shuts down / leaves
    clarity_gain: float = 0.0        # reduces uncertainty in their mind
    agency_gain: float = 0.0         # increases sense of control
    shame_risk: float = 0.0          # risk of triggering shame/guilt
    mode_shift_quality: float = 0.0  # quality of expected mode transition
    trust_preservation: float = 0.5  # maintains trust and safety
    composite: float = 0.0


@dataclass
class InterventionDirective:
    strategy: str
    tone: str
    forbidden_moves: List[str] = field(default_factory=list)
    objective: str = ""
    dissonance_target: str = ""
    horizon: str = "immediate"  # "immediate" | "growth"

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "tone": self.tone,
            "forbidden_moves": self.forbidden_moves,
            "objective": self.objective,
            "dissonance_target": self.dissonance_target,
            "horizon": self.horizon,
        }

    def to_prompt_block(self) -> str:
        """Format directive as a system prompt injection block."""
        forbidden = ", ".join(self.forbidden_moves) if self.forbidden_moves else "none"
        return (
            f"[RESPONSE DIRECTIVE]\n"
            f"Strategy: {self.strategy} — {self.objective}\n"
            f"Tone: {self.tone}\n"
            f"Goal: {self.dissonance_target}\n"
            f"Avoid: {forbidden}"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class InterventionSimulator:
    def __init__(self, db_pool):
        self.db = db_pool

    # --- Sensitivity ---

    async def load_sensitivity(self, user_id: str) -> SensitivityProfile:
        try:
            row = await self.db.fetchrow(
                """SELECT challenge_tolerance, validation_need, shame_sensitivity,
                          autonomy_sensitivity, structure_preference
                   FROM user_sensitivity WHERE user_id=$1""",
                user_id
            )
            if not row:
                return SensitivityProfile()

            def _clamp(v) -> float:
                return min(1.0, max(0.0, float(v)))

            return SensitivityProfile(
                challenge_tolerance=_clamp(row["challenge_tolerance"]),
                validation_need=_clamp(row["validation_need"]),
                shame_sensitivity=_clamp(row["shame_sensitivity"]),
                autonomy_sensitivity=_clamp(row["autonomy_sensitivity"]),
                structure_preference=_clamp(row["structure_preference"]),
            )
        except Exception as e:
            logger.warning(f"InterventionSimulator: sensitivity load failed: {e}")
            return SensitivityProfile()

    async def update_sensitivity(
        self,
        user_id: str,
        used_strategy: str,
        resulting_mode: str,
    ):
        positive_modes = {"exploration", "commitment"}
        delta = 0.04 if resulting_mode in positive_modes else -0.03

        field_map = {
            "gentle_challenge": "challenge_tolerance",
            "validate":         "validation_need",
            "amplify_agency":   "autonomy_sensitivity",
            "structure":        "structure_preference",
        }
        col = field_map.get(used_strategy)
        if not col:
            return

        try:
            await self.db.execute(
                f"""INSERT INTO user_sensitivity (user_id, {col}, updated_at)
                   VALUES ($1, $2, NOW())
                   ON CONFLICT (user_id) DO UPDATE
                   SET {col} = LEAST(1.0, GREATEST(0.0, user_sensitivity.{col} + $3)),
                       updated_at = NOW()""",
                user_id, 0.5 + delta, delta,
            )
        except Exception as e:
            logger.debug(f"Sensitivity update failed: {e}")

    # --- Scoring ---

    def _score(
        self,
        strategy: str,
        mode_state: Any,  # ModeState
        dissonance: Any,  # DissonanceState
        sensitivity: SensitivityProfile,
        tbg_conflict_count: int = 0,
        recent_turning_point: bool = False,
        cold_start: bool = False,
    ) -> StrategyScore:
        s = StrategyScore(strategy=strategy)
        mode = mode_state.mode
        prior = MODE_STRATEGY_PRIOR.get(mode, {}).get(strategy, 0.25)

        # rupture_risk
        if strategy == "gentle_challenge":
            s.rupture_risk = (1.0 - sensitivity.challenge_tolerance) * mode_state.confidence
            if mode == "defense" and mode_state.confidence > 0.7:
                s.rupture_risk = min(1.0, s.rupture_risk + 0.25)
            # TBG: active belief conflicts mean a gentle challenge is grounded in
            # the user's own internal contradiction — not an external imposition.
            # Each confirmed conflict reduces rupture risk slightly (capped at -0.10).
            if tbg_conflict_count > 0:
                s.rupture_risk = max(0.0, s.rupture_risk - min(tbg_conflict_count * 0.05, 0.10))
        elif strategy in ("validate", "ground", "neutral_explore"):
            s.rupture_risk = 0.05
        else:
            s.rupture_risk = 0.15

        # clarity_gain
        if strategy == "structure":
            s.clarity_gain = 0.6 + dissonance.decision_friction * 0.35
        elif strategy == "reframe":
            # TBG: confirmed belief conflicts give reframe a concrete anchor —
            # we're not inventing a new perspective, we're naming a tension that
            # already exists in the user's belief graph.
            tbg_conflict_bonus = min(tbg_conflict_count * 0.07, 0.15)
            s.clarity_gain = 0.45 + dissonance.want_should_gap * 0.30 + tbg_conflict_bonus
        elif strategy == "reflect":
            s.clarity_gain = 0.35 + dissonance.total * 0.25
        elif strategy == "neutral_explore":
            s.clarity_gain = 0.25
        else:
            s.clarity_gain = 0.20

        # agency_gain
        if strategy == "amplify_agency":
            s.agency_gain = 0.75 + sensitivity.autonomy_sensitivity * 0.20
        elif strategy == "structure":
            s.agency_gain = 0.40 + sensitivity.structure_preference * 0.15
        elif strategy in ("validate", "ground"):
            s.agency_gain = 0.20
        else:
            s.agency_gain = 0.30

        # shame_risk
        if strategy in ("gentle_challenge", "structure") and sensitivity.shame_sensitivity > 0.55:
            s.shame_risk = sensitivity.shame_sensitivity * 0.50
        elif strategy == "validate":
            s.shame_risk = 0.0
        else:
            s.shame_risk = 0.05

        # mode_shift_quality
        s.mode_shift_quality = MODE_SHIFT_QUALITY.get(mode, {}).get(strategy, 0.20)

        # trust_preservation
        if strategy in ("validate", "ground", "neutral_explore", "reflect"):
            s.trust_preservation = 0.88
        elif strategy == "gentle_challenge":
            s.trust_preservation = 0.50 + sensitivity.challenge_tolerance * 0.30
        else:
            s.trust_preservation = 0.65

        # composite (weighted)
        s.composite = round(
            prior                        * 0.12 +
            (1 - s.rupture_risk)         * 0.22 +
            s.clarity_gain               * 0.16 +
            s.agency_gain                * 0.14 +
            (1 - s.shame_risk)           * 0.14 +
            s.mode_shift_quality         * 0.12 +
            s.trust_preservation         * 0.10,
            4
        )

        # Recent turning point: person just went through a cognitive shift.
        # Boost strategies that consolidate the new state; suppress destabilizing ones.
        if recent_turning_point:
            if strategy in ("amplify_agency", "validate", "reflect"):
                s.composite = round(s.composite * 1.25, 4)
            elif strategy in ("gentle_challenge", "reframe"):
                s.composite = round(s.composite * 0.6, 4)

        # Cold start: first 5 messages. We lack context, so stick to safe/exploratory.
        if cold_start:
            if strategy in ("neutral_explore", "validate"):
                s.composite = round(s.composite * 1.5, 4)
            elif strategy in ("gentle_challenge", "reframe"):
                s.composite = round(s.composite * 0.3, 4)

        return s

    # --- Main select ---

    def select(
        self,
        mode_state: Any,       # ModeState
        dissonance: Any,       # DissonanceState
        sensitivity: SensitivityProfile,
        tbg_conflict_count: int = 0,
        recent_turning_point: bool = False,
        cold_start: bool = False,
        amf_conf: float = 0.5,
        amf_ambiv: float = 0.0,
    ) -> InterventionDirective:
        """
        Select best intervention strategy.
        Pure Python — no IO. Called synchronously.

        Args:
            tbg_conflict_count: number of active conflict/blocks/contradicts edges
                                 in the user's TBG with confidence >= 0.55.
                                 Adjusts reframe and gentle_challenge scoring.
            recent_turning_point: True if a turning point occurred within last 10 messages.
                                  Boosts consolidation strategies, suppresses destabilizing ones.
            cold_start: True if message_count < 5. Boosts safe/exploratory strategies.
            amf_conf:  AMF stability signal [0,1]. < 0.3 → system has no reliable user model.
            amf_ambiv: AMF volatility signal [0,1). > 0.4 → user is oscillating/shifting.
        """
        mode = mode_state.mode

        # AMF override — runs BEFORE mode safety gate.
        # When the system doesn't understand the user, it has no right to push them.
        amf_regime = "continuation"
        if amf_conf < 0.3:
            # Grounding: system has no reliable model yet — gather data, don't direct.
            amf_regime = "grounding"
            candidates = ["neutral_explore", "validate", "ground"]
            logger.info(
                f"InterventionSimulator: AMF grounding regime "
                f"(conf={amf_conf:.2f} ambiv={amf_ambiv:.2f}) — restricting to safe strategies"
            )
        elif amf_ambiv > 0.4:
            # Calibration: user is oscillating — clarify, don't lead.
            amf_regime = "calibration"
            candidates = ["neutral_explore", "reflect", "validate"]
            logger.info(
                f"InterventionSimulator: AMF calibration regime "
                f"(conf={amf_conf:.2f} ambiv={amf_ambiv:.2f}) — user is in flux"
            )
        else:
            # Continuation: system has reliable model — use normal mode-based scoring.
            # Safety gate: eliminate forbidden strategies
            forbidden = set(SAFETY_GATE.get(mode, []))
            candidates = [s for s in STRATEGIES if s not in forbidden]

        # Score all candidates
        scores = sorted(
            [self._score(s, mode_state, dissonance, sensitivity, tbg_conflict_count, recent_turning_point, cold_start) for s in candidates],
            key=lambda x: x.composite,
            reverse=True,
        )
        best = scores[0]

        # Determine response horizon
        horizon = "growth" if mode == "commitment" else "immediate"

        # Determine dissonance target
        dom = dissonance.dominant_axis()
        target_map = {
            "decision_friction": "reduce decision paralysis",
            "want_should_gap":   "resolve want-vs-should conflict",
            "drive_conflict":    "lower drive conflict",
            "role_strain":       "ease role tension",
        }
        dissonance_target = target_map.get(dom, "lower internal tension")

        directive = InterventionDirective(
            strategy=best.strategy,
            tone=STRATEGY_TONES.get(best.strategy, ""),
            forbidden_moves=STRATEGY_FORBIDDEN_MOVES.get(best.strategy, []),
            objective=STRATEGY_DESCRIPTIONS.get(best.strategy, ""),
            dissonance_target=dissonance_target,
            horizon=horizon,
        )

        logger.info(
            f"InterventionSimulator: {best.strategy} "
            f"(composite={best.composite:.3f}) | mode={mode} | "
            f"dissonance={dissonance.total:.0%} | "
            f"amf_regime={amf_regime} conf={amf_conf:.2f} ambiv={amf_ambiv:.2f}"
        )
        return directive
