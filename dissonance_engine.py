"""
Dissonance Engine v1.0

Computes internal psychological tension across 4 axes.

Architecture:
  drive_conflict   — extracted from TBG conflict edges (NO LLM)
  role_strain      — extracted from TBG conflict edges (NO LLM)
  want_should_gap  — inferred via LLM (~1 call)
  decision_friction — inferred via LLM (same call as above)

Total dissonance = weighted sum of 4 axes.
Blocked = True when decision_friction > 0.7 (person knows but can't act).
"""
import json
import re
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

LLM_TIMEOUT = 12.0  # seconds — aligned with tbg_engine.py

# Categories that represent drives/motivations
DRIVE_CATEGORIES = {"goals", "values", "fears", "finances", "career"}
# Categories that represent identity/roles
ROLE_CATEGORIES = {"identity", "relationships"}

# Axis weights (must sum to 1.0)
_WEIGHTS = {
    "drive_conflict":    0.30,
    "role_strain":       0.25,
    "want_should_gap":   0.25,
    "decision_friction": 0.20,
}

_LLM_PROMPT = """Analyze the user's internal psychological tension based on their message and known beliefs.

KNOWN BELIEFS:
{belief_summary}

CURRENT MESSAGE:
"{message}"

Assess two dimensions:

want_should_gap (0.0-1.0):
  Is the person torn between what they WANT and what they think they SHOULD do?
  0.0 = no tension, wants and obligations are aligned
  1.0 = strong conflict between desire and obligation
  Signals: "I know I should but...", "I want to but...", conflicting duties and wishes

decision_friction (0.0-1.0):
  Does the person know what to do but can't take action?
  0.0 = no friction, able to act freely
  1.0 = complete paralysis despite knowing the right path
  Signals: "I know but...", stuck, overwhelmed, can't move forward

Return ONLY valid JSON, no markdown:
{{"want_should_gap": 0.0-1.0, "decision_friction": 0.0-1.0, "hotspots": ["conflict description 1", "conflict description 2"], "reasoning": "one sentence"}}

hotspots: specific internal conflicts you detected (English, max 2, each under 60 chars)
"""


@dataclass
class DissonanceState:
    total: float = 0.0
    drive_conflict: float = 0.0     # from TBG edges, no LLM
    role_strain: float = 0.0        # from TBG edges, no LLM
    want_should_gap: float = 0.0    # LLM inferred
    decision_friction: float = 0.0  # LLM inferred
    hotspots: List[str] = field(default_factory=list)
    blocked: bool = False           # True when decision_friction > 0.7
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def dominant_axis(self) -> str:
        if self.total < 0.01:
            return "none"
        axes = {
            "drive_conflict":    self.drive_conflict,
            "role_strain":       self.role_strain,
            "want_should_gap":   self.want_should_gap,
            "decision_friction": self.decision_friction,
        }
        return max(axes, key=axes.get)

    def summary(self) -> str:
        if self.hotspots:
            return f"dissonance={self.total:.0%} [{'; '.join(self.hotspots[:2])}]"
        return f"dissonance={self.total:.0%} (dominant: {self.dominant_axis()})"

    def to_dict(self) -> dict:
        return {
            "total": round(self.total, 3),
            "drive_conflict": round(self.drive_conflict, 3),
            "role_strain": round(self.role_strain, 3),
            "want_should_gap": round(self.want_should_gap, 3),
            "decision_friction": round(self.decision_friction, 3),
            "hotspots": self.hotspots,
            "blocked": self.blocked,
        }


class DissonanceEngine:
    def __init__(self, db_pool):
        self.db = db_pool

    # ------------------------------------------------------------------
    # TBG-based axes (free — no LLM)
    # ------------------------------------------------------------------

    def _axes_from_tbg(self, tbg: Any) -> Dict[str, Any]:
        """
        Extract drive_conflict and role_strain directly from TBG graph edges.
        Reads 'contradicts', 'conflicts_with', 'blocks' edges.
        Returns dict with drive_conflict, role_strain, hotspots.
        """
        if not tbg or not tbg.edges:
            return {"drive_conflict": 0.0, "role_strain": 0.0, "hotspots": []}

        drive_scores = []
        role_scores = []
        hotspots = []

        for edge in tbg.edges.values():
            if edge.relation not in ("contradicts", "conflicts_with", "blocks"):
                continue
            if edge.confidence < 0.4:
                continue

            src = tbg.nodes.get(edge.source_id)
            tgt = tbg.nodes.get(edge.target_id)
            if not src or not tgt:
                continue

            # Edge score = edge confidence × average node confidence
            score = edge.confidence * (src.confidence + tgt.confidence) / 2.0
            label = f"{src.label} vs {tgt.label}"
            hotspots.append(label)

            if src.category in DRIVE_CATEGORIES or tgt.category in DRIVE_CATEGORIES:
                drive_scores.append(score)
            if src.category in ROLE_CATEGORIES or tgt.category in ROLE_CATEGORIES:
                role_scores.append(score)

        def mean_capped(scores):
            if not scores:
                return 0.0
            return min(1.0, sum(scores) / len(scores))

        return {
            "drive_conflict": mean_capped(drive_scores),
            "role_strain": mean_capped(role_scores),
            "hotspots": hotspots[:4],
        }

    # ------------------------------------------------------------------
    # DB
    # ------------------------------------------------------------------

    async def load(self, user_id: str) -> DissonanceState:
        try:
            row = await self.db.fetchrow(
                """SELECT total_score, drive_conflict, role_strain,
                          want_should_gap, decision_friction, hotspots, updated_at
                   FROM user_dissonance WHERE user_id=$1""",
                user_id
            )
            if not row:
                return DissonanceState()
            return DissonanceState(
                total=float(row["total_score"]),
                drive_conflict=float(row["drive_conflict"]),
                role_strain=float(row["role_strain"]),
                want_should_gap=float(row["want_should_gap"]),
                decision_friction=float(row["decision_friction"]),
                hotspots=list(row["hotspots"] or []),
                blocked=float(row["decision_friction"]) > 0.7,
                updated_at=row["updated_at"],
            )
        except Exception as e:
            logger.warning(f"DissonanceEngine.load failed for {user_id[:8]}: {e}")
            return DissonanceState()

    async def save(self, user_id: str, state: DissonanceState):
        try:
            now = datetime.now(timezone.utc)
            await self.db.execute(
                """INSERT INTO user_dissonance
                   (user_id, total_score, drive_conflict, role_strain,
                    want_should_gap, decision_friction, hotspots, updated_at)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                   ON CONFLICT (user_id) DO UPDATE
                   SET total_score=$2, drive_conflict=$3, role_strain=$4,
                       want_should_gap=$5, decision_friction=$6, hotspots=$7, updated_at=$8""",
                user_id, state.total, state.drive_conflict, state.role_strain,
                state.want_should_gap, state.decision_friction,
                state.hotspots, now,
            )
        except Exception as e:
            logger.warning(f"DissonanceEngine.save failed for {user_id[:8]}: {e}")

    # ------------------------------------------------------------------
    # Main compute
    # ------------------------------------------------------------------

    async def compute(
        self,
        user_id: str,
        message: str,
        tbg: Any,
        llm_call_fn,
    ) -> DissonanceState:
        """
        Compute full dissonance state.
        TBG axes are extracted from graph (free).
        LLM used only for want_should_gap + decision_friction (1 call).
        """
        tbg_data = self._axes_from_tbg(tbg)

        belief_summary = tbg.summary() if tbg and tbg.nodes else "none"
        prompt = _LLM_PROMPT.format(
            belief_summary=belief_summary,
            message=message[:600],
        )

        want_should = 0.0
        friction = 0.0
        llm_hotspots: List[str] = []

        try:
            raw = await asyncio.wait_for(llm_call_fn(prompt), timeout=LLM_TIMEOUT)
            raw = re.sub(r"```json|```", "", raw).strip()
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e != -1:
                data = json.loads(raw[s:e + 1])
                want_should = min(1.0, max(0.0, float(data.get("want_should_gap", 0.0))))
                friction = min(1.0, max(0.0, float(data.get("decision_friction", 0.0))))
                llm_hotspots = [str(h)[:60] for h in data.get("hotspots", [])[:2]]
        except Exception as e:
            logger.warning(f"DissonanceEngine LLM failed for {user_id[:8]}: {e}")

        dc = tbg_data["drive_conflict"]
        rs = tbg_data["role_strain"]

        total = (
            dc * _WEIGHTS["drive_conflict"] +
            rs * _WEIGHTS["role_strain"] +
            want_should * _WEIGHTS["want_should_gap"] +
            friction * _WEIGHTS["decision_friction"]
        )

        # Deduplicate hotspots (TBG first, then LLM)
        all_hotspots = list(dict.fromkeys(tbg_data["hotspots"] + llm_hotspots))[:5]

        state = DissonanceState(
            total=round(total, 3),
            drive_conflict=round(dc, 3),
            role_strain=round(rs, 3),
            want_should_gap=round(want_should, 3),
            decision_friction=round(friction, 3),
            hotspots=all_hotspots,
            blocked=friction > 0.7,
        )
        await self.save(user_id, state)
        logger.info(f"DissonanceEngine [{user_id[:8]}]: {state.summary()}")
        return state
