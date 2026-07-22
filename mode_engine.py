"""
Mode Engine v1.0

Detects current psychological mode from conversation.
6 modes grounded in Schema Therapy + cognitive psychology research (2024).

Modes:
  exploration  — thinking openly, processing, curious
  defense      — justifying, deflecting, resistant
  overload     — overwhelmed, exhausted, short responses
  shame        — self-critical, guilty, self-blame
  avoidance    — changing topic, minimizing, not engaging
  commitment   — decided, planning, taking ownership

Uses: single LLM call per message (~200ms)
Persists: user_mode table (current state only, fast reads)
"""
import json
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

VALID_MODES = {
    "exploration", "defense", "overload", "shame", "avoidance", "commitment"
}

MODE_DESCRIPTIONS = {
    "exploration":  "Thinking openly, asking questions, processing ideas",
    "defense":      "Justifying, deflecting, contradicting, resistant",
    "overload":     "Overwhelmed, exhausted, unclear, short answers",
    "shame":        "Self-critical, guilty, self-blame, 'I should have'",
    "avoidance":    "Changing topic, minimizing, not engaging with core issue",
    "commitment":   "Decided, planning, taking ownership, specific steps",
}

# Modes where LLM response must be calibrated carefully
HIGH_RISK_MODES = {"overload", "shame"}

_INFERENCE_PROMPT = """You are a cognitive state analyst. Determine the user's current psychological mode.

MODES (pick exactly one):
- exploration:  thinking openly, asking questions, curious, processing
- defense:      "yes but", justifying, deflecting, contradicting, resistant, aggressive
- overload:     overwhelmed, exhausted, "I don't know", "everything is bad", very short answers
- shame:        "I'm to blame", "I should have", self-critical, guilt, self-punishment
- avoidance:    changing topic, "let's forget it", minimizing, not engaging
- commitment:   "I've decided", specific plans, taking ownership, "I will"
{tbg_context}
RECENT CONVERSATION:
{conversation}

CURRENT MESSAGE:
"{message}"

Return ONLY valid JSON, no markdown:
{{"mode": "mode_name", "confidence": 0.0-1.0, "stability": 0.0-1.0, "triggers": ["trigger1", "trigger2"], "reasoning": "one sentence"}}

confidence: how certain you are about this mode (low if message is ambiguous)
stability:  how stable this mode seems (low = likely to shift soon)
triggers:   what caused this mode, max 3 short phrases in English
"""


@dataclass
class ModeState:
    mode: str = "exploration"
    confidence: float = 0.5
    stability: float = 0.5
    triggers: List[str] = field(default_factory=list)
    shift_risk: float = 0.5
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_high_risk(self) -> bool:
        return self.mode in HIGH_RISK_MODES and self.confidence > 0.6

    def allows_challenge(self) -> bool:
        """True if bot can gently challenge user's beliefs."""
        if self.mode in HIGH_RISK_MODES:
            return False
        if self.mode == "defense" and self.confidence > 0.75:
            return False
        return True

    def summary(self) -> str:
        return f"{self.mode}(conf={self.confidence:.0%}, stable={self.stability:.0%})"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "confidence": round(self.confidence, 3),
            "stability": round(self.stability, 3),
            "shift_risk": round(self.shift_risk, 3),
            "triggers": self.triggers,
        }


class ModeEngine:
    def __init__(self, db_pool):
        self.db = db_pool

    async def load(self, user_id: str) -> ModeState:
        try:
            row = await self.db.fetchrow(
                "SELECT mode, confidence, stability, triggers, updated_at "
                "FROM user_mode WHERE user_id=$1",
                user_id
            )
            if not row:
                return ModeState()
            return ModeState(
                mode=row["mode"],
                confidence=float(row["confidence"]),
                stability=float(row["stability"]),
                triggers=list(row["triggers"] or []),
                shift_risk=round(1.0 - float(row["stability"]), 3),
                updated_at=row["updated_at"],
            )
        except Exception as e:
            logger.warning(f"ModeEngine.load failed for {user_id[:8]}: {e}")
            return ModeState()

    async def save(self, user_id: str, state: ModeState):
        try:
            now = datetime.now(timezone.utc)
            await self.db.execute(
                """INSERT INTO user_mode (user_id, mode, confidence, stability, triggers, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6)
                   ON CONFLICT (user_id) DO UPDATE
                   SET mode=$2, confidence=$3, stability=$4, triggers=$5, updated_at=$6""",
                user_id, state.mode, state.confidence, state.stability,
                state.triggers, now,
            )
        except Exception as e:
            logger.warning(f"ModeEngine.save failed for {user_id[:8]}: {e}")

    async def infer(
        self,
        user_id: str,
        message: str,
        conversation_history: List[tuple],
        llm_call_fn,
        tbg_insight: str = "",
    ) -> ModeState:
        """
        Infer current psychological mode from message + recent history.
        Saves result to DB. Falls back to last known state on error.

        Args:
            user_id:             user identifier
            message:             user's current message
            conversation_history: list of (role, text) tuples, last N messages
            llm_call_fn:         async callable (prompt: str) -> str
            tbg_insight:         optional TBG belief summary (from TBGEngine.get_insight)
        """
        conv_str = "\n".join(
            f"{role}: {text[:200]}" for role, text in conversation_history[-4:]
        )
        # Inject TBG belief context only when available — helps LLM ground the mode
        # in the user's persistent beliefs, not just the current message surface.
        tbg_context = (
            f"\nKNOWN BELIEFS (long-term memory context):\n{tbg_insight}\n"
            if tbg_insight else ""
        )
        prompt = _INFERENCE_PROMPT.format(
            tbg_context=tbg_context,
            conversation=conv_str or "none",
            message=message[:600],
        )

        try:
            raw = await llm_call_fn(prompt)
            raw = re.sub(r"```json|```", "", raw).strip()
            s, e = raw.find("{"), raw.rfind("}")
            if s == -1 or e == -1:
                raise ValueError("No JSON found")
            data = json.loads(raw[s:e + 1])

            mode = data.get("mode", "exploration")
            if mode not in VALID_MODES:
                logger.debug(f"ModeEngine: unknown mode '{mode}', defaulting to exploration")
                mode = "exploration"

            conf = float(data.get("confidence", 0.5))
            stab = float(data.get("stability", 0.5))
            triggers = [str(t)[:50] for t in data.get("triggers", [])[:3]]

            state = ModeState(
                mode=mode,
                confidence=round(min(1.0, max(0.0, conf)), 3),
                stability=round(min(1.0, max(0.0, stab)), 3),
                triggers=triggers,
                shift_risk=round(1.0 - min(1.0, max(0.0, stab)), 3),
            )
            await self.save(user_id, state)
            logger.info(f"ModeEngine [{user_id[:8]}]: {state.summary()}")
            return state

        except Exception as e:
            logger.warning(f"ModeEngine.infer failed for {user_id[:8]}: {e}")
            return await self.load(user_id)
