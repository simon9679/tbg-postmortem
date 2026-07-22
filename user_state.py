"""
TBG User Axis State — S2.3 / S2.6

Tracks aggregate cognitive axis positions for a user across all their BeliefNodes.
Recomputed on every apply_delta cycle by TBGEngine._update_axis_state().

Fields:
  positions          — confidence-weighted mean of top-8 nodes by confidence
  peak_positions     — max absolute value seen per axis over full history (never resets)
  volatility         — rolling std over last VOLATILITY_WINDOW updates per axis
  trajectory         — per-axis time series: [[message_count, position], ...]
  updated_at_message — message_count when state was last recomputed

Constants:
  TRAJECTORY_MAX    = 20  — max trajectory points kept per axis
  VOLATILITY_WINDOW =  5  — window size for rolling volatility std
  TOP_N_NODES       =  8  — only top-N nodes by confidence contribute to positions
"""
from typing import Dict, List
from pydantic import BaseModel, Field

TRAJECTORY_MAX = 20
VOLATILITY_WINDOW = 5
TOP_N_NODES = 8


class UserAxisState(BaseModel):
    """Aggregate semantic axis state for one user across all active BeliefNodes."""

    # Confidence-weighted mean over TOP_N_NODES highest-confidence nodes
    positions: Dict[str, float] = Field(default_factory=dict)

    # Max absolute value seen per axis over full history — never resets.
    # Captures the strongest signal that mean-averaging would otherwise dilute.
    peak_positions: Dict[str, float] = Field(default_factory=dict)

    # Rolling standard deviation over last VOLATILITY_WINDOW updates per axis
    volatility: Dict[str, float] = Field(default_factory=dict)

    # Per-axis trajectory: [[message_count, position], ...]
    # Capped at TRAJECTORY_MAX entries; oldest dropped first.
    trajectory: Dict[str, List[List[float]]] = Field(default_factory=dict)

    # message_count when this state was last recomputed
    updated_at_message: int = 0

    model_config = {"extra": "ignore"}

    def summary(self) -> str:
        """
        One-line human-readable summary of current axis positions.
        Format: "evaluation:+0.31 | potency:-0.12 | activity:~0.03 | ..."
        Returns "no axis state" when positions are empty.
        """
        if not self.positions:
            return "no axis state"

        parts = []
        for axis, pos in self.positions.items():
            if pos > 0.1:
                sign = "+"
            elif pos < -0.1:
                sign = "-"
            else:
                sign = "~"
            parts.append(f"{axis}:{sign}{abs(pos):.2f}")
        return " | ".join(parts)
