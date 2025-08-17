
"""
Context adaptation module for the QSE-Émile agent:
 - Tracks current context ID and history
 - Evaluates context resistance to new distinctions
 - Performs context shifts with a refractory period
 - Updates context based on QSE metrics
 - ENHANCED: Hysteresis and dwell time to prevent thrashing
"""
import numpy as np
from .config import CONFIG

class ContextModule:
    """
    Manages the agent's current context and triggers recontextualization
    when the system encounters high distinction beyond that context's
    resistance.
    ENHANCED: Hysteresis and proper dwell time to prevent oscillation.
    """
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        # Current context identifier (e.g., an integer label)
        self.current_context = 0
        # History of contexts (for analysis)
        self.context_history = [0]
        # Step tracking for refractory logic
        self._step_count = 0
        self._last_shift = -getattr(cfg, 'CONTEXT_MIN_DWELL_STEPS', 15)

    def evaluate_resistance(self, distinction_level: float) -> float:
        """
        Return context resistance ∈ [0,1] based on distinction level.
        A higher distinction means the current context resists more strongly.
        For now, use a linear mapping clipped to [0,1].
        """
        return float(np.clip(distinction_level, 0.0, 1.0))

    def update(self, qse_metrics: dict):
        """
        Called each QSE step with metrics including:
          - 'distinction_level': float
        Triggers a context shift using hysteresis and proper dwell time.
        ENHANCED: Prevents context thrashing through hysteresis.
        """
        self._step_count += 1
        distinction = float(qse_metrics.get('distinction_level', 0.0))

        # NEW: Enforce minimum dwell time
        dwell = getattr(self.cfg, 'CONTEXT_MIN_DWELL_STEPS', 15)
        if self._step_count - self._last_shift < dwell:
            return

        # NEW: Hysteresis-based thresholds
        hi_threshold = getattr(self.cfg, 'RECONTEXT_THRESHOLD', 0.35)
        hysteresis = getattr(self.cfg, 'RECONTEXT_HYSTERESIS', 0.08)
        lo_threshold = hi_threshold - hysteresis

        # Upward context shift (increase context complexity)
        if distinction > hi_threshold:
            self._shift_context_up()
        # Downward context shift (reduce context complexity) 
        elif distinction < lo_threshold and self.current_context > 0:
            self._shift_context_down()

    def _shift_context_up(self):
        """Shift to higher context (more complex)"""
        self.current_context += 1
        self.context_history.append(self.current_context)
        self._last_shift = self._step_count

    def _shift_context_down(self):
        """Shift to lower context (simpler)"""
        self.current_context = max(0, self.current_context - 1)
        self.context_history.append(self.current_context)
        self._last_shift = self._step_count

    def shift_context(self):
        """
        Legacy method for compatibility.
        Perform a context switch: increment the context ID,
        record it, and reset refractory.
        """
        self._shift_context_up()

    def get_current(self) -> int:
        """Return the current context ID."""
        return self.current_context

    def get_history(self) -> list:
        """Return the full context history."""
        return list(self.context_history)
