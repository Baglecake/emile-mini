
"""
SymbolicReasoner: the high‑level symbolic processor for the QSE–Émile agent.
Responsibilities:
  - Compute symbolic fields Ψ, Φ, and Σ from the surplus field S
  - Maintain a short history of Σ for monitoring
  - Adapt its internal thresholds (θᵖˢⁱ, θᵖʰⁱ) based on QSE metrics
  - Produce a "curvature" output for each cycle to feed back into QSE
  - ENHANCED: EMA smoothing for stable context decisions
"""

import numpy as np
from .config import CONFIG
from .qse_core import calculate_symbolic_fields

class SymbolicReasoner:
    def __init__(self, cfg=CONFIG):
        """
        Initialize with the shared configuration.
        ENHANCED: Add EMA smoothing for Σ
        """
        self.cfg = cfg
        # Keep a history of mean Σ values for diagnostics
        self.sigma_history = []
        # NEW: EMA smoothed sigma for stable context decisions
        self._sigma_ema = 0.0

    def adjust_parameters(self, qse_metrics: dict):
        """
        Optionally tweak THETA_PSI or THETA_PHI based on QSE state.
        For example, if QSE reports low coherence, lower THETA_PHI slightly
        to make rupture (negative Σ) easier in the next step.

        qse_metrics contains keys like:
          - 'phase_coherence': float in [0,1]
          - 'normalized_entropy': float in [0,1]
          - 'regime': one of 'tension','quantum_oscillation','stable_coherence'
        """
        coh = qse_metrics.get('phase_coherence', None)
        regime = qse_metrics.get('regime', None)
        if coh is not None:
            # If coherence is very low, ease rupture by lowering THETA_PHI
            if coh < 0.3:
                self.cfg.THETA_PHI = max(0.1, self.cfg.THETA_PHI * 0.98)
            # If coherence is very high, raise THETA_PHI to favor stability
            elif coh > 0.8:
                self.cfg.THETA_PHI = min(1.0, self.cfg.THETA_PHI * 1.02)

        # Example: if we just entered a tension regime, make the next curvature stronger
        if regime == 'tension':
            self.cfg.THETA_PSI = max(0.1, self.cfg.THETA_PSI * 0.97)

    def step(self, surplus_field: np.ndarray) -> np.ndarray:
        """
        Given the current surplus field S, compute and return the symbolic curvature Σ.
        Also record its mean in history.
        ENHANCED: Compute EMA smoothed sigma for stable context decisions.
        """
        # Compute Ψ and Φ and get Σ = Ψ − Φ
        psi, phi, sigma = calculate_symbolic_fields(surplus_field, self.cfg)

        # Record raw mean for diagnostics
        mean_sigma = float(np.mean(sigma))
        
        # NEW: EMA smoothing for stable context decisions
        alpha = getattr(self.cfg, 'SIGMA_EMA_ALPHA', 0.20)
        self._sigma_ema = (1 - alpha) * self._sigma_ema + alpha * mean_sigma
        
        # Store the smoothed value in history
        self.sigma_history.append(self._sigma_ema)

        return sigma

    def get_sigma_ema(self) -> float:
        """
        Return the current EMA-smoothed sigma value for stable context decisions.
        """
        return self._sigma_ema

    def get_sigma_history(self) -> np.ndarray:
        """
        Return the recorded history of EMA-smoothed Σ values for plotting or analysis.
        """
        return np.array(self.sigma_history)
