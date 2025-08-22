
from dataclasses import dataclass, fields
import yaml, os

@dataclass
class QSEConfig:
    # Symbolic Field thresholds & gains
    K_PSI: float = 12.0       # Sigmoid gain for Psi
    K_PHI: float = 6.0        # Linear gain for Phi
    THETA_PSI: float = 0.4    # Psi activation threshold
    THETA_PHI: float = 0.5    # Phi activation threshold (lower to allow tension)

    # Surplus Dynamics (Theorem 2 parameters)
    S_GAMMA: float = 0.37         # Growth rate
    S_BETA: float = 0.5          # Curvature coupling
    S_EPSILON: float = 0.35      # Rupture expulsion factor
    S_THETA_RUPTURE: float = 0.5 # Surplus threshold for rupture
    S_TENSION: float = 0.65      # Tension coupling
    S_COUPLING: float = 0.25     # Laplacian coupling strength
    S_DAMPING: float = 0.1       # Damping rate

    # Emergent Time (Theorem 3 parameters)
    TAU_MIN: float = 0.15        # Minimum emergent time
    TAU_MAX: float = 1.0         # Maximum emergent time
    TAU_K: float = 10.0          # Steepness of sigmoid
    TAU_THETA: float = 0.03      # Delta-Sigma threshold

    # Quantum constants
    HBAR: float = 1.0            # Reduced Planck constant
    MASS: float = 1.0            # Particle mass for Schrödinger eq

    # Cognitive input
    INPUT_COUPLING: float = 0.3  # Mixing ratio for external inputs
    GRID_SIZE: int = 64          # Spatial grid dimension

    # NEW: Opt-in multimodal controls
    MULTIMODAL_ENABLED: bool = False
    MODALITY_INFLUENCE_SCALE: float = 0.25

    # Context & regime control
    CONTEXT_REFRACTORY_STEPS: int = 10       # Min steps between context shifts
    RECONTEXTUALIZATION_THRESHOLD: float = 0.20  # ΔS threshold for context shift
    QUANTUM_COUPLING: float = 0.12           # Weight of |ψ|² feedback

    # NEW: Σ smoothing & context switching controls
    SIGMA_EMA_ALPHA: float = 0.20
    RECONTEXT_THRESHOLD: float = 0.35
    RECONTEXT_HYSTERESIS: float = 0.08
    CONTEXT_MIN_DWELL_STEPS: int = 15

    # NEW: Energy balance fixes
    ENERGY_MOVE_COST: float = 0.003
    ENERGY_TURN_COST: float = 0.0015
    ENERGY_EXAMINE_COST: float = 0.0008
    ENERGY_REST_RECOVERY: float = 0.10
    ENERGY_MIN_FLOOR: float = 0.05
    ENERGY_FORAGE_REWARD_MIN: float = 0.08
    ENERGY_FORAGE_REWARD_MAX: float = 0.16

    # NEW: Social / proximity controls
    SOCIAL_DETECT_RADIUS: int = 4
    TEACH_COOLDOWN_STEPS: int = 25
    MIN_CONFIDENCE_RETENTION: float = 0.80

    # NEW: Existential pressure / loop detection
    LOOP_WINDOW: int = 20
    LOOP_SPATIAL_EPS: float = 2.0
    LOOP_BEHAVIORAL_DIVERSITY_MIN: int = 4
    PRESSURE_ENERGY_BOOST: float = 0.30

    # NEW: Memory system controls
    EPISODIC_MEMORY_CAPACITY: int = 1000  # Was 100, now 1000 for experiments

# Safe load of config.yaml, ignoring unknown keys
def load_config(path: str = "config.yaml") -> QSEConfig:
    """Load configuration from YAML, filter to QSEConfig fields."""
    cfg = QSEConfig()
    if os.path.exists(path):
        raw = yaml.safe_load(open(path)) or {}
        valid = {f.name for f in fields(QSEConfig)}
        filtered = {k: v for k, v in raw.items() if k in valid}
        cfg = QSEConfig(**filtered)
    return cfg

# Global config instance
CONFIG = load_config()
