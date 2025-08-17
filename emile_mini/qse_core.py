"""
Core Quantum Surplus Emergence engine:
 - Symbolic curvature kernels
 - Surplus field update
 - Emergent time calculation
 - Schrödinger evolution via split-step FFT
 - Quantum→surplus feedback
"""
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from .config import CONFIG

# -- Symbolic Curvature (Theorem 1) --
def calculate_symbolic_fields(S: np.ndarray, cfg=CONFIG):
    """
    Compute Psi, Phi, and Sigma fields from surplus S.
    Psi = sigmoid(K_PSI*(S - THETA_PSI))
    Phi = max(0, K_PHI*(S - THETA_PHI))
    Sigma = Psi - Phi
    """
    psi = 1.0 / (1.0 + np.exp(-cfg.K_PSI * (S - cfg.THETA_PSI)))
    phi = np.maximum(0.0, cfg.K_PHI * (S - cfg.THETA_PHI))
    sigma = psi - phi
    return psi, phi, sigma

# -- Emergent Time (Theorem 3) --
def calculate_emergent_time(sigma: np.ndarray, sigma_prev: np.ndarray, cfg=CONFIG) -> float:
    """
    Compute emergent time tau based on change in Sigma.
    Tau = TAU_MIN + (TAU_MAX-TAU_MIN)/(1+exp(K*(delta-THETA)))
    """
    if sigma_prev is None:
        return cfg.TAU_MAX
    delta = np.mean(np.abs(sigma - sigma_prev))
    raw = cfg.TAU_MIN + (cfg.TAU_MAX - cfg.TAU_MIN) / (1.0 + np.exp(cfg.TAU_K * (delta - cfg.TAU_THETA)))
    return float(np.clip(raw, cfg.TAU_MIN, cfg.TAU_MAX))

# -- Surplus Dynamics (Theorem 2) --
def update_surplus(S: np.ndarray, sigma: np.ndarray, dt: float, cfg=CONFIG) -> np.ndarray:
    """
    Update surplus field S:
      S_new = (1+gamma*dt)*S + beta*dt*sigma - expel + tension_coupling*laplacian - damping*S + noise
    Expel when |sigma|>theta_rupture.
    """
    g = cfg.S_GAMMA * dt
    b = cfg.S_BETA * dt
    e = cfg.S_EPSILON * dt
    t = cfg.S_TENSION * dt
    c = cfg.S_COUPLING * dt
    d = cfg.S_DAMPING * dt
    # rupture expulsion
    expel = np.where(np.abs(sigma) > cfg.S_THETA_RUPTURE, e * S, 0.0)
    # basic growth + coupling
    S_new = (1.0 + g) * S + b * sigma - expel
    # Laplacian coupling
    lap = np.roll(S, 1) - 2.0 * S + np.roll(S, -1)
    S_new += t * c * lap
    # damping
    S_new -= d * S
    # stochastic noise
    S_new += 0.01 * np.sqrt(dt) * np.random.randn(*S.shape)
    # clamp to [0,1]
    return np.clip(S_new, 0.0, 1.0)

# -- Potential generation --
def create_double_well_potential(x: np.ndarray) -> np.ndarray:
    """Static double-well + barrier potential"""
    width = (x.max() - x.min()) / 8.0
    wells = -np.exp(-((x + 2*width)**2) / (2 * width**2))
    wells += -np.exp(-((x - 2*width)**2) / (2 * width**2))
    barrier = 0.5 * np.exp(-x**2 / (width**2 / 2.0))
    V = wells + barrier
    V = V - V.min()
    return 0.2 * V


def create_dynamic_potential(x: np.ndarray, sigma: np.ndarray, cfg=CONFIG, t: float = 0.0) -> np.ndarray:
    """
    Build a time-varying potential: base double-well + Gaussian barrier + sigma coupling.
    """
    base = create_double_well_potential(x)
    barrier = (0.3 + 0.2 * np.sin(t / 5.0)) * np.exp(-x**2 / ((len(x)/8.0)**2))
    pot = base + barrier + cfg.INPUT_COUPLING * sigma
    return pot - pot.min()

# -- Schrödinger step (Split-step FFT) --
def schrodinger_step(psi: np.ndarray, V: np.ndarray, x: np.ndarray, dt: float, cfg=CONFIG) -> np.ndarray:
    """
    Evolve wavefunction psi under potential V for time dt using split-step Fourier method.
    """
    N = psi.size
    dx = x[1] - x[0]
    # k-space frequencies
    k = fftfreq(N, d=dx) * 2.0 * np.pi
    # half-step kinetic
    psi_k = fft(psi)
    psi = ifft(np.exp(-1j * cfg.HBAR * k**2 / (2 * cfg.MASS) * dt / 2.0) * psi_k)
    # potential step
    psi = np.exp(-1j * V * dt / cfg.HBAR) * psi
    # half-step kinetic
    psi_k = fft(psi)
    psi = ifft(np.exp(-1j * cfg.HBAR * k**2 / (2 * cfg.MASS) * dt / 2.0) * psi_k)
    return psi

# -- QSE Engine --
class QSEEngine:
    """
    Encapsulates the QSE loop: surplus update, quantum evolution, feedback.
    """
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        # Surplus field
        self.S = 0.1 + 0.05 * np.random.rand(cfg.GRID_SIZE)
        self.sigma_prev = None
        # Spatial grid for quantum
        self.x = np.linspace(-1.0, 1.0, cfg.GRID_SIZE)
        dx = self.x[1] - self.x[0]
        # Initial Gaussian wavepacket
        psi0 = np.exp(-self.x**2 / (2.0 * (0.2)**2))
        norm = np.sqrt((np.abs(psi0)**2).sum() * dx)
        self.psi = psi0 / norm
        # Time tracker
        self.time = 0.0
        # History
        self.history = []

    def step(self, sigma: np.ndarray, dt: float = 0.01) -> dict:
        """
        Perform one QSE step:
          1) Update surplus using sigma
          2) Build potential & evolve quantum state
          3) Feed quantum back into surplus
          4) Record metrics
        Returns a dict of key metrics.
        """
        # 1) Surplus update
        self.S = update_surplus(self.S, sigma, dt, self.cfg)
        # 2) Quantum evolution
        V = create_dynamic_potential(self.x, sigma, self.cfg, self.time)
        self.psi = schrodinger_step(self.psi, V, self.x, dt, self.cfg)
        # 3) Quantum->Surplus feedback
        prob = np.abs(self.psi)**2
        alpha = self.cfg.QUANTUM_COUPLING
        self.S = (1.0 - alpha) * self.S + alpha * prob
        # 4) Record and advance
        metrics = {
            'time': self.time,
            'surplus_mean': float(self.S.mean()),
            'sigma_mean': float(sigma.mean()),
            'prob_density': prob.copy(),
            'psi': self.psi.copy(),
        }
        self.history.append(metrics)
        self.sigma_prev = sigma.copy()
        self.time += dt

        return metrics
