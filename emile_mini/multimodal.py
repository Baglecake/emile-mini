from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

@dataclass
class ModalityFeature:
    name: str
    vec: np.ndarray
    weight: float = 1.0

class TextAdapter:
    def __init__(self, dim: int = 64):
        self.dim = dim
    def encode(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(self.dim, dtype=float)
        v = np.zeros(self.dim, dtype=float)
        for tok in str(text).lower().split():
            v[hash(tok) % self.dim] += 1.0
        return v

class ImageAdapter:
    def __init__(self, dim: int = 128):
        self.dim = dim
    def encode(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return np.zeros(self.dim, dtype=float)
        x = np.asarray(img, dtype=float).ravel()
        if x.size <= self.dim:
            out = np.zeros(self.dim, dtype=float)
            out[:x.size] = x
            return out
        stride = max(1, x.size // self.dim)
        return x[::stride][:self.dim]

class AudioAdapter:
    def __init__(self, dim: int = 64):
        self.dim = dim
    def encode(self, wav: np.ndarray) -> np.ndarray:
        if wav is None:
            return np.zeros(self.dim, dtype=float)
        x = np.asarray(wav, dtype=float).ravel()
        if x.size == 0:
            return np.zeros(self.dim, dtype=float)
        frames = np.array_split(x, self.dim)
        return np.array([float(np.mean(f**2)) for f in frames], dtype=float)

class ModalityFusion:
    """
    Lightweight, deterministic fusion:
    - Layer-norm the concatenated weighted feature vector
    - Fixed random projection → K coefficients
    - Combine sinusoidal basis → 1D curvature pattern with bounded influence
    """
    def __init__(self, grid_size: int, basis_count: int = 8, influence_scale: float = 0.25):
        self.grid_size = int(grid_size)
        self.basis_count = int(basis_count)
        self.influence_scale = float(influence_scale)
        x = np.linspace(0, np.pi, self.grid_size)
        self.basis = np.stack([np.sin((k+1) * x) for k in range(self.basis_count)], axis=0)
        self._proj_seed = 12345
        self._proj = None

    def fuse(self, mods: List[ModalityFeature]) -> Optional[np.ndarray]:
        if not mods:
            return None
        parts = []
        for m in mods:
            v = np.asarray(m.vec, dtype=float).ravel()
            if v.size == 0 or not np.all(np.isfinite(v)):
                continue
            parts.append(float(m.weight) * v)
        if not parts:
            return None
        z = np.concatenate(parts, axis=0)
        mu, sd = float(np.mean(z)), float(np.std(z) + 1e-8)
        return (z - mu) / sd

    def to_sigma(self, fused: Optional[np.ndarray]) -> np.ndarray:
        if fused is None or fused.size == 0:
            return np.zeros(self.grid_size, dtype=float)
        K = self.basis_count
        if self._proj is None or self._proj.shape[1] != fused.size:
            rng = np.random.default_rng(self._proj_seed)
            self._proj = rng.standard_normal((K, fused.size)) / np.sqrt(max(1, fused.size))
        coeffs = self._proj @ fused   # [K]
        sigma_mod = coeffs @ self.basis  # [N]
        sigma_mod -= float(np.mean(sigma_mod))
        denom = float(np.std(sigma_mod) + 1e-6)
        sigma_mod = (sigma_mod / denom) * self.influence_scale
        return np.clip(sigma_mod, -1.0, 1.0)

class ModalityAttentionPolicy:
    """
    Compute per-modality weight multipliers from agent internal state.
    Priority:
      1) agent.attention_mode ('listening', 'reading', 'looking')
      2) agent.context.get_current() heuristics
      3) default 1.0 for all
    """
    def __init__(self):
        self.mode_weights: Dict[str, Dict[str, float]] = {
            'listening': {'audio': 1.8, 'text': 0.8, 'vision': 0.8, 'proprio': 1.0},
            'reading':   {'text': 1.8,  'audio': 0.7, 'vision': 1.2, 'proprio': 1.0},
            'looking':   {'vision': 1.8, 'text': 0.8, 'audio': 0.8, 'proprio': 1.0},
        }
        self.context_weights: Dict[int, Dict[str, float]] = {
            1: {'vision': 1.3},
            2: {'vision': 1.2, 'proprio': 1.2},
            4: {'vision': 1.2},
        }
    def weights_for(self, agent) -> Dict[str, float]:
        w: Dict[str, float] = {}
        mode = getattr(agent, 'attention_mode', None)
        if isinstance(mode, str) and mode in self.mode_weights:
            w.update(self.mode_weights[mode])
        try:
            ctx = int(agent.context.get_current()) if hasattr(agent, 'context') else 0
        except Exception:
            ctx = 0
        w2 = self.context_weights.get(ctx, {})
        for k, v in w2.items():
            w[k] = max(w.get(k, 1.0), v)
        return w or {'vision': 1.0, 'text': 1.0, 'audio': 1.0, 'proprio': 1.0}