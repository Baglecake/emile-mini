"""
EmileAgent: orchestrates the QSE core and cognitive modules.
Each step performs:
  1) Symbolic reasoning -> Σ
  2) QSE engine update -> surplus, quantum state
  3) Context adaptation
  4) Goal selection & feedback
  5) Memory storage
  6) History logging
ENHANCED: EMA smoothed Σ, proper memory writes, existential pressure handling
"""
from types import SimpleNamespace
from .config import CONFIG
from .qse_core import QSEEngine
from .symbolic import SymbolicReasoner
from .memory import MemoryModule
from .context import ContextModule
from .goal import GoalModule
import numpy as np
from collections import deque

class _NullBody:
    def __init__(self, cfg):
        self.state = SimpleNamespace(energy=getattr(cfg, 'ENERGY_MIN_FLOOR', 0.05))


class EmileAgent:
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg or QSEConfig()
        # Core engine
        self.qse = QSEEngine(cfg)
        # Cognitive modules
        self.symbolic = SymbolicReasoner(cfg)
        self.context = ContextModule(cfg)
        self.memory = MemoryModule(cfg)
        self.goal = GoalModule(cfg)
        # Episode history
        self.history = {
            'surplus_mean': [],
            'sigma_mean': [],
            'context_id': [],
            'goal': []
        }
        
        # NEW: Existential pressure detection
        self.position_history = deque(maxlen=getattr(cfg, 'LOOP_WINDOW', 20))
        self.action_history = deque(maxlen=getattr(cfg, 'LOOP_WINDOW', 20))
        self.force_explore = False
        self.step_counter = 0
        
        # NEW: Body state for embodied agents (if needed)
        self.body = None  # Will be set by embodied subclasses
        if not hasattr(self, 'body') or self.body is None:
            self.body = _NullBody(self.cfg)

    def detect_repetition(self, window=None):
        """
        NEW: Detect if agent is stuck in repetitive loops
        """
        if window is None:
            window = getattr(self.cfg, 'LOOP_WINDOW', 20)
            
        if len(self.position_history) < window:
            return False
            
        # Spatial repetition check
        recent_positions = list(self.position_history)[-window:]
        unique_positions = len(set(tuple(p) if hasattr(p, '__iter__') else p for p in recent_positions))
        spatial_eps = getattr(self.cfg, 'LOOP_SPATIAL_EPS', 2.0)
        
        if unique_positions <= spatial_eps:
            return True
            
        # Behavioral diversity check
        if len(self.action_history) >= window:
            recent_actions = list(self.action_history)[-window:]
            action_diversity = len(set(recent_actions))
            min_diversity = getattr(self.cfg, 'LOOP_BEHAVIORAL_DIVERSITY_MIN', 4)
            
            if action_diversity < min_diversity:
                return True
                
        return False

    def step(self, dt: float = 0.01, external_input: dict = None) -> dict:
        """
        Perform one cognitive step:
          - Compute Σ from surplus
          - Advance QSE engine
          - Update context
          - Select and evaluate goal
          - Store in memory
          - Log history
        external_input may contain 'reward' or other signals.
        Returns the QSE metrics dict.
        ENHANCED: Uses EMA'd Σ, proper memory writes, existential pressure
        """
        self.step_counter += 1
        
        # 1) Symbolic: compute curvature Σ
        sigma = self.symbolic.step(self.qse.S)

        # 2) QSE update
        metrics = self.qse.step(sigma, dt)
        surplus = metrics['surplus_mean']
        sigma_mean = metrics['sigma_mean']

        # 3) Context adaptation - NEW: Use EMA'd Σ for stable decisions
        distinction = abs(self.symbolic.get_sigma_ema())  # Use smoothed value!
        qse_metrics = {
            'distinction_level': distinction,
            'normalized_entropy': metrics.get('normalized_entropy', 0.5)
        }
        old_context = self.context.get_current()
        self.context.update(qse_metrics)
        ctx = self.context.get_current()

        # 4) Goal selection & feedback
        goal_id = self.goal.select_action(qse_metrics)
        # Feedback: use external reward if provided
        reward = 0.0
        if external_input and 'reward' in external_input:
            reward = external_input['reward']
        self.goal.feedback(reward)

        # NEW: 5) Existential pressure detection and intervention
        if self.detect_repetition():
            print(f"⚠️ Agent existential pressure detected at step {self.step_counter}")
            
            # Energy boost to escape death spiral
            if self.body and hasattr(self.body.state, 'energy'):
                energy_boost = getattr(self.cfg, 'PRESSURE_ENERGY_BOOST', 0.30)
                self.body.state.energy = min(1.0, self.body.state.energy + energy_boost)
                print(f"   💊 Energy boosted to {self.body.state.energy:.3f}")
            
            # Set exploration flag
            self.force_explore = True
            
            # Reset exploitation bias in goal system
            if hasattr(self.goal, 'reset_exploitation_bias'):
                self.goal.reset_exploitation_bias()

        # 6) Memory storage - NEW: Comprehensive structured memory
        memory_entry = {
            'step': self.step_counter,
            'surplus': float(surplus),
            'sigma_mean_raw': float(sigma_mean),
            'sigma_mean_ema': float(distinction),
            'context': int(ctx),
            'goal': goal_id,
            'reward': float(reward),
            'existential_pressure': self.detect_repetition(),
            'context_switched': ctx != old_context
        }
        
        # Add position if available (for embodied agents)
        if self.body and hasattr(self.body.state, 'position'):
            memory_entry['position'] = self.body.state.position
            memory_entry['energy'] = float(self.body.state.energy)
            
            # Track position for loop detection
            self.position_history.append(self.body.state.position)
        
        self.memory.store(memory_entry, tags={'type': 'episodic'})

        # 7) History logging
        self.history['surplus_mean'].append(surplus)
        self.history['sigma_mean'].append(sigma_mean)
        self.history['context_id'].append(ctx)
        self.history['goal'].append(goal_id)

        # NEW: Enhanced metrics return
        enhanced_metrics = metrics.copy()
        enhanced_metrics.update({
            'context': ctx,
            'context_switched': ctx != old_context,
            'distinction_level': distinction,
            'goal': goal_id,
            'reward': reward,
            'existential_pressure': self.detect_repetition(),
            'force_explore': self.force_explore
        })

        return enhanced_metrics

    def get_history(self) -> dict:
        """Return recorded history of the agent."""
        return self.history
        
    def reset_exploration_flag(self):
        """Reset the force_explore flag (called by environment)"""
        self.force_explore = False
