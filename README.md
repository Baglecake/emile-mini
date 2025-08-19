# émile-Mini: a lite Enactive Learner  
v0.2.0

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![University of Toronto](https://img.shields.io/badge/University%20of-Toronto-003F7F.svg)](https://www.utoronto.ca/)
[![Research](https://img.shields.io/badge/Type-Research-brightgreen.svg)](https://github.com)
 
**émile-Mini** is a lightweight implementation of **enactive cognition** for research and evaluation. It demonstrates how intelligent behavior can emerge from an agent's ongoing interaction with its environment, featuring endogenous context switching and embodied learning dynamics.

* **Package name:** `emile-mini`
* **Import path:** `emile_mini`
* **CLI command:** `emile-mini`
* **New in v0.2.0:** Comprehensive evaluation framework with RL baselines

---

## 🌱 What is enactive cognition?

Enactivism frames cognition as **meaning-making through reciprocal environment-agent interaction**. Rather than building fixed internal representations, agents develop skills and interpret significance via sensorimotor engagement and context-dependent adaptation.

In **émile-Mini** this manifests as:

* **Learning through doing** (experience changes behavior)
* **Context sensitivity** (same stimulus, different meaning by history)
* **Embodied experience** (energy, proximity, movement matter)
* **Emergent goals/strategies** (not hard-coded rules)
* **Endogenous context switching** (autonomous reframing of situations)

---

## ✨ Key Features (v0.2.0)

### Core Cognitive Architecture
* **Endogenous context switching**: First implementation of autonomous context reframing based on symbolic curvature
* **Quantum-symbolic coupling**: Bidirectional information flow between quantum dynamics and symbolic reasoning
* **Embodied learning**: Sensorimotor integration with energy and spatial dynamics
* **Social cognition**: Multi-agent teaching, learning, and knowledge transfer

### Evaluation & Comparison Framework
* **Cognitive Battery**: Multi-protocol evaluation (solo, context-switch, memory-cued)
* **RL Baselines**: Direct comparison with PPO on standardized navigation tasks
* **Gymnasium Interface**: Standard RL environment compatibility
* **Statistical Analysis**: Comprehensive reporting with confidence intervals

### CLI Demos & Tools
```bash
# Core demos
emile-mini social          # Multi-agent social learning
emile-mini maze            # Context switching in deceptive environments  
emile-mini extinction      # Knowledge preservation without rewards

# Evaluation & comparison
emile-mini battery         # Cognitive battery protocols
emile-mini nav-demo        # Navigation system test
emile-mini nav-compare     # Direct PPO comparison
emile-mini nav-report      # Systematic evaluation across conditions
```

---

## 📊 Benchmark Results

### Navigation Task Performance
Evaluation on standardized navigation tasks (400 episodes per condition):

| Metric | émile-Mini | PPO Baseline | Ratio |
|--------|------------|--------------|-------|
| Success Rate | 64.5% | 10.3% | 6.3x |
| Average Steps | 35.4 | 72.3 | 0.49x |
| SPL Score | 0.655 | 0.127 | 5.2x |

Performance advantage holds across obstacle densities (0.1-0.3) and target quadrants.

### Cognitive Battery Results
| Protocol | Description | Performance |
|----------|-------------|-------------|
| A (Solo) | Embodied exploration | 28.16 ± 10.70 reward |
| C1 (Context) | Multi-phase adaptation | 41.76 ± 11.78 reward |
| C2 (Memory) | Cue-guided navigation | 7.2 ± 5.1 steps to target |

---

## 📦 Installation

**Requirements**
- Python 3.8–3.12
- pip (latest recommended)
- *(Optional, for PPO baseline)* gymnasium ≥ 0.28, stable-baselines3 ≥ 2.3.0

**Tip:** Use a virtual environment to avoid system-package conflicts.

<h4>1) Create & activate a virtual environment (optional but recommended)</h4>
<pre><code class="language-bash"># Unix / macOS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel
</code></pre>

<pre><code class="language-bash"># Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel
</code></pre>

---

### From source (recommended)

<h4>HTTPS</h4>
<pre><code class="language-bash">git clone https://github.com/Baglecake/emile-mini.git
cd emile-mini
python -m pip install -e .
</code></pre>

<h4>SSH (if you use keys)</h4>
<pre><code class="language-bash">git clone git@github.com:Baglecake/emile-mini.git
cd emile-mini
python -m pip install -e .
</code></pre>

<h4>Exact version (this release)</h4>
<pre><code class="language-bash">git fetch --tags
git checkout v0.2.0
python -m pip install -e .
</code></pre>

<h4>Shallow clone of just this tag (faster)</h4>
<pre><code class="language-bash">git clone --depth 1 --branch v0.2.0 https://github.com/Baglecake/emile-mini.git
cd emile-mini
python -m pip install -e .
</code></pre>

---

### Optional: PPO baseline dependencies

<h4>Install PPO extras</h4>
<pre><code class="language-bash">python -m pip install "gymnasium&gt;=0.28" stable-baselines3
</code></pre>

*(If you use CUDA, install the appropriate PyTorch build first, then SB3.)*

---

### Colab 

<h4>Clone + install</h4>
<pre><code class="language-bash">!git clone -q https://github.com/Baglecake/emile-mini.git
%cd emile-mini
!python -m pip install -e .
</code></pre>

<h4>With PPO baseline</h4>
<pre><code class="language-bash">!python -m pip install "gymnasium&gt;=0.28" stable-baselines3
</code></pre>


### From TestPyPI
```bash
pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple emile-mini
```

---

## 🚀 Quick Start

### Basic Cognitive Demos
```bash
# Social learning with multiple agents
emile-mini social --agents 3 --steps 120

# Context switching in deceptive maze
emile-mini maze --steps 200 --size 12

# Knowledge preservation during extinction
emile-mini extinction --episodes 1 --phase-steps 120
```

### Evaluation & Baselines
```bash
# Run complete cognitive battery
emile-mini battery --episodes-a 5 --episodes-c1 3 --episodes-c2 5

# Train PPO baseline for comparison
emile-mini nav-ppo-train --timesteps 50000 --obstacles 0.20

# Direct comparison with PPO
emile-mini nav-compare --episodes 400 --obstacles 0.20

# Comprehensive evaluation report
emile-mini nav-report --episodes 400 --quadrants NE,NW,SE,SW,C
```

### Programmatic Usage
```python
from emile_mini.social_qse_agent_v2 import run_social_experiment

# Run social learning experiment
env, agents, results = run_social_experiment(
    n_agents=5,
    steps=500,
    cluster_spawn=True
)

# Analyze agent knowledge and strategies
for agent in agents:
    print(f"{agent.agent_id}: {agent.current_social_strategy}")
    if agent.embodied_mappings:
        for category, values in agent.embodied_mappings.items():
            print(f"  {category}: {len(values)} experiences")
```

---

## 🔬 Reproduction & Baselines

### Reproducing PPO Comparison
```bash
# Train strong PPO baseline
emile-mini nav-ppo-train --timesteps 100000 --size 20 --max-steps 80 --obstacles 0.20 \
  --quadrant NE --start-mode random --seed 42 --model ppo_nav_strong.zip \
  --progress-k 0.5 --step-cost 0.02 --turn-penalty 0.01 --collision-penalty 0.10 --success-bonus 2.0

# Evaluate both systems
emile-mini nav-compare --episodes 400 --size 20 --max-steps 80 --obstacles 0.20 \
  --quadrant NE --start-mode random --seed 123 --model ppo_nav_strong.zip \
  --progress-k 0.5 --step-cost 0.02 --turn-penalty 0.01 --collision-penalty 0.10 --success-bonus 2.0
```

**Primary metrics:** Success rate and SPL are shaping-independent. Both systems use identical reward coefficients for fair comparison.

### Reward Shaping (Training/Evaluation)
The navigation environment uses transparent reward shaping:
* `progress_k = 0.5` : reward for reducing target distance
* `step_cost = 0.02` : small time penalty
* `turn_penalty = 0.01` : discourages gratuitous turning
* `collision_penalty = 0.10` : penalizes blocked movement
* `success_bonus = 2.0` : bonus on reaching goal

**Note:** Success rate and SPL do **not** depend on these coefficients. Returns are reported alongside success metrics for complete comparison.

---

## 📁 Project Structure

```
emile_mini/
├── __init__.py                       # Package exports and version
├── cli.py                           # Command-line interface
├── cognitive_battery.py             # Multi-protocol evaluation framework
├── nav_env_gym.py                   # Gymnasium interface for RL comparison  
├── ppo_nav_baseline.py              # PPO training and evaluation
├── complete_navigation_system_d.py  # Enhanced navigation agent
├── nav_report.py                    # Systematic evaluation reporting
├── agent.py                         # Core cognitive architecture
├── qse_core.py                      # Quantum surplus emergence engine
├── social_qse_agent_v2.py           # Multi-agent social cognition
├── embodied_qse_emile.py            # Embodied agent implementation
├── maze_environment.py              # Deceptive maze environments
├── extinction_experiment.py         # Knowledge preservation experiments
├── visual_maze_demo.py              # Maze visualization
├── maze_comparison.py               # QSE vs baseline utilities
├── context.py                       # Context switching module
├── goal.py                          # Goal formation and Q-learning
├── memory.py                        # Hierarchical memory system
├── config.py                        # Configuration parameters
└── viz.py                           # Plotting utilities
```

---

## 🧪 Research Framework

### Architectural Contributions
- **Endogenous context switching**: Autonomous reframing based on symbolic curvature thresholds
- **Quantum-symbolic coupling**: Bidirectional information flow between quantum and symbolic systems
- **Embodied integration**: Physical state directly influences cognitive processes
- **Social learning**: Knowledge transfer with trust and novelty mechanisms

### Evaluation Protocols
- **Protocol A**: Solo embodied exploration baseline
- **Protocol C1**: Context-switch adaptation across spatial targets
- **Protocol C2**: Memory-cued navigation with learned spatial knowledge
- **RL Comparison**: Direct evaluation against PPO on identical tasks

### Statistical Validation
- 400+ episodes per condition for robust statistics
- Confidence intervals and effect size analysis
- Multiple environmental conditions (obstacle density, target locations)
- Reproducible evaluation with documented hyperparameters

---

## 🎓 Theoretical Background

**émile-Mini** implements computational enactive cognition, where meaning emerges through agent-environment interaction rather than internal representation processing. Key theoretical foundations:

**Enactive Learning**: Cognition as embodied sense-making through reciprocal interaction, not passive information processing.

**Context-Dependent Adaptation**: The same sensory input can have different meanings based on the agent's experiential history and current context.

**Autopoietic Dynamics**: Self-maintaining cognitive processes that preserve learned knowledge even without external reinforcement.

**Social Meaning-Making**: Knowledge transfer through social interaction, with emergent trust and teaching behaviors.

This work contributes to ongoing research in cognitive architectures, alternatives to standard RL approaches, and computational implementations of enactive cognitive science.

---

## 📊 Previous Research Results

**Dynamic Adaptation**: 100% vs 4.4% success vs epsilon-greedy Q-learning with goal switching (effect size: 0.956)

**Meta-Cognitive Tasks**: 173.61 vs 131.02 vs Q-learning with fast adaptation (p < 0.001)

**Statistical validation**: 4000+ trials across multiple conditions

**Ablation studies**: Confirms quantum-symbolic coupling contribution

---

## 🗺️ Roadmap

* ✅ Comprehensive RL evaluation framework (v0.2.0)
* Real PyPI release with CI/CD
* Additional evaluation protocols and baselines
* Enhanced visualization and analysis tools
* Docker containerization for reproducible environments

---

## 🤝 Contributing

Issues and pull requests welcome! Areas of particular interest:

* New evaluation scenarios and baselines
* Analysis and visualization improvements
* Documentation and example expansion
* Performance optimization

---

## 📚 Citation

```bibtex
@software{emile_mini_2025,
  author = {Coburn, Del},
  title = {émile-mini: a lite enactive learner},
  version = {0.2.0},
  year = {2025},
  url = {https://github.com/Baglecake/emile-mini}
}
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).
