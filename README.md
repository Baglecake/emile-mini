# émile-Mini: a lite enactive learner

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**émile-Mini** is a lightweight, packaged demo of **enactive cognition** for research and teaching. It shows how intelligent behavior can emerge from an agent’s ongoing interaction with its environment—no static world model required.

* **Package name:** `emile-mini` (TestPyPI)
* **Import path:** `emile_mini`
* **CLI command:** `emile-mini`
* **Current focus:** the **social** scenario (cooperation, teaching, loop-breaking, knowledge transfer)

---

## 🌱 What is enactive cognition?

Enactivism frames cognition as **meaning-making through action**. Rather than building fixed internal representations, agents develop skills and significance via sensorimotor engagement.

In **émile-Mini** this looks like:

* **Learning through doing** (experience changes behavior)
* **Context sensitivity** (same stimulus, different meaning by history)
* **Embodied experience** (energy, proximity, movement matter)
* **Emergent goals/strategies** (not hard-coded “if X then Y” rules)

---

## ✨ Key features (v0.1.1)

* **Social learning demo (CLI)**: multi-agent grid world with teaching, helping, cooperative monitoring
* **Knowledge transfer**: trust/novelty checks with high retention of shared values
* **Loop‑breaking pressure**: agents detect repetitive dynamics and force exploration
* **“Stuck” detection & help‑seeking**: agents ask nearby peers for help
* **Episodic surplus logging**: per‑step memory summaries for analysis
* **Proximity & energy models**: shared social detect radius and simple energy costs
* **Visualization**: auto‑saves `enhanced_social_qse_analysis.png` (trajectories, proximities, strategies, signals/relationships)

---

## 📦 Installation

### From TestPyPI (current release)

```bash
pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple emile-mini==0.1.1
```

> **Names to remember**
>
> * Install: `emile-mini`
> * Import: `import emile_mini`
> * CLI: `emile-mini ...`

### From source (dev)

```bash
git clone https://github.com/Baglecake/emile-mini.git
cd emile-mini
pip install -e .
```

---

## 🚀 Quick start

### CLI (recommended)

```bash
# default: 3 agents, 120 steps, cluster spawn radius 4
emile-mini social

# customize
emile-mini social --steps 300 --agents 5 --cluster-radius 6

# check version
emile-mini --version
```

The run logs social events and saves a figure as `enhanced_social_qse_analysis.png`.

### Programmatic (Python)

```python
from emile_mini.social_qse_agent_v2 import run_social_experiment

env, agents, analysis = run_social_experiment(
    n_agents=3,
    steps=120,
    cluster_spawn=True,
    cluster_radius=4,
)

print(analysis["spatial"]["Agent_0"])  # example: proximity stats for Agent_0
```

### Version in code

```python
import emile_mini
print(emile_mini.__version__)
```

---

## 📁 Project structure (installed package)

```
emile_mini/
├── __init__.py                 # exposes __version__ and top-level imports
├── cli.py                      # CLI entry point (emile-mini)
├── agent.py                    # core agent scaffolding
├── qse_core.py                 # surplus/learning dynamics core
├── social_qse_agent_v2.py      # social agents + run_social_experiment(...)
├── embodied_qse_emile.py       # embodied agent/environment (internals)
├── maze_environment.py         # grid world mechanics
├── context.py, goal.py, memory.py, config.py
└── viz.py                      # plotting utilities
```

> In v0.1.1, **the CLI exposes the social scenario**. Other modules are imported in code but aren’t wired into the CLI yet.

---

## 📊 What you’ll observe

* **Emergent social behavior**: teaching, helping, cooperative monitoring
* **Loop‑breaking**: “existential pressure” to exit repetitive dynamics
* **Stuck recovery**: random moves + help‑seeking
* **Knowledge transfer**: retained values (e.g., `crimson_fruit = -0.90`)
* **Proximity dynamics**: close‑encounter counts, average distances
* **Summaries & figure**: automatic analysis + `enhanced_social_qse_analysis.png`

---

## 🎓 Method & research framing

* **MRP context**: émile‑mini is part of a Major Research Paper (computational cognitive science) exploring enactive principles in artificial systems.
* **Computational autoethnography**: the project treats system building and interaction logs as first‑person empirical material—linking design choices, behaviors, and reflection.
* **Empirical semiotics**: meaning is measured by **change**—how architectural impositions and interactions reconfigure the agent–environment system.
* **Relation to enactive AI**: learning is grounded in **sensorimotor coupling** and viability maintenance, not static internal symbols.

### Relation to **émile‑cogito** (multi‑model platform)

émile‑mini is a focused, accessible articulation of a larger research program (Émile/“émile‑cogito”), which explores multi‑module orchestration (e.g., QSE core, symbolic/context/goal/memory stacks) and experiment generation/orchestration paradigms. émiIe‑mini packages a concrete, reproducible slice—**social learning**—for teaching, demos, and analysis.

---

## 🗺️ Roadmap

* Real PyPI release (CI trusted publishing)
* Verbosity flags & deterministic seeding
* More demos exposed via CLI (maze, extinction resilience)
* Dockerfile + GitHub Actions (build/test/publish)

---

## 🤝 Contributing

Issues and PRs welcome! Ideas especially appreciated for:

* New scenarios
* Analysis/visualization
* Docs/examples
* Performance and UX

---

## 📚 Citation

```bibtex
@software{emile_mini_2025,
  author = {Coburn, Del},
  title = {émile-mini: a lite enactive learner},
  year = {2025},
  url = {https://github.com/Baglecake/emile-mini}
}
```

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

