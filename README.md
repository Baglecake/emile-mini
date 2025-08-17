# émile-mini: a lite enactive learner

**émile-mini** is a lightweight, packaged demo of **enactive cognition** for research and teaching. It shows how intelligent behavior can emerge from an agent’s ongoing interaction with its environment—no static world model required.

* **Package name:** `emile-mini` (TestPyPI)
* **Import path:** `emile_mini`
* **CLI command:** `emile-mini`
* **Current CLI demos:** **social**, **maze**, **extinction** (cooperation/teaching; context‑switching; extinction/recovery)

---

## 🌱 What is enactive cognition?

Enactivism frames cognition as **meaning‑making through action**. Rather than building fixed internal representations, agents develop skills and significance via sensorimotor engagement.

In **émile-mini** this looks like:

* **Learning through doing** (experience changes behavior)
* **Context sensitivity** (same stimulus, different meaning by history)
* **Embodied experience** (energy, proximity, movement matter)
* **Emergent goals/strategies** (not hard‑coded “if X then Y” rules)

---

## ✨ Key features (v0.1.3)

* **Social learning demo (CLI)**: multi‑agent grid world with teaching, helping, cooperative monitoring
* **Knowledge transfer**: trust/novelty checks with high retention of shared values
* **Loop‑breaking pressure**: agents detect repetitive dynamics and force exploration
* **“Stuck” detection & help‑seeking**: agents ask nearby peers for help
* **Episodic surplus logging**: per‑step memory summaries for analysis
* **Proximity & energy models**: shared social detect radius and simple energy costs
* **Visualization**: auto‑saves `enhanced_social_qse_analysis.png` (trajectories, proximities, strategies, signals/relationships)

## 🆕 What’s new in v0.1.3

* Exposed **maze** and **extinction** demos via the CLI (`emile-mini maze`, `emile-mini extinction`).
* Refactored imports and broke circulars for packaged demos.
* Added `-V/--version` flag and updated docs/examples table.
* Packaging metadata tidy‑ups for TestPyPI.

---

## 📦 Installation

### From TestPyPI (current release)

```bash
pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple emile-mini==0.1.3
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
# v0.1.3: social demo (default: 3 agents, 120 steps, cluster radius 4)
emile-mini social

# customize
emile-mini social --steps 300 --agents 5 --cluster-radius 6

# additional CLI demos
emile-mini maze --steps 200 --size 12
emile-mini extinction --episodes 1 --phase-steps 120

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
├── maze_comparison.py          # QSE vs baseline utilities for maze demo
├── visual_maze_demo.py         # visual maze CLI/demo
├── extinction_experiment.py    # extinction/recovery CLI/demo
├── context.py
├── goal.py
├── memory.py
├── config.py
└── viz.py                      # plotting utilities
```

> In **v0.1.3**, the CLI exposes **social**, **maze**, and **extinction** demos. The rest of the research scripts remain source‑only.

---

## 🧪 Examples & demo scripts (from source)

Most research scripts live in the repository; two popular demos are accessible via the CLI, and the rest can be run from source.

* CLI equivalents: `emile-mini maze`, `emile-mini extinction`

| Script                               | What it does                                         | How to run                                  |
| ------------------------------------ | ---------------------------------------------------- | ------------------------------------------- |
| `visual_maze_demo.py`                | Visual maze demo with rich plots                     | `python visual_maze_demo.py`                |
| `run_maze_demo.py`                   | Minimal maze demo runner                             | `python run_maze_demo.py`                   |
| `extinction_experiment.py`           | Reward learning → extinction → recovery              | `python extinction_experiment.py`           |
| `maze_comparison.py`                 | QSE vs standard RL in a deceptive maze               | `python maze_comparison.py`                 |
| `fruit_categorization_experiment.py` | Embodied fruit categorization vs pattern‑matching RL | `python fruit_categorization_experiment.py` |
| `experiment_runner.py`               | Clean runner for single/paired experiments           | `python experiment_runner.py`               |
| `comprehensive_runner.py`            | Parameter sweeps & multi‑trial research runs         | `python comprehensive_runner.py`            |
| `definitive_validation.py`           | Consolidated validation plots/mechanisms             | `python definitive_validation.py`           |
| `complete_demo_suite.py`             | End‑to‑end showcase; can emit a final report         | `python complete_demo_suite.py`             |
| `cognitive_story.py`                 | Narrative analysis of an experiment                  | `python cognitive_story.py`                 |
| `concise_analyzer.py`                | Extract key findings from saved results              | `python concise_analyzer.py`                |
| `audit_emile_mini.py`                | Quick import/config audit harness                    | `python audit_emile_mini.py`                |

> Tips: most scripts accept flags (try `-h`). Results and figures are saved alongside the run unless the script specifies an output directory.

---

## 🎓 Method & research framing

* **MRP context.** *émile‑mini* is part of a Major Research Paper exploring how **enactive cognition** can be implemented and studied in artificial systems.
* **Computational autoethnography.** The project treats system‑building, debugging, and run logs as first‑person empirical material. The researcher is a participant in the process; design moves, failures, and reframings are data that connect architecture → behavior → interpretation.
* **Empirical semiotics (meaning‑as‑change).** We measure meaning by **observable reconfiguration** in the agent–environment system (e.g., proximity dynamics, memory traces, strategy shifts) produced by architectural impositions and interactions.
* **Relation to émile‑cogito.** *émile‑mini* packages a concrete slice of a broader platform (Émile/“émile‑cogito”) exploring multi‑module orchestration (QSE core; symbolic, context, goal, and memory stacks; orchestration patterns like RAO/EAO). Here, *émile‑mini* focuses on a reproducible **social learning** scenario for demos and analysis.

---

## 🗺️ Roadmap

* ✅ Expose maze + extinction demos via CLI (v0.1.3)
* Real PyPI release (CI trusted publishing)
* Verbosity flags & deterministic seeding
* More demos exposed via CLI (comparisons, fruit categorization)
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
