# émile-Mini: a lite enactive learner  
v0.1.3 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![University of Toronto](https://img.shields.io/badge/University%20of-Toronto-003F7F.svg)](https://www.utoronto.ca/)
[![Research](https://img.shields.io/badge/Type-Research-brightgreen.svg)](https://github.com)
 
**émile-Mini** is a lightweight, packaged demo of **enactive cognition** for research and teaching. It shows how intelligent behavior can emerge from an agent’s ongoing interaction with its environment—no static world model required.

* **Package name:** `emile-mini` (TestPyPI)
* **Import path:** `emile_mini`
* **CLI command:** `emile-mini`
* **Current CLI demos:** **social**, **maze**, **extinction** (cooperation/teaching; context‑switching; extinction/recovery)

---

## 🌱 What is enactive cognition?

Enactivism frames cognition as **meaning‑making through reciprocal environment-agent interaction**. Rather than building fixed internal representations, agents develop skills and interpret significance via sensorimotor engagement and qualia differentiation informed through context.

In **émile-Mini** this looks like:

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
<img width="5970" height="4718" alt="image" src="https://github.com/user-attachments/assets/b98adff9-8711-4af5-a629-bfaa4ea54e52" />

---

## 🆕 What else is new in v0.1.3?
* **Experimental Module Exposure Through the CLI**: Experimental modules place émile-Mini in competition with standard RL models to show how émile-Mini's enactive learning approach boosts performance beyond traditional reinforcement systems. The visuals below show the results of a maze challenge, where learners need to navigate a 2-d space to reach an objective, while avoiding traps disguised as ideal conditions. Results show how émile-Mini excels at challenges requiring adaptive decision making, changing context to avoid being caught in stasis or deceived by local optima. Context switching allows for émile-Mini to adapt policy to situation endogenously, offering a far more flexible set of actions than traditional RL models.

##    
### 🆕 Exposed: **emile-mini maze** demo via the CLI (`emile-mini maze`, `emile-mini extinction`)
<img width="4582" height="2550" alt="image" src="https://github.com/user-attachments/assets/5094b32c-75df-44f6-8b91-8d2ffc55bdb0" />
  
<img width="4582" height="2550" alt="image" src="https://github.com/user-attachments/assets/5d698423-e9d0-414f-8103-a4a7883861ec" />
  
---
##   
### 🆕 Exposed: **emile-mini extinction** demo via the CLI (`emile-mini extinction`)
* -> Key Innovation: Intrinsic QSE dynamics preserve knowledge without external rewards
* -> Complementary to: Context-switching for escaping local optima  
```
==================================================
EXTINCTION RESILIENCE ANALYSIS
==================================================
KNOWLEDGE PRESERVATION DURING EXTINCTION:
  QSE-Émile:   0.49% of pre-extinction knowledge preserved
  Standard RL: 0.00% of pre-extinction knowledge preserved
  → Advantage: 0.49% → (100% or infinite when Standard RL does not recover)

RECOVERY AFTER EXTINCTION:
  QSE-Émile:   114.80% recovery vs pre-extinction
  Standard RL: 0.00% recovery vs pre-extinction
  → Advantage: 114.80%

FINAL Q-VALUES:
  QSE-Émile:   0.0611
  Standard RL: 0.0219
```
  
---
-> **v0.1.3 Housekeeping**:
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
import numpy as np # Import numpy for potential calculations

# Run the social experiment simulation
env, agents, analysis = run_social_experiment(
    n_agents=5,
    steps=500,
    cluster_spawn=True,
    cluster_radius=4,
)
```
-> Then, examine the data collected by the simulation for further insight:

**--- Demo Insights ---** (example)
  
* -> **Final Agent Positions**
```
print("\n--- Simulation Insights ---")

# Display the final positions of the agents
print("\nFinal Agent Positions:")
for agent in agents:
    # Access the last position from the position_history
    if agent.position_history:
        final_position = agent.position_history[-1]
        print(f"{agent.agent_id}: {final_position}")
    else:
        print(f"{agent.agent_id}: No position history recorded")
```
  
* -> **Knowledge Gained and Embodied Cognition Mapping (social strategy)**
```
# Print insights into knowledge gained and behaviors (social strategy)
print("\nAgent Knowledge and Social Strategy:")
for agent in agents:
    print(f"\n{agent.agent_id}:")
    print(f"  Final Social Strategy: {agent.current_social_strategy}")

    # Show embodied mappings (knowledge gained)
    if agent.embodied_mappings:
        print(f"  Embodied Knowledge:")
        for category, values in agent.embodied_mappings.items():
            # Calculate and display a summary of learned knowledge for each category
            if values:
                 # Corrected: values is already a list, no need for .values()
                 avg_value = np.mean(values)
                 print(f"    {category}: learned about {len(values)} items, avg value={avg_value:.2f}")
            else:
                 print(f"    {category}: learned about 0 items")
    else:
        print("  Embodied Knowledge: None gained")
```

**--- Simulation Insights ---** (example)
```
Final Agent Positions:
Agent_0: (1, 1)
Agent_1: (3, 1)
Agent_2: (1, 1)
Agent_3: (1, 1)
Agent_4: (1, 1)

Agent Knowledge and Social Strategy:

Agent_0:
  Final Social Strategy: independent
  Embodied Knowledge:
    crimson_fruit: learned about 6 items, avg value=-0.90
    blue_fruit: learned about 1 items, avg value=0.35

Agent_1:
  Final Social Strategy: independent
  Embodied Knowledge:
    crimson_fruit: learned about 2 items, avg value=-0.90
    blue_fruit: learned about 2 items, avg value=0.35

... etc. (values for fruits will be identical, but the number of items learned will vary)

```
* You could add other insights here, for example:  
 -> Summary of actions taken (from **action_history**)  
 -> Analysis of social interactions (from **social_knowledge_transfer** or **social_memory**)  
---
  
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
  -> Programmatic options:
```.md
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
```
---

## 🎓 Method & research framing

* **MRP context.** *émile‑Mini* is part of a Major Research Paper exploring how **enactive cognition** can be implemented and studied in artificial systems.
* **Computational autoethnography** The project treats system‑building, debugging, and run logs as first‑person empirical material. The researcher is a participant in the process; design moves, failures, and reframings are data that connect architecture → behavior → interpretation.
* **Empirical semiotics (meaning‑as‑change).** We measure meaning by **observable reconfiguration** in the agent–environment system (e.g., proximity dynamics, memory traces, strategy shifts) produced by architectural impositions and interactions.
* **Enactive learning** through **social** multi-agent dynamics and **embodied** learning driven by a mechanistic feedback loop between the agent's physical state and its quantum-coupled cognitive core.
* **Relation to Émile‑cogito.** *émile‑Mini* packages a concrete slice of a broader platform (Émile‑cogito”) exploring multi‑module orchestration (QSE core; symbolic, context, goal, and memory stacks; orchestration patterns). Here, *émile‑Mini* focuses on reproducible **social** and **embodied** learning scenarios for demos and analysis.

---
  
## 📊 Early Insights & Validation

**Empirical results demonstrate strong performance across cognitive benchmarks:**

### Key Results
- **Dynamic Adaptation**: 100% vs 4.4% vs Epsilon-greedy Q-learning with goal switching (effect size: 0.956)
- **Meta-Cognitive Tasks**: 173.61 vs 131.02 vs Q-learning with fast adaptation (p < 0.001)
- **Statistical validation**: 4000+ trials across multiple conditions
- **Ablation studies**: Confirms quantum-symbolic coupling contribution

### Architectural Contributions
- Endogenous context switching for escaping local optima
- Bidirectional quantum-symbolic information flow
- Autopoietic self-maintenance without external rewards
- Emergent social learning with knowledge transfer
- Robust performance across architectural variations

### Benchmarks
- Context switching vs standard RL approaches
- Knowledge preservation during reward extinction  
- Social cognition emergence in multi-agent settings
- Embodied categorization through sensorimotor experience

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
