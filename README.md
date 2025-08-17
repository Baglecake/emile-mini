# émile-Mini: A Lite Enactive Learner
  
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Research](https://img.shields.io/badge/Research-MRP-orange.svg)](https://github.com/baglecake/emile-mini)


A lightweight, accessible implementation of enactive cognition principles for research and education. Émile-mini demonstrates how intelligent behavior emerges from the dynamic interaction between an agent and its environment, without pre-programmed knowledge or fixed behavioral rules.
  
## 🌱 What is Enactive Cognition?
Enactive cognition proposes that intelligence isn't about processing information, but about **making meaning through interaction**. Instead of representing the world internally, agents create understanding by engaging with their environment and developing skills through experience.
  
Émile-mini embodies these principles:
- **No pre-built knowledge** - the agent starts with only basic capacities
- **Learning through doing** - understanding emerges from interaction
- **Context-sensitive behavior** - the same situation can mean different things
- **Embodied experience** - cognition is grounded in sensorimotor activity
- **Emergent goals** - purposes arise from experience, not programming
  
## 🎯 Key Features
### 🔄 Dynamic Context Switching
The agent doesn't just learn responses - it learns to **recontextualize** situations, seeing the same environment in new ways as experience accumulates.
  
### 🧠 Memory Integration
- **Working Memory**: Immediate cognitive workspace
- **Episodic Memory**: Experiential narratives with context
- **Semantic Memory**: Emergent categorical knowledge
  
### 🎮 Goal Emergence
Goals aren't programmed - they emerge from interaction patterns and are refined through Q-learning based on experiential feedback.
### 🌍 Embodied Interaction
Agents exist in rich sensorimotor environments where they must navigate, discover objects, manage energy, and learn about their world through direct experience.
  
## 🚀 Quick Start
  
### Installation
```bash
git clone https://github.com/baglecake/emile-mini.git
cd emile-mini
pip install -e .
```
  
### Basic Usage
```python
from emile_mini import EmileAgent, EmbodiedEnvironment

# Create an enactive agent
agent = EmileAgent()

# Add some basic goals (these will evolve through experience)
for goal in ["explore", "maintain", "adapt"]:
agent.goal.add_goal(goal)

# Create an environment for embodied interaction
env = EmbodiedEnvironment(size=15)

# Run the enactive learning loop
for step in range(1000):
result = agent.embodied_step(env)
print(f"Step {step}: {result['action']} at {agent.body.state.position}")
```
  
### Run Example Experiments
```bash
# Basic cognitive simulation
python main.py

# Embodied sensorimotor experiment
python embodied_qse_emile.py

# Social learning demonstration
python social_qse_agent_v2.py

# Comprehensive research suite
python experiment_runner.py
```
## 📁 Project Structure
```
emile-mini/
├── agent.py # Main cognitive agent orchestrating all modules
├── qse_core.py # Quantum surplus emergence engine
├── symbolic.py # Symbolic reasoning and curvature calculation
├── context.py # Dynamic context switching system
├── goal.py # Emergent goal management with Q-learning
├── memory.py # Hierarchical memory system
├── config.py # Configuration parameters
├── embodied_qse_emile.py # Embodied sensorimotor cognition
├── social_qse_agent_v2.py# Social learning and interaction
├── experiment_runner.py # Research experiment suite
├── viz.py # Visualization utilities
└── main.py # Entry point
```
  
## 🧪 Experiments & Demonstrations
  
### 1. **Basic Cognitive Dynamics**
Observe how the agent develops stable cognitive patterns while maintaining flexibility:
  
```python
python main.py
```
  
### 2. **Embodied Learning**
Watch an agent navigate and learn about objects in a 2D world:
  
```python
python embodied_qse_emile.py
```
  
### 3. **Social Learning**
Multiple agents teaching and learning from each other:
  
```python
python social_qse_agent_v2.py
```
  
### 4. **Research Suite**
Comprehensive experiments for academic analysis:
  
```python
python experiment_runner.py
```
  
## 📊 What You'll Observe

### Emergent Behaviors
- **Context sensitivity**: Same stimuli → different responses based on experience
- **Goal evolution**: Initial goals get refined and new ones emerge
- **Memory integration**: Episodes connect into coherent narratives
- **Social learning**: Knowledge transfer between agents
- **Adaptive exploration**: Intelligent balance of exploration vs exploitation
  
### Research Metrics
- Context switching patterns and triggers
- Memory formation and retrieval dynamics
- Goal preference evolution over time
- Learning efficiency and knowledge retention
- Social interaction and knowledge transfer success
  
## 🔬 Research Applications

### Cognitive Science
- Study emergence of meaning through interaction
- Investigate context-dependent perception and action
- Analyze memory consolidation in dynamic systems
  
### AI Research
- Develop alternatives to representation-heavy AI
- Explore intrinsically motivated learning systems
- Research social cognition and knowledge sharing

### Education
- Demonstrate enactive principles with concrete examples
- Show how intelligence can emerge without programming
- Illustrate the role of embodiment in cognition
  
## 📖 Theoretical Background
Émile-mini implements insights from:
- **Enactivism** (Varela, Thompson, Rosch): Cognition as embodied action
- **Autopoiesis** (Maturana, Varela): Self-organizing systems
- **Empirical Semiotics**: Meaning through experiential grounding
- **Cognitive Development**: Learning as system reorganization
The technical implementation uses a novel Quantum Surplus Emergence (QSE) framework that creates the bidirectional dynamics necessary for enactive cognition, but the focus is on the cognitive principles, not the technical mechanism.

## 🤝 Contributing
This is a research project for computational cognitive science. Contributions welcome for:
- New experimental scenarios
- Analysis tools and visualizations
- Documentation improvements
- Educational examples
- Performance optimizations
  
## 📚 Citation
If you use Émile-mini in your research:
```bibtex
@software{emile_mini_2025,
author = {Coburn, Del},
title = {Émile-Mini: A Lite Enactive Learner},
url = {https://github.com/baglecake/emile-mini},
year = {2025}
}
```
## 📄 License
MIT License - See [LICENSE](LICENSE) for details.
## 🎓 Academic Context
This project is part of an MRP (Major Research Paper) in computational cognitive science, exploring how enactive principles can be implemented in artificial systems to create more robust, adaptive, and meaningful forms of artificial intelligence.

---
*"Intelligence is not about having the right representations, but about doing the right thing at the right time."* - Enactive Cognition Principle

