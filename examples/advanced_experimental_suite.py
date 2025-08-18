
"""
Advanced Experimental Suite for QSE-Émile

Tests the unique capabilities of the cognitive architecture against meaningful baselines:
1. Dynamic Environment Adaptation
2. Long-term Intrinsic Learning
3. Meta-Cognitive Benchmarks
4. Real-World Supply Chain Problem
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time
from pathlib import Path
from agent import EmileAgent  
from config import QSEConfig
import json

# ==================== 1. DYNAMIC ENVIRONMENT ADAPTATION ====================

class MorphingMaze:
    """Maze that changes structure during navigation"""

    def __init__(self, size=15, morph_interval=50):
        self.size = size
        self.morph_interval = morph_interval
        self.step_count = 0
        self.current_config = 0

        # Define multiple maze configurations
        self.configurations = self._create_maze_configs()
        self.maze = self.configurations[0]

        self.start_pos = (1, 1)
        self.goal_pos = (size-2, size-2)
        self.current_pos = self.start_pos

    def _create_maze_configs(self):
        """Create different maze configurations"""
        configs = []

        for config_id in range(3):
            maze = np.ones((self.size, self.size))
            maze[1:-1, 1:-1] = 0  # Open interior

            # Add configuration-specific walls
            if config_id == 0:
                # Horizontal corridors
                for i in range(3, self.size-2, 4):
                    maze[i, 3:-3] = 1
                    maze[i, 7] = 0  # Opening

            elif config_id == 1:
                # Vertical corridors
                for i in range(3, self.size-2, 4):
                    maze[3:-3, i] = 1
                    maze[7, i] = 0  # Opening

            elif config_id == 2:
                # Diagonal pattern
                for i in range(2, self.size-2, 3):
                    for j in range(2, self.size-2, 3):
                        if (i + j) % 6 == 0:
                            maze[i, j] = 1

            configs.append(maze)

        return configs

    def step(self, action):
        """Take action and potentially morph maze"""

        # Move agent
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        if action < len(moves):
            dx, dy = moves[action]
            new_x = self.current_pos[0] + dx
            new_y = self.current_pos[1] + dy

            if (0 <= new_x < self.size and 0 <= new_y < self.size and
                self.maze[new_x, new_y] == 0):
                self.current_pos = (new_x, new_y)

        self.step_count += 1

        # Check for morphing
        if self.step_count % self.morph_interval == 0:
            self._morph_maze()

        # Calculate reward
        reward = self._calculate_reward()
        done = (self.current_pos == self.goal_pos)

        return self._get_state(), reward, done

    def _morph_maze(self):
        """Change maze configuration"""
        old_config = self.current_config
        self.current_config = (self.current_config + 1) % len(self.configurations)
        self.maze = self.configurations[self.current_config]

        # Make sure agent isn't stuck in a wall
        if self.maze[self.current_pos] == 1:
            self.current_pos = self._find_nearest_open_space()

        print(f"🔄 Maze morphed: Config {old_config} -> {self.current_config} at step {self.step_count}")

    def _find_nearest_open_space(self):
        """Find nearest open space if agent gets stuck in wall"""
        for radius in range(1, self.size):
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    x = self.current_pos[0] + dx
                    y = self.current_pos[1] + dy
                    if (0 <= x < self.size and 0 <= y < self.size and
                        self.maze[x, y] == 0):
                        return (x, y)
        return self.start_pos  # Fallback

    def _get_state(self):
        return np.array([
            self.current_pos[0] / self.size,
            self.current_pos[1] / self.size,
            self.current_config / len(self.configurations)
        ])

    def _calculate_reward(self):
        if self.current_pos == self.goal_pos:
            return 100.0

        # Distance-based reward
        dist = np.sqrt((self.current_pos[0] - self.goal_pos[0])**2 +
                      (self.current_pos[1] - self.goal_pos[1])**2)
        return -0.1 - 0.01 * dist

    def reset(self):
        self.current_pos = self.start_pos
        self.step_count = 0
        self.current_config = 0
        self.maze = self.configurations[0]
        return self._get_state()


class ContextSensitiveGoals:
    """Environment where goal meanings change based on context"""

    def __init__(self):
        self.step_count = 0
        self.context_switches = 0
        self.current_context = "exploration"
        self.contexts = ["exploration", "exploitation", "conservation"]

        # Context-dependent goal meanings
        self.goal_meanings = {
            "exploration": {"explore": 1.0, "exploit": -0.5, "maintain": 0.2},
            "exploitation": {"explore": -0.2, "exploit": 1.0, "maintain": 0.1},
            "conservation": {"explore": -0.3, "exploit": -0.3, "maintain": 1.0}
        }

    def step(self, goal_selected):
        """Reward based on context-appropriate goal selection"""

        # Context switches every 100 steps
        if self.step_count > 0 and self.step_count % 100 == 0:
            old_context = self.current_context
            self.current_context = np.random.choice(self.contexts)
            if old_context != self.current_context:
                self.context_switches += 1
                print(f"🔄 Context switch: {old_context} -> {self.current_context}")

        # Calculate reward based on current context
        reward = 0.0
        if goal_selected in self.goal_meanings[self.current_context]:
            reward = self.goal_meanings[self.current_context][goal_selected]

        self.step_count += 1
        return reward, self.current_context


# ==================== 2. LONG-TERM INTRINSIC LEARNING ====================

class ExtendedExtinctionExperiment:
    """Multi-cycle extinction-recovery with long intrinsic periods"""

    def __init__(self, agent, learning_phase=200, extinction_phase=300, cycles=3):
        self.agent = agent
        self.learning_phase = learning_phase
        self.extinction_phase = extinction_phase
        self.cycles = cycles

    def run(self):
        """Run multiple extinction-recovery cycles"""

        results = {
            'cycles': [],
            'q_value_evolution': [],
            'intrinsic_behaviors': [],
            'recovery_rates': []
        }

        for cycle in range(self.cycles):
            print(f"\n=== Extinction Cycle {cycle + 1}/{self.cycles} ===")

            cycle_results = self._run_single_cycle()
            results['cycles'].append(cycle_results)

            # Analyze recovery rate
            if cycle > 0:
                recovery_rate = self._calculate_recovery_rate(
                    results['cycles'][cycle-1], cycle_results
                )
                results['recovery_rates'].append(recovery_rate)

        return results

    def _run_single_cycle(self):
        """Run one learning -> extinction -> recovery cycle"""

        cycle_data = {
            'learning': [],
            'extinction': [],
            'recovery': []
        }

        # Learning phase
        print("📚 Learning phase...")
        for t in range(self.learning_phase):
            reward = 0.8 if t % 20 == 0 else 0.0
            ext_input = {'reward': reward} if reward > 0 else None
            metrics = self.agent.step(dt=0.01, external_input=ext_input)

            if t % 20 == 0:  # Sample periodically
                cycle_data['learning'].append({
                    'step': t,
                    'q_values': dict(self.agent.goal.q_values),
                    'surplus_mean': np.mean(self.agent.qse.S),
                    'context': self.agent.context.get_current()
                })

        # Extinction phase
        print("💀 Extinction phase...")
        for t in range(self.extinction_phase):
            metrics = self.agent.step(dt=0.01)  # No rewards

            if t % 30 == 0:  # Sample periodically
                cycle_data['extinction'].append({
                    'step': t,
                    'q_values': dict(self.agent.goal.q_values),
                    'surplus_mean': np.mean(self.agent.qse.S),
                    'context': self.agent.context.get_current(),
                    'goal_selected': self.agent.goal.current_goal
                })

        # Recovery phase
        print("🔄 Recovery phase...")
        for t in range(self.learning_phase // 2):
            reward = 0.8 if t % 20 == 0 else 0.0
            ext_input = {'reward': reward} if reward > 0 else None
            metrics = self.agent.step(dt=0.01, external_input=ext_input)

            if t % 20 == 0:
                cycle_data['recovery'].append({
                    'step': t,
                    'q_values': dict(self.agent.goal.q_values),
                    'surplus_mean': np.mean(self.agent.qse.S),
                    'context': self.agent.context.get_current()
                })

        return cycle_data

    def _calculate_recovery_rate(self, prev_cycle, current_cycle):
        """Calculate how quickly the agent recovers compared to previous cycle"""

        if not prev_cycle['recovery'] or not current_cycle['recovery']:
            return 0.0

        prev_final_q = max(prev_cycle['recovery'][-1]['q_values'].values())
        curr_final_q = max(current_cycle['recovery'][-1]['q_values'].values())

        return curr_final_q / max(prev_final_q, 0.001)  # Avoid division by zero


# ==================== 3. META-COGNITIVE BENCHMARKS ====================

class SimpleHierarchicalAgent:
    """Simple hierarchical RL baseline for comparison"""

    def __init__(self, high_level_goals=None):
        self.high_level_goals = high_level_goals or ["explore", "exploit", "maintain"]
        self.current_high_goal = "explore"
        self.high_level_q = {goal: 0.0 for goal in self.high_level_goals}
        self.low_level_q = defaultdict(lambda: defaultdict(float))

        self.epsilon = 0.1
        self.learning_rate = 0.1
        self.goal_switch_interval = 50
        self.step_count = 0

    def select_action(self, state, environment):
        """Hierarchical action selection"""

        # High-level goal selection (periodic)
        if self.step_count % self.goal_switch_interval == 0:
            if np.random.random() < self.epsilon:
                self.current_high_goal = np.random.choice(self.high_level_goals)
            else:
                best_goal = max(self.high_level_q.items(), key=lambda x: x[1])[0]
                self.current_high_goal = best_goal

        # Low-level action selection based on current high-level goal
        state_key = tuple(np.round(state, 2))  # Discretize state

        if np.random.random() < self.epsilon:
            action = np.random.randint(4)
        else:
            q_values = self.low_level_q[self.current_high_goal][state_key]
            if isinstance(q_values, dict):
                action = max(q_values.items(), key=lambda x: x[1])[0] if q_values else 0
            else:
                action = np.random.randint(4)

        self.step_count += 1
        return action

    def update(self, state, action, reward, next_state, done):
        """Update both levels"""

        state_key = tuple(np.round(state, 2))

        # Update low-level Q-values
        if not isinstance(self.low_level_q[self.current_high_goal][state_key], dict):
            self.low_level_q[self.current_high_goal][state_key] = {}

        current_q = self.low_level_q[self.current_high_goal][state_key].get(action, 0.0)
        self.low_level_q[self.current_high_goal][state_key][action] = current_q + self.learning_rate * (reward - current_q)

        # Update high-level Q-values (less frequently)
        if self.step_count % self.goal_switch_interval == 0:
            self.high_level_q[self.current_high_goal] += self.learning_rate * reward


class SimpleMetaLearner:
    """Simple meta-learning baseline"""

    def __init__(self, adaptation_steps=10):
        self.adaptation_steps = adaptation_steps
        self.base_q = defaultdict(float)
        self.adapted_q = defaultdict(float)
        self.adaptation_history = []
        self.current_task_experience = []

    def adapt_to_new_task(self, initial_experiences):
        """Quick adaptation based on few examples"""

        # Reset adapted Q-values to base
        self.adapted_q = self.base_q.copy()

        # Fast adaptation on initial experiences
        for state, action, reward, next_state in initial_experiences[-self.adaptation_steps:]:
            state_action = (tuple(np.round(state, 2)), action)
            current_q = self.adapted_q[state_action]
            # Fast learning rate for adaptation
            self.adapted_q[state_action] = current_q + 0.5 * (reward - current_q)

    def select_action(self, state):
        state_tuple = tuple(np.round(state, 2))

        # Epsilon-greedy with adapted Q-values
        if np.random.random() < 0.1:
            return np.random.randint(4)
        else:
            best_action = 0
            best_q = float('-inf')
            for action in range(4):
                q_val = self.adapted_q.get((state_tuple, action), 0.0)
                if q_val > best_q:
                    best_q = q_val
                    best_action = action
            return best_action


# ==================== 4. SUPPLY CHAIN SIMULATION ====================

class SupplyChainEnvironment:
    """Multi-objective supply chain management simulation"""

    def __init__(self):
        self.inventory = 50.0
        self.customer_demand = 0.0
        self.supplier_reliability = 1.0
        self.step_count = 0

        # Disruption schedule
        self.disruptions = [
            (100, 150, 0.3),  # Steps 100-150: 30% reliability
            (300, 400, 0.1),  # Steps 300-400: 10% reliability
            (600, 700, 0.0),  # Steps 600-700: 0% reliability (total disruption)
        ]

    def step(self, action_dict):
        """
        action_dict should contain goal selections like:
        {"primary_goal": "maintain", "secondary_actions": {...}}
        """

        # Update environment state
        self._update_demand()
        self._update_supplier_reliability()

        # Calculate rewards based on actions
        rewards = self._calculate_rewards(action_dict)

        # Update inventory based on actions and environment
        self._update_inventory(action_dict)

        self.step_count += 1

        state = self._get_state()
        done = self.step_count >= 800

        return state, rewards, done

    def _update_demand(self):
        """Update customer demand (varies over time)"""
        base_demand = 5.0
        seasonal = 2.0 * np.sin(self.step_count * 2 * np.pi / 100)
        noise = np.random.normal(0, 1)
        self.customer_demand = max(0, base_demand + seasonal + noise)

    def _update_supplier_reliability(self):
        """Update supplier reliability based on disruption schedule"""
        self.supplier_reliability = 1.0  # Default

        for start, end, reliability in self.disruptions:
            if start <= self.step_count <= end:
                self.supplier_reliability = reliability
                break

    def _calculate_rewards(self, action_dict):
        """Calculate multi-objective rewards"""

        goal = action_dict.get("primary_goal", "maintain")

        rewards = {}

        # Service level reward (meeting demand)
        if self.inventory >= self.customer_demand:
            rewards["service"] = 1.0
        else:
            rewards["service"] = -2.0 * (self.customer_demand - self.inventory)

        # Inventory cost (holding too much is expensive)
        rewards["inventory_cost"] = -0.01 * max(0, self.inventory - 30)

        # Goal-specific rewards
        if goal == "explore":
            # Reward for trying new suppliers during disruptions
            if self.supplier_reliability < 0.5:
                rewards["exploration"] = 0.5
            else:
                rewards["exploration"] = -0.1

        elif goal == "exploit":
            # Reward for using reliable suppliers efficiently
            if self.supplier_reliability > 0.8:
                rewards["exploitation"] = 0.3
            else:
                rewards["exploitation"] = -0.3

        elif goal == "maintain":
            # Reward for keeping stable inventory
            if 20 <= self.inventory <= 60:
                rewards["maintenance"] = 0.4
            else:
                rewards["maintenance"] = -0.2

        # Total reward
        rewards["total"] = sum(rewards.values())

        return rewards

    def _update_inventory(self, action_dict):
        """Update inventory based on actions"""

        goal = action_dict.get("primary_goal", "maintain")

        # Fulfill demand (reduce inventory)
        self.inventory -= min(self.inventory, self.customer_demand)

        # Restock based on goal and supplier reliability
        if goal == "exploit" and self.supplier_reliability > 0.5:
            # Aggressive restocking
            restock = 8.0 * self.supplier_reliability
        elif goal == "maintain":
            # Conservative restocking
            restock = 5.0 * self.supplier_reliability
        elif goal == "explore":
            # Variable restocking (simulating trying different suppliers)
            restock = np.random.uniform(3, 7) * max(0.2, self.supplier_reliability)
        else:
            restock = 3.0 * self.supplier_reliability

        self.inventory = min(100, self.inventory + restock)  # Max capacity 100

    def _get_state(self):
        """Get current state"""
        return {
            "inventory_level": self.inventory / 100.0,
            "demand_level": self.customer_demand / 10.0,
            "supplier_reliability": self.supplier_reliability,
            "step": self.step_count / 800.0
        }

    def reset(self):
        self.inventory = 50.0
        self.customer_demand = 0.0
        self.supplier_reliability = 1.0
        self.step_count = 0
        return self._get_state()


# ==================== EXPERIMENTAL RUNNER ====================

class AdvancedExperimentalSuite:
    """Comprehensive experimental suite for QSE-Émile"""

    def __init__(self, save_dir="advanced_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def run_all_experiments(self):
        """Run all four experimental categories"""

        print("🚀 ADVANCED QSE-ÉMILE EXPERIMENTAL SUITE")
        print("=" * 60)

        results = {}

        # 1. Dynamic Environment Adaptation
        print("\n1️⃣ DYNAMIC ENVIRONMENT ADAPTATION")
        results['dynamic_adaptation'] = self._test_dynamic_adaptation()

        # 2. Long-term Intrinsic Learning
        print("\n2️⃣ LONG-TERM INTRINSIC LEARNING")
        results['intrinsic_learning'] = self._test_intrinsic_learning()

        # 3. Meta-Cognitive Benchmarks
        print("\n3️⃣ META-COGNITIVE BENCHMARKS")
        results['meta_cognitive'] = self._test_meta_cognitive()

        # 4. Supply Chain Simulation
        print("\n4️⃣ SUPPLY CHAIN SIMULATION")
        results['supply_chain'] = self._test_supply_chain()

        # Save results
        self._save_results(results)

        # Generate summary
        self._generate_summary(results)

        return results

    def _test_dynamic_adaptation(self):
        """Test dynamic environment adaptation"""

        print("Testing morphing maze navigation...")

        # QSE-Émile agent
        emile_agent = EmileAgent()
        for goal in ["explore", "exploit", "maintain", "adapt"]:
            emile_agent.goal.add_goal(goal)

        # Hierarchical baseline
        hierarchical_agent = SimpleHierarchicalAgent()

        results = {"emile": [], "hierarchical": []}

        for agent_type, agent in [("emile", emile_agent), ("hierarchical", hierarchical_agent)]:
            print(f"  Testing {agent_type} agent...")

            maze = MorphingMaze(size=12, morph_interval=40)
            state = maze.reset()
            done = False
            step_count = 0
            total_reward = 0
            context_switches = 0

            while not done and step_count < 300:
                if agent_type == "emile":
                    # Get context before step
                    old_context = agent.context.get_current()
                    action = self._emile_to_action(agent, state, maze)
                    new_context = agent.context.get_current()
                    if old_context != new_context:
                        context_switches += 1
                else:
                    action = agent.select_action(state, maze)

                next_state, reward, done = maze.step(action)
                total_reward += reward

                if agent_type == "emile":
                    agent.goal.feedback(reward)
                else:
                    agent.update(state, action, reward, next_state, done)

                state = next_state
                step_count += 1

            results[agent_type] = {
                "success": maze.current_pos == maze.goal_pos,
                "steps": step_count,
                "total_reward": total_reward,
                "context_switches": context_switches,
                "final_distance": np.sqrt((maze.current_pos[0] - maze.goal_pos[0])**2 +
                                         (maze.current_pos[1] - maze.goal_pos[1])**2)
            }

        return results

    def _test_intrinsic_learning(self):
        """Test long-term intrinsic learning"""

        print("Testing multi-cycle extinction-recovery...")

        agent = EmileAgent()
        for goal in ["explore", "exploit", "maintain", "adapt", "consolidate"]:
            agent.goal.add_goal(goal)

        experiment = ExtendedExtinctionExperiment(
            agent, learning_phase=150, extinction_phase=200, cycles=3
        )

        return experiment.run()

    def _test_meta_cognitive(self):
        """Test against meta-cognitive baselines"""

        print("Testing against meta-learning baseline...")

        # Simple task: context-sensitive goals
        goal_env = ContextSensitiveGoals()

        # QSE-Émile
        emile_agent = EmileAgent()
        for goal in ["explore", "exploit", "maintain"]:
            emile_agent.goal.add_goal(goal)

        # Meta-learning baseline
        meta_agent = SimpleMetaLearner()

        results = {"emile": [], "meta_learner": []}

        for agent_type, agent in [("emile", emile_agent), ("meta_learner", meta_agent)]:
            print(f"  Testing {agent_type}...")

            total_reward = 0
            context_adaptations = 0

            for step in range(400):
                if agent_type == "emile":
                    metrics = agent.step(dt=0.01)
                    goal_selected = agent.goal.current_goal
                else:
                    # Simplified for meta-learner
                    goal_selected = np.random.choice(["explore", "exploit", "maintain"])

                reward, current_context = goal_env.step(goal_selected)
                total_reward += reward

                if agent_type == "emile":
                    agent.goal.feedback(reward)

            results[agent_type] = {
                "total_reward": total_reward,
                "context_switches": goal_env.context_switches,
                "adaptation_score": total_reward / max(1, goal_env.context_switches)
            }

        return results

    def _test_supply_chain(self):
        """Test supply chain management"""

        print("Testing supply chain disruption management...")

        agent = EmileAgent()
        for goal in ["explore", "exploit", "maintain", "adapt"]:
            agent.goal.add_goal(goal)

        env = SupplyChainEnvironment()
        state = env.reset()

        results = {
            "total_rewards": defaultdict(list),
            "inventory_levels": [],
            "service_levels": [],
            "disruption_responses": []
        }

        done = False
        while not done:
            # Agent selects goal
            metrics = agent.step(dt=0.01)
            goal_selected = agent.goal.current_goal

            # Environment step
            action_dict = {"primary_goal": goal_selected}
            next_state, rewards, done = env.step(action_dict)

            # Feedback to agent
            agent.goal.feedback(rewards["total"])

            # Record results
            for reward_type, reward_value in rewards.items():
                results["total_rewards"][reward_type].append(reward_value)

            results["inventory_levels"].append(env.inventory)
            results["service_levels"].append(1.0 if env.inventory >= env.customer_demand else 0.0)

            # Track responses during disruptions
            if env.supplier_reliability < 0.5:
                results["disruption_responses"].append({
                    "step": env.step_count,
                    "goal": goal_selected,
                    "reliability": env.supplier_reliability,
                    "inventory": env.inventory,
                    "reward": rewards["total"]
                })

            state = next_state

        return results

    def _emile_to_action(self, agent, state, maze):
        """Convert Émile agent's goal to maze action"""
        metrics = agent.step(dt=0.01)
        goal = agent.goal.current_goal

        if goal == "explore":
            # Move to less visited areas (simplified)
            return np.random.randint(4)
        elif goal == "exploit":
            # Move toward goal
            dx = maze.goal_pos[0] - maze.current_pos[0]
            dy = maze.goal_pos[1] - maze.current_pos[1]
            if abs(dx) > abs(dy):
                return 1 if dx > 0 else 0
            else:
                return 3 if dy > 0 else 2
        else:
            # Default exploration
            return np.random.randint(4)

    def _save_results(self, results):
        """Save experimental results"""
        timestamp = int(time.time())
        filename = f"advanced_results_{timestamp}.json"

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj

        serializable_results = convert_numpy(results)

        with open(self.save_dir / filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"💾 Results saved to {filename}")

    def _generate_summary(self, results):
        """Generate experimental summary"""

        print(f"\n📊 EXPERIMENTAL SUMMARY")
        print("=" * 50)

        # Dynamic adaptation summary
        if 'dynamic_adaptation' in results:
            da = results['dynamic_adaptation']
            print(f"🔄 Dynamic Adaptation:")
            print(f"  Émile: {'SUCCESS' if da['emile']['success'] else 'FAILED'} ({da['emile']['steps']} steps)")
            print(f"  Hierarchical: {'SUCCESS' if da['hierarchical']['success'] else 'FAILED'} ({da['hierarchical']['steps']} steps)")
            print(f"  Context Switches: {da['emile']['context_switches']}")

        # Intrinsic learning summary
        if 'intrinsic_learning' in results:
            il = results['intrinsic_learning']
            print(f"🧠 Intrinsic Learning:")
            print(f"  Completed {len(il['cycles'])} extinction cycles")
            if il['recovery_rates']:
                avg_recovery = np.mean(il['recovery_rates'])
                print(f"  Average recovery rate: {avg_recovery:.3f}")

        # Meta-cognitive summary
        if 'meta_cognitive' in results:
            mc = results['meta_cognitive']
            print(f"🎯 Meta-Cognitive:")
            print(f"  Émile reward: {mc['emile']['total_reward']:.2f}")
            print(f"  Meta-learner reward: {mc['meta_learner']['total_reward']:.2f}")
            print(f"  Émile adaptation score: {mc['emile']['adaptation_score']:.3f}")

        # Supply chain summary
        if 'supply_chain' in results:
            sc = results['supply_chain']
            avg_service = np.mean(sc['service_levels'])
            print(f"📦 Supply Chain:")
            print(f"  Average service level: {avg_service:.3f}")
            print(f"  Disruption responses: {len(sc['disruption_responses'])}")


def main():
    """Run the advanced experimental suite"""

    suite = AdvancedExperimentalSuite()
    results = suite.run_all_experiments()

    print(f"\n🎉 ADVANCED EXPERIMENTS COMPLETE!")
    print(f"Check the 'advanced_results' directory for detailed data.")

    return results


if __name__ == "__main__":
    main()
