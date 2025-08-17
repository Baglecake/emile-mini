
"""
maze_environment.py - Deceptive maze environment for QSE-Émile context-switching demo

Creates mazes with local optima traps that cause standard RL to get stuck.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
from .config import CONFIG
from .agent import EmileAgent


class DeceptiveMaze:
    """Maze environment designed to trap RL agents in local optima"""

    def __init__(self, size=15):
        self.size = size
        self.start_pos = (1, 1)
        self.goal_pos = (size-2, size-2)
        self.maze = self._create_deceptive_maze()
        self.current_pos = self.start_pos
        self.step_count = 0
        self.max_steps = 1000

        # Track exploration for detecting stuck states
        self.position_visits = defaultdict(int)
        self.recent_positions = []
        self.stuck_threshold = 8  # If revisiting same area frequently

    def _create_deceptive_maze(self):
        """Create a maze with attractive dead ends (local optima)"""
        maze = np.ones((self.size, self.size))  # 1 = wall

        # Create basic paths
        maze[1:-1, 1:-1] = 0  # Open interior

        # Add walls to create structure
        for i in range(3, self.size-2, 4):
            maze[i, 3:-3] = 1  # Horizontal walls
            maze[3:-3, i] = 1  # Vertical walls

        # Create openings in walls
        for i in range(3, self.size-2, 4):
            maze[i, i] = 0  # Diagonal openings
            maze[i+1, i+2] = 0 if i+2 < self.size-1 else maze[i+1, i+2]

        # Add deceptive dead ends with small rewards
        self._add_deceptive_areas(maze)

        # Ensure start and goal are open
        maze[self.start_pos] = 0
        maze[self.goal_pos] = 0

        return maze

    def _add_deceptive_areas(self, maze):
        """Add areas that look promising but are dead ends"""
        # Create attractive dead ends in corners
        corner_positions = [
            (3, 3), (3, self.size-4),
            (self.size-4, 3), (7, 7)
        ]

        for pos in corner_positions:
            if pos[0] < self.size-1 and pos[1] < self.size-1:
                # Create small open area
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        x, y = pos[0] + dx, pos[1] + dy
                        if 0 < x < self.size-1 and 0 < y < self.size-1:
                            maze[x, y] = 0

        # Block some exits to make them dead ends
        if self.size > 10:
            maze[4, 3] = 1  # Block exit from first attractive area
            maze[3, self.size-3] = 1  # Block another exit

    def reset(self):
        """Reset environment to initial state"""
        self.current_pos = self.start_pos
        self.step_count = 0
        self.position_visits.clear()
        self.recent_positions = []
        return self._get_state()

    def step(self, action):
        """Take action in environment"""
        # Actions: 0=up, 1=down, 2=left, 3=right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        if action < len(moves):
            dx, dy = moves[action]
            new_x = self.current_pos[0] + dx
            new_y = self.current_pos[1] + dy

            # Check bounds and walls
            if (0 <= new_x < self.size and 0 <= new_y < self.size and
                self.maze[new_x, new_y] == 0):
                self.current_pos = (new_x, new_y)

        # Update tracking
        self.position_visits[self.current_pos] += 1
        self.recent_positions.append(self.current_pos)
        if len(self.recent_positions) > 20:
            self.recent_positions.pop(0)

        self.step_count += 1

        # Calculate reward
        reward = self._calculate_reward()

        # Check if done
        done = (self.current_pos == self.goal_pos or
                self.step_count >= self.max_steps)

        return self._get_state(), reward, done

    def _get_state(self):
        """Get current state observation"""
        # Simple state: (x, y, distance_to_goal)
        dx = self.goal_pos[0] - self.current_pos[0]
        dy = self.goal_pos[1] - self.current_pos[1]
        distance = np.sqrt(dx*dx + dy*dy)

        return np.array([
            self.current_pos[0] / self.size,  # Normalized position
            self.current_pos[1] / self.size,
            distance / (self.size * 1.414),   # Normalized distance
        ])

    def _calculate_reward(self):
        """Calculate reward with deceptive local optima"""
        # Large reward for reaching goal
        if self.current_pos == self.goal_pos:
            return 100.0

        # Small negative reward for each step (encourages efficiency)
        reward = -0.1

        # Deceptive rewards in local optima areas
        deceptive_areas = [(3, 3), (3, self.size-4), (self.size-4, 3), (7, 7)]
        for area in deceptive_areas:
            if area[0] < self.size and area[1] < self.size:
                dist_to_area = abs(self.current_pos[0] - area[0]) + abs(self.current_pos[1] - area[1])
                if dist_to_area <= 2:
                    reward += 0.5  # Small attractive reward

        # Penalty for revisiting same position too much
        if self.position_visits[self.current_pos] > 3:
            reward -= 0.2

        return reward

    def is_stuck(self):
        """Detect if agent is stuck in local area"""
        if len(self.recent_positions) < self.stuck_threshold:
            return False

        # Check if agent is cycling in small area
        recent_unique = set(self.recent_positions[-self.stuck_threshold:])
        return len(recent_unique) <= 3  # Cycling among 3 or fewer positions

    def render(self, show_path=False):
        """Visualize the maze and agent position"""
        plt.figure(figsize=(8, 8))

        # Show maze
        display_maze = self.maze.copy()

        # Mark start, current, and goal positions
        display_maze[self.start_pos] = 0.3  # Start (light gray)
        display_maze[self.current_pos] = 0.6  # Current (medium gray)
        display_maze[self.goal_pos] = 0.9  # Goal (dark gray)

        plt.imshow(display_maze, cmap='viridis')
        plt.title(f'Deceptive Maze - Step {self.step_count}')
        plt.colorbar(label='0=path, 1=wall, 0.3=start, 0.6=current, 0.9=goal')

        if show_path and len(self.recent_positions) > 1:
            # Show recent path
            path_x = [pos[1] for pos in self.recent_positions]
            path_y = [pos[0] for pos in self.recent_positions]
            plt.plot(path_x, path_y, 'r-', alpha=0.7, linewidth=2, label='Recent path')
            plt.legend()

        plt.show()


class QSEMazeAgent(EmileAgent):
    """QSE-Émile agent adapted for maze navigation"""

    def __init__(self, config=CONFIG):
        super().__init__(config)

        # Add navigation-specific goals
        nav_goals = ["explore", "exploit", "approach_goal", "escape_area", "revisit"]
        for goal in nav_goals:
            self.goal.add_goal(goal)

        # Tracking for analysis
        self.action_history = []
        self.context_switches = []
        self.stuck_episodes = []
        self.breakthrough_moments = []

    def select_action(self, state, env):
        """Convert cognitive state to environment action"""
        # Run cognitive step
        cognitive_metrics = self.step(dt=0.01)

        # Check for context switch
        current_context = self.context.get_current()
        if (len(self.context_switches) == 0 or
            current_context != self.context_switches[-1]['context']):
            self.context_switches.append({
                'step': len(self.action_history),
                'context': current_context,
                'position': env.current_pos,
                'was_stuck': env.is_stuck()
            })

        # Convert current goal to action
        current_goal = self.goal.current_goal
        action = self._goal_to_action(current_goal, state, env)

        self.action_history.append({
            'action': action,
            'goal': current_goal,
            'context': current_context,
            'surplus_mean': np.mean(self.qse.S),
            'position': env.current_pos,
            'stuck': env.is_stuck()
        })

        return action

    def _goal_to_action(self, goal, state, env):
        """Convert goal to maze action"""
        if goal == "approach_goal":
            # Move toward goal
            dx = env.goal_pos[0] - env.current_pos[0]
            dy = env.goal_pos[1] - env.current_pos[1]

            if abs(dx) > abs(dy):
                return 1 if dx > 0 else 0  # down/up
            else:
                return 3 if dy > 0 else 2  # right/left

        elif goal == "explore":
            # Move to less visited areas
            least_visited = None
            min_visits = float('inf')

            for action in range(4):
                moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                dx, dy = moves[action]
                new_pos = (env.current_pos[0] + dx, env.current_pos[1] + dy)

                if (0 <= new_pos[0] < env.size and 0 <= new_pos[1] < env.size and
                    env.maze[new_pos] == 0):
                    visits = env.position_visits[new_pos]
                    if visits < min_visits:
                        min_visits = visits
                        least_visited = action

            return least_visited if least_visited is not None else np.random.randint(4)

        elif goal == "escape_area":
            # Try to move away from current area
            return np.random.randint(4)

        else:
            # Default: random exploration
            return np.random.randint(4)

    def analyze_performance(self, env):
        """Analyze context switching vs breakthrough performance"""
        results = {
            'total_steps': len(self.action_history),
            'context_switches': len(self.context_switches),
            'reached_goal': env.current_pos == env.goal_pos,
            'final_distance': np.sqrt(
                (env.current_pos[0] - env.goal_pos[0])**2 +
                (env.current_pos[1] - env.goal_pos[1])**2
            )
        }

        # Analyze breakthrough moments (escaping stuck states)
        breakthroughs = []
        for i, switch in enumerate(self.context_switches):
            if switch['was_stuck']:
                # Find if agent escaped after this context switch
                escape_step = None
                for j in range(switch['step'], min(switch['step'] + 20, len(self.action_history))):
                    if not self.action_history[j]['stuck']:
                        escape_step = j
                        break

                if escape_step:
                    breakthroughs.append({
                        'switch_step': switch['step'],
                        'escape_step': escape_step,
                        'escape_time': escape_step - switch['step']
                    })

        results['breakthroughs'] = breakthroughs
        results['breakthrough_rate'] = len(breakthroughs) / max(1, len(self.context_switches))

        return results


def run_maze_demo():
    """Run the context-switching maze demonstration"""
    print("QSE-ÉMILE CONTEXT-SWITCHING MAZE DEMO")
    print("=" * 40)

    # Create environment and agent
    maze = DeceptiveMaze(size=15)
    agent = QSEMazeAgent()

    print("Running maze navigation...")
    state = maze.reset()
    done = False

    while not done:
        action = agent.select_action(state, maze)
        state, reward, done = maze.step(action)

        # Provide reward feedback
        agent.goal.feedback(reward)

        # Optional: show progress every 100 steps
        if maze.step_count % 100 == 0:
            print(f"Step {maze.step_count}: Position {maze.current_pos}, "
                  f"Context switches: {len(agent.context_switches)}")

    # Analyze results
    results = agent.analyze_performance(maze)

    print(f"\n=== RESULTS ===")
    print(f"Total steps: {results['total_steps']}")
    print(f"Reached goal: {results['reached_goal']}")
    print(f"Final distance to goal: {results['final_distance']:.2f}")
    print(f"Context switches: {results['context_switches']}")
    print(f"Breakthrough episodes: {len(results['breakthroughs'])}")
    print(f"Breakthrough rate: {results['breakthrough_rate']:.2f}")

    # Visualize final state
    maze.render(show_path=True)

    return maze, agent, results


if __name__ == "__main__":
    maze, agent, results = run_maze_demo()
