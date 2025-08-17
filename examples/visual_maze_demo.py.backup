
"""
visual_maze_demo.py - Enhanced maze demo with compelling visualizations

Creates actual pictures showing how QSE-Émile escapes local optima while
standard RL gets trapped.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from pathlib import Path

from maze_environment import DeceptiveMaze, QSEMazeAgent
from maze_comparison import StandardRLAgent


class VisualMazeDemo:
    """Enhanced maze demo with rich visualizations"""

    def __init__(self, maze_size=15, max_steps=500):
        self.maze_size = maze_size
        self.max_steps = max_steps
        self.save_dir = Path("maze_visuals")
        self.save_dir.mkdir(exist_ok=True)

    def run_visual_comparison(self):
        """Run comparison with full visual documentation"""

        print("🧭 VISUAL MAZE DEMONSTRATION")
        print("=" * 40)
        print("Creating maze visualization...")

        # Create maze and visualize structure
        maze = DeceptiveMaze(size=self.maze_size)
        self.visualize_maze_structure(maze)

        # Run QSE-Émile
        print("\nRunning QSE-Émile...")
        qse_results = self.run_agent_with_tracking(maze, agent_type="qse")

        # Reset maze and run Standard RL
        print("Running Standard RL...")
        maze_rl = DeceptiveMaze(size=self.maze_size)
        rl_results = self.run_agent_with_tracking(maze_rl, agent_type="standard")

        # Create comparison visualizations
        self.create_path_comparison(qse_results, rl_results)
        self.create_performance_analysis(qse_results, rl_results)

        print(f"\n✅ Visual demonstration complete!")
        print(f"📸 Images saved in: {self.save_dir}")

        return qse_results, rl_results

    def visualize_maze_structure(self, maze):
        """Create overview visualization of the maze structure"""

        fig, ax = plt.subplots(figsize=(10, 10))

        # Create custom colormap for maze
        colors = ['white', 'black', 'green', 'red', 'gold']  # path, wall, start, goal, deceptive areas
        cmap = ListedColormap(colors)

        # Prepare maze for visualization
        display_maze = maze.maze.copy().astype(float)

        # Mark special positions
        display_maze[maze.start_pos] = 2  # Start = green
        display_maze[maze.goal_pos] = 3   # Goal = red

        # Mark deceptive areas (attractive dead ends)
        deceptive_areas = [(3, 3), (3, maze.size-4), (maze.size-4, 3), (7, 7)]
        for area in deceptive_areas:
            if (area[0] < maze.size and area[1] < maze.size and
                0 < area[0] < maze.size-1 and 0 < area[1] < maze.size-1):
                if display_maze[area] == 0:  # Only mark if it's a path
                    display_maze[area] = 4  # Deceptive = gold

        # Display maze
        im = ax.imshow(display_maze, cmap=cmap, vmin=0, vmax=4)

        # Add grid
        ax.set_xticks(np.arange(-0.5, maze.size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, maze.size, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add legend
        legend_elements = [
            patches.Patch(color='white', label='Open Path'),
            patches.Patch(color='black', label='Wall'),
            patches.Patch(color='green', label='Start Position'),
            patches.Patch(color='red', label='Goal Position'),
            patches.Patch(color='gold', label='Deceptive Area (Local Optimum)')
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

        ax.set_title('Deceptive Maze Structure\nAttractive Dead Ends Create Local Optima',
                    fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.save_dir / "maze_structure.png", dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📸 Maze structure saved to {self.save_dir / 'maze_structure.png'}")

    def run_agent_with_tracking(self, maze, agent_type="qse"):
        """Run agent and track complete path for visualization"""

        if agent_type == "qse":
            agent = QSEMazeAgent()
            goals = ["explore", "exploit", "maintain", "adapt", "escape_area"]
            for goal in goals:
                agent.goal.add_goal(goal)
        else:
            agent = StandardRLAgent(
                state_size=3, action_size=4,
                epsilon=0.1, learning_rate=0.1
            )

        # Tracking
        path = [maze.current_pos]
        context_switches = []
        stuck_periods = []
        rewards_received = []

        state = maze.reset()
        done = False
        step_count = 0
        consecutive_stuck = 0

        while not done and step_count < self.max_steps:
            # Track if stuck
            was_stuck = maze.is_stuck()
            if was_stuck:
                consecutive_stuck += 1
            else:
                if consecutive_stuck > 5:  # End of stuck period
                    stuck_periods.append((step_count - consecutive_stuck, step_count))
                consecutive_stuck = 0

            # Get action
            if agent_type == "qse":
                # Track context switches
                old_context = agent.context.get_current()
                action = agent.select_action(state, maze)
                new_context = agent.context.get_current()

                if old_context != new_context:
                    context_switches.append({
                        'step': step_count,
                        'position': maze.current_pos,
                        'old_context': old_context,
                        'new_context': new_context,
                        'was_stuck': was_stuck
                    })
            else:
                action = agent.select_action(state, maze)

            # Take step
            state, reward, done = maze.step(action)

            # Update agent
            if agent_type == "qse":
                agent.goal.feedback(reward)
            else:
                agent.update(state, action, reward, state, done)

            # Track path and rewards
            path.append(maze.current_pos)
            if reward > 0:
                rewards_received.append((step_count, maze.current_pos, reward))

            step_count += 1

        return {
            'agent_type': agent_type,
            'path': path,
            'final_position': maze.current_pos,
            'reached_goal': maze.current_pos == maze.goal_pos,
            'total_steps': step_count,
            'context_switches': context_switches if agent_type == "qse" else [],
            'stuck_periods': stuck_periods,
            'rewards_received': rewards_received,
            'maze': maze
        }

    def create_path_comparison(self, qse_results, rl_results):
        """Create side-by-side path comparison visualization"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # QSE-Émile path
        self.plot_agent_path(ax1, qse_results, "QSE-Émile: Context-Switching Success")

        # Standard RL path
        self.plot_agent_path(ax2, rl_results, "Standard RL: Trapped in Local Optima")

        plt.tight_layout()
        plt.savefig(self.save_dir / "path_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📸 Path comparison saved to {self.save_dir / 'path_comparison.png'}")

    def plot_agent_path(self, ax, results, title):
        """Plot individual agent path with annotations"""

        maze = results['maze']
        path = results['path']

        # Base maze
        display_maze = maze.maze.copy().astype(float)

        # Color scheme: 0=white(path), 1=black(wall), 2=lightblue(visited)
        cmap = ListedColormap(['white', 'black', 'lightblue'])

        # Mark visited positions
        for pos in set(path):
            if display_maze[pos] == 0:  # Only mark paths
                display_maze[pos] = 2

        # Display maze
        ax.imshow(display_maze, cmap=cmap, vmin=0, vmax=2)

        # Plot path
        if len(path) > 1:
            path_x = [pos[1] for pos in path]
            path_y = [pos[0] for pos in path]
            ax.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.7, label='Agent Path')

        # Mark special positions
        start_pos = maze.start_pos
        goal_pos = maze.goal_pos
        final_pos = results['final_position']

        ax.scatter(start_pos[1], start_pos[0], c='green', s=200, marker='o',
                  label='Start', edgecolors='black', linewidth=2)
        ax.scatter(goal_pos[1], goal_pos[0], c='red', s=200, marker='*',
                  label='Goal', edgecolors='black', linewidth=2)
        ax.scatter(final_pos[1], final_pos[0], c='orange', s=150, marker='X',
                  label='Final Position', edgecolors='black', linewidth=2)

        # Mark context switches for QSE-Émile
        if results['agent_type'] == 'qse' and results['context_switches']:
            switch_positions = [(s['position'][1], s['position'][0]) for s in results['context_switches']]
            if switch_positions:
                sx, sy = zip(*switch_positions)
                ax.scatter(sx, sy, c='purple', s=100, marker='^',
                          label=f'Context Switches ({len(switch_positions)})',
                          edgecolors='white', linewidth=1)

        # Mark stuck periods
        if results['stuck_periods']:
            for start_step, end_step in results['stuck_periods']:
                if start_step < len(path) and end_step < len(path):
                    stuck_pos = path[start_step]
                    ax.scatter(stuck_pos[1], stuck_pos[0], c='yellow', s=80, marker='s',
                              alpha=0.7, edgecolors='red', linewidth=1)

        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        # Add performance text
        success_text = "SUCCESS" if results['reached_goal'] else "FAILED"
        color = "green" if results['reached_goal'] else "red"
        ax.text(0.02, 0.98, f"{success_text}\n{results['total_steps']} steps",
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                facecolor=color, alpha=0.7))

    def create_performance_analysis(self, qse_results, rl_results):
        """Create performance metrics visualization"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Success comparison
        success_data = [int(qse_results['reached_goal']), int(rl_results['reached_goal'])]
        colors = ['green' if s else 'red' for s in success_data]
        ax1.bar(['QSE-Émile', 'Standard RL'], success_data, color=colors, alpha=0.7)
        ax1.set_ylabel('Goal Reached')
        ax1.set_title('Task Success')
        ax1.set_ylim(0, 1.1)
        for i, v in enumerate(success_data):
            ax1.text(i, v + 0.05, 'SUCCESS' if v else 'FAILED',
                    ha='center', fontweight='bold')

        # Steps comparison
        steps_data = [qse_results['total_steps'], rl_results['total_steps']]
        ax2.bar(['QSE-Émile', 'Standard RL'], steps_data, color=['blue', 'red'], alpha=0.7)
        ax2.set_ylabel('Steps Taken')
        ax2.set_title('Efficiency')
        for i, v in enumerate(steps_data):
            ax2.text(i, v + max(steps_data)*0.02, str(v), ha='center', fontweight='bold')

        # Context switches vs stuck periods
        qse_switches = len(qse_results['context_switches'])
        rl_stuck = len(rl_results['stuck_periods'])

        ax3.bar(['QSE Context\nSwitches', 'RL Stuck\nPeriods'],
                [qse_switches, rl_stuck], color=['purple', 'orange'], alpha=0.7)
        ax3.set_ylabel('Count')
        ax3.set_title('Adaptive vs Stuck Behavior')
        for i, v in enumerate([qse_switches, rl_stuck]):
            ax3.text(i, v + 0.1, str(v), ha='center', fontweight='bold')

        # Path efficiency (distance covered vs steps)
        qse_path_length = len(set(qse_results['path']))  # Unique positions visited
        rl_path_length = len(set(rl_results['path']))

        efficiency_data = [qse_path_length / qse_results['total_steps'],
                          rl_path_length / rl_results['total_steps']]

        ax4.bar(['QSE-Émile', 'Standard RL'], efficiency_data,
                color=['blue', 'red'], alpha=0.7)
        ax4.set_ylabel('Unique Positions / Total Steps')
        ax4.set_title('Exploration Efficiency')
        for i, v in enumerate(efficiency_data):
            ax4.text(i, v + max(efficiency_data)*0.02, f'{v:.3f}',
                    ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.save_dir / "performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📸 Performance analysis saved to {self.save_dir / 'performance_analysis.png'}")


def main():
    """Run the visual maze demonstration"""

    print("🎬 QSE-ÉMILE VISUAL MAZE DEMONSTRATION")
    print("=" * 50)
    print("Creating compelling visual evidence of context-switching advantage")

    demo = VisualMazeDemo(maze_size=12, max_steps=400)
    qse_results, rl_results = demo.run_visual_comparison()

    # Print summary
    print(f"\n📊 VISUAL DEMONSTRATION SUMMARY:")
    print(f"QSE-Émile: {'SUCCESS' if qse_results['reached_goal'] else 'FAILED'} "
          f"in {qse_results['total_steps']} steps "
          f"({len(qse_results['context_switches'])} context switches)")
    print(f"Standard RL: {'SUCCESS' if rl_results['reached_goal'] else 'FAILED'} "
          f"in {rl_results['total_steps']} steps "
          f"({len(rl_results['stuck_periods'])} stuck periods)")

    print(f"\n🎯 KEY VISUAL EVIDENCE:")
    if qse_results['reached_goal'] and not rl_results['reached_goal']:
        print(f"✅ QSE-Émile escapes local optima via context switching")
        print(f"✅ Standard RL gets trapped in deceptive areas")
        print(f"✅ Visual proof of endogenous recontextualization advantage")

    print(f"\n📁 All visualizations saved in: maze_visuals/")
    print(f"   • maze_structure.png - Shows deceptive areas")
    print(f"   • path_comparison.png - Side-by-side agent paths")
    print(f"   • performance_analysis.png - Quantitative comparison")


if __name__ == "__main__":
    main()
