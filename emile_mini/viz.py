# viz.py
"""
Visualization functions for QSE-Émile simulation history:
 - plot_surplus_sigma: surplus and sigma over time
 - plot_context_timeline: context ID across steps
 - plot_goal_timeline: goals selected over time
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_surplus_sigma(history: dict, dt: float = 0.01):
    """
    Plot mean surplus and mean sigma over time.

    Args:
        history: dict containing 'surplus_mean' and 'sigma_mean' lists
        dt: timestep increment

    Shows a two-panel plot.
    """
    t = np.arange(len(history['surplus_mean'])) * dt
    surplus = np.array(history['surplus_mean'])
    sigma = np.array(history['sigma_mean'])

    plt.figure(figsize=(10,6))

    plt.subplot(2,1,1)
    plt.plot(t, surplus, label='Mean Surplus')
    plt.ylabel('Surplus')
    plt.title('Surplus over Time')
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(t, sigma, label='Mean Sigma', color='orange')
    plt.ylabel('Sigma')
    plt.xlabel('Time')
    plt.title('Symbolic Curvature over Time')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_context_timeline(history: dict, dt: float = 0.01):
    """
    Plot the context ID progression over time.

    Args:
        history: dict containing 'context_id' list
        dt: timestep increment
    """
    t = np.arange(len(history['context_id'])) * dt
    ctx = history['context_id']

    plt.figure(figsize=(10,3))
    plt.step(t, ctx, where='post')
    plt.ylabel('Context ID')
    plt.xlabel('Time')
    plt.title('Context Shifts over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_goal_timeline(history: dict, dt: float = 0.01):
    """
    Plot the sequence of selected goals over time.

    Args:
        history: dict containing 'goal' list (categorical)
        dt: timestep increment
    """
    t = np.arange(len(history['goal'])) * dt
    goals = history['goal']
    # Map goal IDs to integers for plotting
    unique = list(dict.fromkeys(goals))  # preserve order
    mapping = {g:i for i,g in enumerate(unique)}
    goal_ids = [mapping[g] if g in mapping else -1 for g in goals]

    plt.figure(figsize=(10,3))
    plt.scatter(t, goal_ids, c=goal_ids, cmap='tab20', s=10)
    plt.yticks(range(len(unique)), unique)
    plt.ylabel('Goal ID')
    plt.xlabel('Time')
    plt.title('Goals Selected over Time')
    plt.grid(True)
    plt.tight_layout()

    plt.show()
