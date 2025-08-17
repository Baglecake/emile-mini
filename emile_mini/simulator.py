"""
Run cognitive simulation using EmileAgent.
Provides functions to run episodes with optional external inputs,
and collect full history for analysis.
"""
from .config import CONFIG
from .agent import EmileAgent


def run_simulation(steps: int = 500, dt: float = 0.01, input_sequence: list = None):
    """
    Execute a single simulation of the cognitive agent.

    Args:
        steps: number of time steps to simulate
        dt: time increment
        input_sequence: optional list of dicts with external inputs (e.g., {'reward':float})

    Returns:
        history: dict containing time series for surplus, sigma, context, and goals
    """
    agent = EmileAgent(CONFIG)

    # Add goals so the system actually works
    for goal_id in ["explore", "exploit", "maintain", "adapt"]:
        agent.goal.add_goal(goal_id)

    # Add basic rewards if none provided
    if input_sequence is None:
        input_sequence = []
        for t in range(steps):
            reward = 0.8 if t % 25 == 0 else (0.6 if t % 40 == 0 else 0.0)
            input_sequence.append({'reward': reward} if reward > 0 else None)

    for t in range(steps):
        ext = None
        if input_sequence and t < len(input_sequence):
            ext = input_sequence[t]
        metrics = agent.step(dt=dt, external_input=ext)

    return agent.get_history()
