"""
Entry point for the QSE-Émile cognitive simulation.
Runs the agent, then visualizes key metrics.
"""
from simulator import run_simulation
from viz import plot_surplus_sigma, plot_context_timeline, plot_goal_timeline
from config import CONFIG

if __name__ == "__main__":
    # Simulation parameters
    STEPS = 500
    DT = 0.01

    print(f"Running simulation for {STEPS} steps (dt={DT})...")
    history = run_simulation(steps=STEPS, dt=DT)

    # Visualize surplus and sigma over time
    print("Plotting surplus and symbolic curvature...")
    plot_surplus_sigma(history, dt=DT)

    # Visualize context shifts
    print("Plotting context timeline...")
    plot_context_timeline(history, dt=DT)

    # Visualize goal selections
    print("Plotting goal timeline...")
    plot_goal_timeline(history, dt=DT)

    print("Simulation complete.")
