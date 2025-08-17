# emile_mini/basic_sim.py
from emile_mini.simulator import run_simulation
from emile_mini.viz import plot_surplus_sigma, plot_context_timeline, plot_goal_timeline

def run(steps: int = 500, dt: float = 0.01):
    print(f"Running simulation for {steps} steps (dt={dt})...")
    history = run_simulation(steps=steps, dt=dt)

    print("Plotting surplus & curvature…")
    plot_surplus_sigma(history, dt=dt)

    print("Plotting context timeline…")
    plot_context_timeline(history, dt=dt)

    print("Plotting goal timeline…")
    plot_goal_timeline(history, dt=dt)

    print("Simulation complete.")
    return history

