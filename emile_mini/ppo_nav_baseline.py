import os
import numpy as np
from typing import Dict, Any, Tuple
from .nav_env_gym import NavEnv

def _try_import_sb3():
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as e:
        raise ImportError("stable-baselines3 is required. pip install stable-baselines3 gymnasium") from e
    return PPO, Monitor, DummyVecEnv

def make_env_kwargs(size=20, max_steps=80, obstacle_density=0.15, quadrant="NE",
                    start_mode="random", seed=42,
                    progress_k=0.5, step_cost=0.02, turn_penalty=0.01,
                    collision_penalty=0.10, success_bonus=2.0):
    return dict(size=size, max_steps=max_steps, obstacle_density=obstacle_density,
                quadrant=quadrant, start_mode=start_mode, seed=seed,
                progress_k=progress_k, step_cost=step_cost, turn_penalty=turn_penalty,
                collision_penalty=collision_penalty, success_bonus=success_bonus)


def train_ppo(timesteps: int = 20000, save_path: str = "ppo_nav.zip", **env_kwargs):
    PPO, Monitor, DummyVecEnv = _try_import_sb3()
    def _make():
        return Monitor(NavEnv(**env_kwargs))
    env = DummyVecEnv([_make])
    model = PPO("MlpPolicy", env, verbose=1, gamma=0.99, n_steps=1024, batch_size=256,
                gae_lambda=0.95, ent_coef=0.01, learning_rate=3e-4)
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save(save_path)
    return save_path

def _episode(env: NavEnv, policy=None, deterministic=True, seed=None):
    obs, _ = env.reset(seed=seed)
    done = False
    ret, steps = 0.0, 0
    info = {"dist": 1e9, "shortest": None, "path_len": 0}
    while not done and steps < env.max_steps:
        if policy is None:
            act = env.action_space.sample()
        else:
            a, _ = policy.predict(obs, deterministic=deterministic)
            act = int(a)
        obs, r, term, trunc, info = env.step(act)
        ret += r; steps += 1
        done = term or trunc
    success = (info["dist"] < 2.0)
    shortest = info["shortest"]
    L = max(1, info["path_len"])
    spl = (shortest / L) if success and shortest is not None else 0.0
    return ret, steps, success, spl

def eval_policy(n_episodes=100, policy_path="ppo_nav.zip", **env_kwargs) -> Dict[str, Any]:
    PPO, Monitor, DummyVecEnv = _try_import_sb3()
    env = NavEnv(**env_kwargs)
    model = PPO.load(policy_path)
    rets, steps, succ, spls = [], [], [], []
    for i in range(n_episodes):
        R, S, OK, SPL = _episode(env, model, deterministic=True, seed=1000+i)
        rets.append(R); steps.append(S); succ.append(OK); spls.append(SPL)
    return {
        "episodes": n_episodes,
        "success_rate": float(np.mean(succ)),
        "avg_return": float(np.mean(rets)),
        "avg_steps": float(np.mean(steps)),
        "avg_spl": float(np.mean(spls)),
    }

def eval_handcrafted(n_episodes=100, **env_kwargs) -> Dict[str, Any]:
    env = NavEnv(**env_kwargs)

    def _agent_action():
        agent = env.agent
        old_pos = np.array(agent.body.state.position, dtype=float)
        biases = agent.memory_goal.get_action_bias(old_pos, agent.body.state.orientation)
        avail = ["move_forward", "turn_left", "turn_right"]
        act = agent.memory_goal.select_action_with_bias(biases, avail)
        return {"move_forward":0, "turn_left":1, "turn_right":2}.get(act, 0)

    rets, steps, succ, spls = [], [], [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=2000+i)
        done = False
        R, S = 0.0, 0
        info = {"dist": 1e9, "shortest": None, "path_len": 0}
        while not done and S < env.max_steps:
            act = _agent_action()
            obs, r, term, trunc, info = env.step(act)
            R += r; S += 1
            done = term or trunc
        success = (info["dist"] < 2.0)
        shortest = info["shortest"]
        L = max(1, info["path_len"])
        SPL = (shortest / L) if success and shortest is not None else 0.0
        rets.append(R); steps.append(S); succ.append(success); spls.append(SPL)

    return {
        "episodes": n_episodes,
        "success_rate": float(np.mean(succ)),
        "avg_return": float(np.mean(rets)),
        "avg_steps": float(np.mean(steps)),
        "avg_spl": float(np.mean(spls)),
    }

def compare_ppo_vs_handcrafted(n_episodes=200, timesteps=50000, model_path="ppo_nav.zip", **env_kwargs):
    from stable_baselines3 import PPO  # ensure installed
    if not os.path.exists(model_path):
        train_ppo(timesteps=timesteps, save_path=model_path, **env_kwargs)
    ppo_stats = eval_policy(n_episodes=n_episodes, policy_path=model_path, **env_kwargs)
    hc_stats  = eval_handcrafted(n_episodes=n_episodes, **env_kwargs)
    return {"ppo": ppo_stats, "handcrafted": hc_stats}
