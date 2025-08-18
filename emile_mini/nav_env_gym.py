import numpy as np
import collections
from typing import Tuple, Optional
import gymnasium as gym
from gymnasium import spaces

from .complete_navigation_system_d import ClearPathEnvironment, ProactiveEmbodiedQSEAgent
from .config import QSEConfig

def _bfs_shortest_path_len(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[int]:
    sx, sy = start; gx, gy = goal
    if (sx, sy) == (gx, gy): return 0
    n = grid.shape[0]
    seen = np.zeros_like(grid, dtype=bool)
    q = collections.deque([(sx, sy, 0)])
    seen[sx, sy] = True
    while q:
        x, y, d = q.popleft()
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if nx<0 or ny<0 or nx>=n or ny>=n: continue
            if seen[nx, ny]: continue
            if grid[nx, ny] >= 0.8: continue
            if (nx, ny) == (gx, gy): return d+1
            seen[nx, ny] = True
            q.append((nx, ny, d+1))
    return None

def _random_free_cell(grid: np.ndarray, rng: np.random.RandomState):
    free = np.argwhere(grid < 0.8)
    i = rng.randint(0, len(free))
    x, y = map(int, free[i])
    return (x, y)

def _target_for_quadrant(size: int, quadrant: str):
    lookup = {
        "NE": (size-5, size-5),
        "NW": (5,        size-5),
        "SE": (size-5,   5),
        "SW": (5,        5),
        "C":  (size//2,  size//2),
    }
    return lookup.get(quadrant, (size//2, size//2))

class NavEnv(gym.Env):
    """
    Gymnasium env for PPO vs ProactiveEmbodiedQSEAgent comparison.
    """
    metadata = {"render_modes": []}

    def __init__(self, size=20, max_steps=80, obstacle_density=0.15, quadrant="NE",
             start_mode="random", seed=42,
             progress_k=0.5, step_cost=0.02, turn_penalty=0.01,
             collision_penalty=0.10, success_bonus=2.0):
        super().__init__()
        self.size = int(size)
        self.max_steps = int(max_steps)
        self.obstacle_density = float(obstacle_density)
        self.quadrant = quadrant
        self.start_mode = start_mode
        self._rng = np.random.RandomState(seed)

        self.action_space = spaces.Discrete(3)
        high = np.array([1,1,1,1,1,1,1,1,1], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.env = None
        self.agent = None
        self.steps = 0
        self.target = _target_for_quadrant(self.size, self.quadrant)
        self._last_dist = None
        self._path_len = 0
        self._shortest = None

        self.progress_k = float(progress_k)
        self.step_cost = float(step_cost)
        self.turn_penalty = float(turn_penalty)
        self.collision_penalty = float(collision_penalty)
        self.success_bonus = float(success_bonus)
        self._prev_pos = None

    def _make_grid_with_obstacles(self):
        grid = np.zeros((self.size, self.size), dtype=float)
        mask = self._rng.rand(self.size, self.size) < self.obstacle_density
        cx, cy = self.size//2, self.size//2
        tx, ty = self.target
        for (x,y) in [(cx,cy),(tx,ty)]:
            x0, x1 = max(0,x-1), min(self.size, x+2)
            y0, y1 = max(0,y-1), min(self.size, y+2)
            mask[x0:x1, y0:y1] = False
        grid[mask] = 1.0
        return grid

    def _obs(self):
        x, y = self.agent.body.state.position
        th = self.agent.body.state.orientation
        tx, ty = self.target
        dx, dy = tx - x, ty - y
        dist = np.sqrt(dx*dx + dy*dy) / np.sqrt(2*(self.size**2))
        return np.array([
            x/(self.size-1), y/(self.size-1),
            np.cos(th), np.sin(th),
            tx/(self.size-1), ty/(self.size-1),
            dx/(self.size-1), dy/(self.size-1),
            dist
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
        self.steps = 0
        self._path_len = 0

        self.env = ClearPathEnvironment(size=self.size)
        self.target = _target_for_quadrant(self.size, self.quadrant)

        for _ in range(200):
            # curriculum: sample density U[0, obstacle_density]
            orig = self.obstacle_density
            tmp_density = float(self._rng.uniform(0.0, max(0.0, orig)))
            self.obstacle_density = tmp_density
            grid = self._make_grid_with_obstacles()
            self.obstacle_density = orig
            if self.start_mode == "center":
                sx, sy = self.size//2, self.size//2
                if grid[sx, sy] >= 0.8:
                    continue
            else:
                sx, sy = _random_free_cell(grid, self._rng)

            if grid[self.target] >= 0.8:
                continue
            sp = _bfs_shortest_path_len(grid, (sx, sy), self.target)
            if sp is None:
                continue

            self.env.grid = grid
            self._shortest = sp
            break
        else:
            self.env.grid = np.zeros((self.size, self.size), dtype=float)
            sx, sy = self.size//2, self.size//2
            self._shortest = abs(sx-self.target[0]) + abs(sy-self.target[1])

        self.agent = ProactiveEmbodiedQSEAgent(QSEConfig())
        self.agent.body.state.position = (int(sx), int(sy))
        self.agent.body.state.orientation = 0.0
        self.agent.receive_memory_cue({
            "type":"navigation_cue", "target_quadrant": self.quadrant,
            "instruction":"Navigate", "priority":"high"
        })

        self._last_dist = float(np.linalg.norm(np.array(self.target) - np.array([sx, sy])))
        self._prev_pos = (int(sx), int(sy))
        return self._obs(), {}

    def step(self, action: int):
        act_name = ["move_forward", "turn_left", "turn_right"][int(action)]

        # Force chosen action for one step
        sel = self.agent.memory_goal.select_action_with_bias
        self.agent.memory_goal.select_action_with_bias = lambda b, a: act_name
        result = self.agent.embodied_step(self.env)
        self.agent.memory_goal.select_action_with_bias = sel

        self.steps += 1
        self._path_len += 1

        # Positions / distances
        new_pos_tuple = tuple(self.agent.body.state.position)
        new_pos = np.array(new_pos_tuple, dtype=float)
        dist = float(np.linalg.norm(np.array(self.target) - new_pos))

        # --- Reward shaping ---
        r = 0.0
        # progress
        r += self.progress_k * (self._last_dist - dist)
        # time penalty
        r -= self.step_cost
        # encourage using turns sparingly
        if act_name in ("turn_left", "turn_right"):
            r -= self.turn_penalty
        # collision: tried to go forward but didn't move
        if act_name == "move_forward" and new_pos_tuple == self._prev_pos:
            r -= self.collision_penalty
        # success bonus
        terminated = (dist < 2.0)
        truncated  = (self.steps >= self.max_steps)
        if terminated:
            r += self.success_bonus

        # update trackers
        self._last_dist = dist
        self._prev_pos = new_pos_tuple

        info = {"dist": dist, "cue_rate": result["cue_follow_rate"],
                "shortest": self._shortest, "path_len": self._path_len,
                "target": self.target}
        return self._obs(), float(r), terminated, truncated, info
