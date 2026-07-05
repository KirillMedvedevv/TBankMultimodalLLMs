"""
Data collection and replay buffer for the world model (MultiObjEnv, size=10).

Collect episodes with a MOSTLY random policy, store frames as uint8 (cheap),
sample fixed-length chunks as float tensors in the (T, B, ...) layout RSSM wants.

Why goal-biased episodes:
MiniGrid reward is sparse (positive only on the green goal, once). A purely
random policy almost never reaches it, so the RewardModel would train on
all-zeros and the reward-only baseline collapses to random. We steer a fraction
`goal_biased_frac` of episodes to the goal with a real WALL-AWARE BFS planner,
so the reward head sees GUARANTEED positive examples. For variety the planner
randomises its search order and sometimes routes via a random waypoint, so the
goal is reached along SEVERAL DIFFERENT paths (better generalisation of the
reward head and the decoder). Set frac=0.0 for pure-random behaviour.

Frames come from env.render() with highlight=False (clean full-grid picture).
"""

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from env_tasks import MultiObjEnv

torch.set_num_threads(6)

ACTION_DIM = 7

# MiniGrid dir index -> (dx, dy): 0=east, 1=south, 2=west, 3=north
_DIRS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
_VEC2DIR = {v: k for k, v in _DIRS.items()}


def make_env(size=10):
    return MultiObjEnv(size=size, render_mode="rgb_array", highlight=False)


def _prep_uint8(frame):
    img = Image.fromarray(frame).resize((64, 64))
    return np.asarray(img, dtype=np.uint8)              # (64, 64, 3)


# ----------------------------------------------------------------- planning
def _passable(env, cell, goal):
    """Floor is walkable; the goal is the target; walls / key / locked door block."""
    if cell == goal:
        return True
    x, y = cell
    if not (0 <= x < env.width and 0 <= y < env.height):
        return False
    return env.grid.get(x, y) is None


def _bfs(env, start, goal, rng):
    """Shortest cell path start -> goal (list of cells), or None. Neighbour order
    is shuffled so different calls yield different (equally short) paths."""
    seen = {start}
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            path = [cur]
            while prev[path[-1]] is not None:
                path.append(prev[path[-1]])
            return path[::-1]
        nbrs = [(cur[0] + dx, cur[1] + dy) for dx, dy in _DIRS.values()]
        rng.shuffle(nbrs)
        for nb in nbrs:
            if nb not in seen and _passable(env, nb, goal):
                seen.add(nb); prev[nb] = cur; q.append(nb)
    return None


def _random_free_cell(env, rng):
    for _ in range(200):
        x = int(rng.integers(1, env.width - 1))
        y = int(rng.integers(1, env.height - 1))
        if env.grid.get(x, y) is None:
            return (x, y)
    return None


def _actions_for_path(path, start_dir):
    """Cell path -> MiniGrid action list (turn as needed, then forward).
    0=left, 1=right, 2=forward."""
    acts, cur = [], start_dir
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        want = _VEC2DIR[(x1 - x0, y1 - y0)]
        diff = (want - cur) % 4
        if diff == 1:
            acts.append(1)
        elif diff == 3:
            acts.append(0)
        elif diff == 2:
            acts += [1, 1]
        cur = want
        acts.append(2)                                   # step forward onto next cell
    return acts


def plan_goal_actions(env, rng, detour_prob=0.5):
    """GUARANTEED action sequence that walks the agent onto the green goal.
    With prob `detour_prob` routes via a random waypoint -> a genuinely different
    path than the straight shortest one. Returns [] if unreachable (shouldn't
    happen — env asserts reachability)."""
    start = (int(env.agent_pos[0]), int(env.agent_pos[1]))
    goal = tuple(int(c) for c in env.goal_pos)

    path = None
    if rng.random() < detour_prob:
        wp = _random_free_cell(env, rng)
        if wp and wp not in (start, goal):
            p1 = _bfs(env, start, wp, rng)
            p2 = _bfs(env, wp, goal, rng)
            if p1 and p2:
                path = p1[:-1] + p2                      # concat, drop duplicated waypoint
    if path is None:
        path = _bfs(env, start, goal, rng)
    if path is None:
        return []
    return _actions_for_path(path, int(env.agent_dir))


# --------------------------------------------------------------- collection
def collect_episodes(num_episodes=200, max_len=64, size=10, seed=0,
                     goal_biased_frac=0.4, detour_prob=0.5):
    env = make_env(size=size)
    n_biased = int(round(num_episodes * goal_biased_frac))
    rng = np.random.default_rng(seed)
    episodes = []

    for ep in range(num_episodes):
        biased = ep < n_biased
        env.reset(seed=seed + ep)
        imgs, acts, rews = [], [], []

        # optional random wander first, so goal episodes start from varied states
        prefix = int(rng.integers(4, 10)) if biased else 0
        plan, plan_i = None, 0

        for t in range(max_len):
            imgs.append(_prep_uint8(env.render()))       # frame BEFORE the action

            if biased and t >= prefix:
                if plan is None:                          # plan once, from current state
                    plan = plan_goal_actions(env, rng, detour_prob)
                a = plan[plan_i] if plan_i < len(plan) else env.action_space.sample()
                plan_i += 1
            else:
                a = int(rng.choice([0, 1, 2])) if rng.random() < 0.8 else env.action_space.sample()           # random policy, all 7 actions

            _, r, term, trunc, _ = env.step(a)
            acts.append(a); rews.append(float(r))
            if term or trunc:
                break

        episodes.append(dict(
            images=np.stack(imgs),                       # (T, 64, 64, 3) uint8
            actions=np.asarray(acts, dtype=np.int64),    # (T,)
            rewards=np.asarray(rews, dtype=np.float32),  # (T,)
        ))
    return episodes


class ReplayBuffer:
    def __init__(self, episodes, min_len=2):
        self.episodes = [e for e in episodes if len(e["actions"]) >= min_len]

    def sample(self, batch_size, chunk_len):
        pool = [e for e in self.episodes if len(e["actions"]) >= chunk_len]
        assert pool, "no episodes >= chunk_len; lower chunk_len or raise max_len"

        imgs, acts, rews = [], [], []
        for _ in range(batch_size):
            e = pool[np.random.randint(len(pool))]
            T = len(e["actions"])
            start = np.random.randint(0, T - chunk_len + 1)
            sl = slice(start, start + chunk_len)
            imgs.append(e["images"][sl]); acts.append(e["actions"][sl]); rews.append(e["rewards"][sl])

        images = torch.from_numpy(np.stack(imgs)).float().div(255).sub(0.5)
        images = images.permute(1, 0, 4, 2, 3).contiguous()          # (L,B,3,64,64) [-0.5,0.5]
        actions = torch.from_numpy(np.stack(acts))
        actions = F.one_hot(actions, ACTION_DIM).float().permute(1, 0, 2)   # (L,B,A)
        rewards = torch.from_numpy(np.stack(rews)).permute(1, 0)            # (L,B)
        return images, actions, rewards