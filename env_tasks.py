from __future__ import annotations
from minigrid.manual_control import ManualControl
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv


class MultiObjEnv(MiniGridEnv):

    TASK_PROMPTS = {
        "key":  "the red triangle standing next to the yellow key",
        "door": "the red triangle standing next to the blue door",
        "goal": "the red triangle standing on the green square",
    }

    # внутренние стены как отрезки (x0,y0,x1,y1), включительно. Для size=10 (клетки 1..8).
    WALL_SEGMENTS = [
        (5, 1, 5, 2),   # верх-центр  — вертикаль, строки 1-2
        (7, 4, 8, 4),   # центр-право — горизонталь, столбцы 7-8
        (6, 7, 6, 8),   # низ-право   — вертикаль, строки 7-8
        (1, 6, 3, 6),   # низ-лево    — горизонталь, столбцы 1-3
    ]

    def __init__(self, size=10, random_start=True,
                 agent_start_pos=(1, 1), agent_start_dir=0,
                 key_color="yellow", door_color="blue",
                 max_steps=None, **kwargs):
        self.random_start = random_start
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.key_color = key_color
        self.door_color = door_color

        # объекты по углам внутренней области (клетки 1..size-2)
        self.door_pos = (size - 2, 1)
        self.key_pos  = (1, 5)
        self.goal_pos = (size - 2, size - 2)

        # развернём отрезки в список клеток-стен
        self.wall_cells = []
        for x0, y0, x1, y1 in self.WALL_SEGMENTS:
            for x in range(min(x0, x1), max(x0, x1) + 1):
                for y in range(min(y0, y1), max(y0, y1) + 1):
                    self.wall_cells.append((x, y))

        # стены не должны лезть на рамку или на объекты
        objs = {self.door_pos, self.key_pos, self.goal_pos}
        for (x, y) in self.wall_cells:
            assert 1 <= x <= size - 2 and 1 <= y <= size - 2, \
                f"стена {(x, y)} вне игрового поля (size={size}) — поправь WALL_SEGMENTS"
            assert (x, y) not in objs, f"стена {(x, y)} на объекте — поправь WALL_SEGMENTS"

        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 4 * size ** 2
        super().__init__(mission_space=mission_space, grid_size=size,
                         see_through_walls=True, max_steps=max_steps, **kwargs)

    @staticmethod
    def _gen_mission():
        return "reach the target object"

    # ------------------------------------------------------------------ grid
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)      # внешняя рамка

        self.put_obj(Door(self.door_color, is_locked=True), *self.door_pos)
        self.put_obj(Key(self.key_color), *self.key_pos)
        self.put_obj(Goal(), *self.goal_pos)

        for wx, wy in self.wall_cells:                # внутренние препятствия
            self.put_obj(Wall(), wx, wy)
        self._assert_reachable()                      # стены не должны запирать объекты

        if self.random_start:
            self._place_agent_free()
        else:
            self.agent_pos = np.array(self.agent_start_pos)
            self.agent_dir = self.agent_start_dir

        self.mission = "reach the target object"

    def _assert_reachable(self):
        """BFS от цели: дверь/ключ/цель должны быть достижимы. Если полоса-стена
        отрезала объект — падаем с понятной ошибкой (а не молча ломаем задачу)."""
        from collections import deque

        def passable(x, y):
            c = self.grid.get(x, y)
            return c is None or isinstance(c, (Door, Key, Goal))   # объекты проходимы для BFS

        seen = {self.goal_pos}
        q = deque([self.goal_pos])
        while q:
            x, y = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (nx, ny) not in seen and 0 <= nx < self.width and 0 <= ny < self.height \
                        and passable(nx, ny):
                    seen.add((nx, ny)); q.append((nx, ny))
        for name, pos in [("door", self.door_pos), ("key", self.key_pos), ("goal", self.goal_pos)]:
            if pos not in seen:
                raise RuntimeError(f"стены заперли {name}@{pos} — поправь WALL_SEGMENTS")

    def _place_agent_free(self):
        """Случайная клетка: НЕ стена/объект и НЕ впритык к цели. RNG сидирован."""
        targets = [self.door_pos, self.key_pos, self.goal_pos]
        for _ in range(2000):
            x = self._rand_int(1, self.width - 1)
            y = self._rand_int(1, self.height - 1)
            if self.grid.get(x, y) is not None:                       # стена или объект
                continue
            if any(abs(x - tx) + abs(y - ty) <= 1 for tx, ty in targets):
                continue
            self.agent_pos = np.array([x, y])
            self.agent_dir = self._rand_int(0, 4)
            return
        self.agent_pos = np.array(self.agent_start_pos)
        self.agent_dir = self.agent_start_dir

    # -------------------------------------------------- ground-truth success
    def _agent_adjacent_to(self, pos):
        ax, ay = self.agent_pos
        return abs(int(ax) - pos[0]) + abs(int(ay) - pos[1]) == 1

    def carrying_key(self):
        return isinstance(self.carrying, Key)

    def door_is_open(self):
        cell = self.grid.get(*self.door_pos)
        return isinstance(cell, Door) and cell.is_open

    def at_goal(self):
        return tuple(int(c) for c in self.agent_pos) == tuple(self.goal_pos)

    def task_success(self, task):
        if task == "key":
            return self.carrying_key() or self._agent_adjacent_to(self.key_pos)
        if task == "door":
            return self.door_is_open() or self._agent_adjacent_to(self.door_pos)
        if task == "goal":
            return self.at_goal()
        raise ValueError(f"unknown task {task!r}")


def main():
    env = MultiObjEnv(render_mode="human", highlight=False, size=10)
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()