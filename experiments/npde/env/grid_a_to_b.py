from __future__ import annotations

import random
from itertools import count

import imageio
import minigrid.core.world_object as Obj
import numpy as np
from gymnasium import spaces
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from PIL import Image


class A(Obj.Floor):
    def __init__(self):
        super().__init__(color="yellow")


class B(Obj.Floor):
    def __init__(self):
        super().__init__(color="blue")


class C(Obj.Floor):
    def __init__(self):
        super().__init__(color="red")


class D(Obj.Floor):
    def __init__(self):
        super().__init__(color="green")


class AtoB(MiniGridEnv):
    def __init__(
        self,
        size=9,
        max_steps: int | None = 100,
        test: bool = False,
        **kwargs,
    ):
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 4 * size**2
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
            agent_pov=False,
        )
        self.grid_size = size

        # Flat obs
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(np.prod((self.agent_view_size, self.agent_view_size, 3)),),
            dtype="uint8",
        )
        self.action_space = spaces.Discrete(3)
        self.test = test
        self.goals = [0, 1, 2, 3]
        self.goal_locs = [(1, 1), (3, 3), (5, 5), (7, 7)]
        self.goal_tiles = [A, B, C, D]
        self.pairs = [(i, j) for i in self.goals for j in self.goals if i is not j]
        # self.test_pairs = random.sample(self.pairs, 4)
        self.test_pairs = [(2, 0), (3, 0), (0, 1), (3, 1)]
        self.train_pairs = [pair for pair in self.pairs if pair not in self.test_pairs]
        self.current_goal_set = self.test_pairs[0]
        self.current_goal = self.current_goal_set[0]
        self.met_goal = [False, False]

    @staticmethod
    def _gen_mission():
        return "test"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.grid.set(1, 1, A())
        self.grid.set(3, 3, B())
        self.grid.set(5, 5, C())
        self.grid.set(7, 7, D())

        self.grid.set(width - 3, 1, self.goal_tiles[self.current_goal_set[0]]())
        self.grid.set(width - 2, 1, self.goal_tiles[self.current_goal_set[1]]())

        # self.grid.set(1, height - 2, self.goal_tiles[self.current_goal_set[0]]())
        # self.grid.set(2, height - 2, self.goal_tiles[self.current_goal_set[1]]())

        # Place the agent
        self.agent_dir = 3
        self.place_agent(top=(width - 3, height - 2), rand_dir=False)

        self.mission = "A to B"

    def step(self, action):
        obs, reward, done, terminated, info = super().step(action)
        done = terminated
        reward = -0.01

        # Goal loc other than the goal:
        if (self.agent_pos in self.goal_locs) and (
            (self.agent_pos != self.goal_locs[self.current_goal_set[0]])
            and (self.agent_pos != self.goal_locs[self.current_goal_set[1]])
        ):
            reward = -1
            done = True

        # Get first goal:
        if self.agent_pos == self.goal_locs[self.current_goal_set[0]] and not self.met_goal[0]:
            self.met_goal[0] = True
            reward = 1
            self.current_goal = self.current_goal_set[1]

        # Get second goal:
        if self.agent_pos == self.goal_locs[self.current_goal_set[1]] and self.met_goal[0]:
            reward = 1
            done = True

        # Out of steps:
        if terminated:
            reward = -1

        return obs["image"].flatten(), reward, done, info

    def reset(self):
        if self.test:
            self.current_goal_set = random.sample(self.test_pairs, 1)[0]
            self.current_goal = self.current_goal_set[0]

        else:
            self.current_goal_set = random.sample(self.train_pairs, 1)[0]
            self.current_goal = self.current_goal_set[0]

        self.met_goal = [False, False]
        obs = super().reset()
        return obs[0]["image"].flatten()


if __name__ == "__main__":
    env = AtoB(max_steps=200, render_mode="rgb_array")

    for _ in range(10):
        obs = env.reset()
        img = env.render()
        imageio.imwrite("test.png", img)
        done = False
        images = []
        for i in count():
            # action = np.random.randint(0, 3)
            action = int(input("Action (0:Turn Left, 1: Turn Right, 2: Forward):"))
            obs, reward, done, info = env.step(action)
            img = env.render()
            images.append(Image.fromarray(img.astype("uint8")))

            # images.append(img)
            imageio.imwrite("test.png", img)
            # os.system("clear")
            print(i, reward)
            images[0].save("a2b.gif", save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)
            if done:
                break
