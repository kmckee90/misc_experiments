from __future__ import annotations

import os
import time
from itertools import count

import imageio
import minigrid.core.world_object as Obj
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from PIL import Image


class Food(Obj.Floor):
    def __init__(self):
        super().__init__(color="yellow")


class Water(Obj.Floor):
    def __init__(self):
        super().__init__(color="blue")


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=20,
        max_steps: int | None = None,
        **kwargs,
    ):
        mission_space = MissionSpace(mission_func=self._gen_mission)
        if max_steps is None:
            max_steps = 4 * size**2
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
            agent_pov=True,
        )
        self.grid_size = size

        self.h_water = 1000
        self.h_food = 1000
        self.water_decrement = 10
        self.food_decrement = 10
        self.water_increment = 200
        self.food_increment = 200
        self.n_water = 5
        self.n_food = 5
        self.n_water_remaining = self.n_water
        self.n_food_remaining = self.n_food
        self.water_regen_prob = 0.05
        self.food_regen_prob = 0.05

    @staticmethod
    def _gen_mission():
        return "test"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)
        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        n_obst = 50
        for i in range(n_obst):
            wall_x = np.random.randint(1, width - 3)
            wall_y = np.random.randint(1, height - 2)
            self.grid.set(wall_x, wall_y, Obj.Wall())

        for i in range(self.n_food):
            wall_x = np.random.randint(1, width - 3)
            wall_y = np.random.randint(1, height - 2)
            self.grid.set(wall_x, wall_y, Food())

        for i in range(self.n_water):
            wall_x = np.random.randint(1, width - 3)
            wall_y = np.random.randint(1, height - 2)
            self.grid.set(wall_x, wall_y, Water())

        # Place the agent
        self.agent_pos = (np.random.randint(2, width - 3), np.random.randint(2, height - 3))
        self.place_agent()

        self.mission = "BOAD"

    # Wrap super.step() to add in 'nutrient' behavior of colored squares.
    def step(self, action):
        self.h_water -= self.food_decrement
        self.h_food -= self.water_decrement
        obs, reward, done, terminated, info = super().step(action)
        if isinstance(self.grid.get(*self.agent_pos), Water):
            self.h_water = min(self.h_water + self.water_increment, 1000)
            self.grid.set(*self.agent_pos, None)
            self.n_water_remaining -= 1
        if isinstance(self.grid.get(*self.agent_pos), Food):
            self.h_food = min(self.h_food + self.food_increment, 1000)
            self.grid.set(*self.agent_pos, None)
            self.n_food_remaining -= 1
        done = terminated

        if self.h_food * self.h_water == 0:
            reward = -1
            done = True

        # Regenerate nutrients on the board
        if self.n_food_remaining < self.n_food and np.random.rand() < self.food_regen_prob:
            wall_x = np.random.randint(1, self.grid_size - 3)
            wall_y = np.random.randint(1, self.grid_size - 3)
            self.grid.set(wall_x, wall_y, Food())
            self.n_food_remaining += 1

        if self.n_water_remaining < self.n_water and np.random.rand() < self.water_regen_prob:
            wall_x = np.random.randint(1, self.grid_size - 3)
            wall_y = np.random.randint(1, self.grid_size - 3)
            self.grid.set(wall_x, wall_y, Water())
            self.n_water_remaining += 1

        return obs, reward, done, terminated, info

    def reset(self):
        self.h_water = 1000
        self.h_food = 1000
        self.n_food_remaining = self.n_food
        self.n_water_remaining = self.n_water
        return super().reset()


env = SimpleEnv(max_steps=1000, render_mode="rgb_array")

for _ in range(1):
    obs, info = env.reset()
    img = env.render()
    imageio.imwrite("test.png", img)
    done = False
    time.sleep(1)
    images = []
    for i in count():
        # action = np.random.randint(0, 3)
        action = int(input("Action (0:Turn Left, 1: Turn Right, 2: Forward):"))
        obs, reward, done, terminated, info = env.step(action)
        img = env.render()
        images.append(Image.fromarray(img.astype("uint8")))

        # images.append(img)
        imageio.imwrite("test.png", img)
        os.system("clear")
        print(i, reward)
        print("Food:", env.h_food)
        print("Water:", env.h_water)
        images[0].save("boad_pov.gif", save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)
        if done or terminated:
            break
