import argparse
import time
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path

import torch

# from tasks import WaterMazeEasy
from env.grid_a_to_b import AtoB
from PIL import Image
from rdqn_policy import Agent

import wandb

SAVE_PATH = "ood/saves/"


@dataclass
class Experiment_Config:
    notes = "DQN_policy_test"
    task = AtoB
    agent = Agent
    rl_iters = 10000000
    rl_task_args = {"render_mode": "rgb_array"}
    device = "cuda:2"
    gather_length = 2000


@dataclass
class dqn_hyperpars:
    memory_length: int = 100000
    gamma: float = 0.975
    layer_size: int = 256
    learning_rate: float = 3e-5
    n_optimizer_steps: int = 500
    target_update_interval: int = 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    parser.add_argument("--load")
    args = parser.parse_args()

    config = Experiment_Config()
    if hasattr(config, "seed"):
        torch.manual_seed(config.seed)
    task = config.task
    task0 = task(**config.rl_task_args)

    device = args.device if args.device else config.device
    torch.set_default_device(device)
    config.device = device

    base_model_path = SAVE_PATH + f"basemodel_{device}.pkl"

    model = config.agent(task0.observation_space.shape[0], task0.action_space.n, **asdict(dqn_hyperpars())).to(device)

    print("Using device:", device)
    print("Config:", config)

    wandb.init(
        project="NPDE",
        name=f"NPDE_{device}_{config.notes}",
        settings=wandb.Settings(
            log_internal=str(Path(__file__).parent / "wandb" / "null"),
            _disable_stats=True,
            _disable_meta=True,
        ),
    )

    print("Gathering data")
    env_instance = task(**config.rl_task_args)
    # Training
    for ep in range(config.gather_length):
        t0_reset = time.time()
        obs = env_instance.reset()
        obs_torch = torch.FloatTensor(obs).to(device).requires_grad_(False)
        score = 0
        t0 = time.time()
        model.rnn.h *= 0

        for step in count():
            t0_step = time.time()
            with torch.no_grad():
                action = model.act_epsilon(obs_torch)
                obs, reward, done, _ = env_instance.step(action)
                obs_torch = torch.FloatTensor(obs).to(device)
            model.remember(action, reward, done)
            score += reward
            if done:
                wandb.log({"RL Score:": score}, step=ep)
                wandb.log({"Compute time (episode)": time.time() - t0}, step=ep)
                break

    print("Training")
    env_instance = task(**config.rl_task_args)
    eps = 0.05
    ep0 = ep
    # Training
    for ep in range(ep0, config.rl_iters):
        if ep % 100 == 0:
            env_instance.test = True
        else:
            env_instance.test = False
        t0_reset = time.time()
        obs = env_instance.reset()
        obs_torch = torch.FloatTensor(obs).to(device).requires_grad_(False)
        score = 0
        t0 = time.time()
        model.rnn.h *= 0
        images = []
        img = env_instance.render()
        images.append(Image.fromarray(img.astype("uint8")))
        for step in count():
            t0_step = time.time()
            with torch.no_grad():
                action = (
                    model.act_epsilon(obs_torch)
                    if (torch.rand(1) < eps) and not env_instance.test
                    else model.act_policy(obs_torch)
                )
                obs, reward, done, _ = env_instance.step(action)
                obs_torch = torch.FloatTensor(obs).to(device)
            model.remember(action, reward, done)
            score += reward

            for j in range(4):
                if ep > 100 and ep % (250 + j) == 0:
                    img = env_instance.render()
                    images.append(Image.fromarray(img.astype("uint8")))
            if env_instance.test:
                img = env_instance.render()
                images.append(Image.fromarray(img.astype("uint8")))

            if done:
                if not env_instance.test:
                    if ep % 10 == 0:
                        model.update()
                        torch.save(model.state_dict(), f=SAVE_PATH + f"/dqn_{device}.pkl")
                    wandb.log({"RL Score:": score}, step=ep)
                    wandb.log({"Compute time (episode)": time.time() - t0}, step=ep)
                    for j in range(4):
                        if ep > 100 and ep % (250 + j) == 0:
                            images[0].save(
                                f"ood/gif/train/dqn_a2b_{device}_{j}.gif",
                                save_all=True,
                                append_images=images[1:],
                                optimize=False,
                                duration=100,
                                loop=0,
                            )
                else:
                    wandb.log({"RL Score (Test):": score}, step=ep)
                    wandb.log({"Compute time (episode) (Test)": time.time() - t0}, step=ep)
                    images[0].save(
                        f"ood/gif/test/dqn_a2b_{device}.gif",
                        save_all=True,
                        append_images=images[1:],
                        optimize=False,
                        duration=100,
                        loop=0,
                    )
                break

    print("Training finished.")
