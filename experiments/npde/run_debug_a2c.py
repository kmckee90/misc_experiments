import argparse
import time
from dataclasses import dataclass
from itertools import count
from pathlib import Path

import numpy as np
import torch
from a2c_policy import A2C
from env.grid_a_to_b import AtoB
from PIL import Image

import wandb

SAVE_PATH = "meta_dqn/saves/"


@dataclass
class Experiment_Config:
    notes = "A2C_policy_test_WM"
    task = AtoB
    agent = A2C
    rl_task_args = {"render_mode": "rgb_array"}
    device = "cuda:2"


@dataclass
class a2c_hyperpars:
    # RL params
    lrate: float = 1e-5
    beta_e: float = 1e-4
    max_gradient: float = 1.0
    discount: float = 0.975
    train_every: int = 100
    layer_size: int = 256


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

    model = config.agent(task0.observation_space.shape[0], a2c_hyperpars(), task0.action_space.n).to(device)

    print("Using device:", device)
    print("Config:", config)

    wandb.init(
        project="NPDE",
        name=f"NPDE_A2C_Debug_{device}_{config.notes}",
        settings=wandb.Settings(
            log_internal=str(Path(__file__).parent / "wandb" / "null"),
            _disable_stats=True,
            _disable_meta=True,
        ),
    )

    print("Training")
    env_instance = task(**config.rl_task_args)
    # Training
    for ep in count():
        if ep % 100 == 0:
            env_instance.test = True
        else:
            env_instance.test = False
        obs = env_instance.reset()
        obs_torch = torch.FloatTensor(obs).to(device).requires_grad_(False)
        score = 0
        t0 = time.time()
        model.init_hidden()
        model.rnn.h *= 0
        images = []
        img = env_instance.render()
        images.append(Image.fromarray(img.astype("uint8")))
        for step in count():
            action = model(obs_torch)
            obs, reward, done, _ = env_instance.step(action)
            obs_torch = torch.FloatTensor(obs).to(device)
            model.rewards[0] = np.sign(reward)
            model.rewards[1] = reward
            model.trainer.collect([action], model.v, reward, done)
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
                    model.trainer.collect([action], model.v * 0, reward, True)  # None of this gets used
                    model.trainer.train()
                    model.trainer.clear_history()
                    wandb.log({"RL Score:": score}, step=ep)
                    wandb.log({"Compute time (episode)": time.time() - t0}, step=ep)
                    wandb.log({"Actor Entropy": model.actor.dist.entropy().mean().item()}, step=ep)
                    if hasattr(model.trainer, "critic_loss"):
                        wandb.log({"Critic Loss": model.trainer.critic_loss.item()}, step=ep)
                    for j in range(4):
                        if ep > 100 and ep % (250 + j) == 0:
                            images[0].save(
                                f"ood/gif/train/a2c_a2b_{device}_{j}.gif",
                                save_all=True,
                                append_images=images[1:],
                                optimize=False,
                                duration=100,
                                loop=0,
                            )
                else:
                    model.trainer.clear_history()
                    wandb.log({"RL Score (Test):": score}, step=ep)
                    wandb.log({"Compute time (episode) (Test)": time.time() - t0}, step=ep)
                    images[0].save(
                        f"ood/gif/test/a2c_a2b_{device}.gif",
                        save_all=True,
                        append_images=images[1:],
                        optimize=False,
                        duration=100,
                        loop=0,
                    )
                break

    print("Training finished.")
