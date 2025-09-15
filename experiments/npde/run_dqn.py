import argparse
import time
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path

import torch

# from tasks import WaterMazeEasy
from env.grid_a_to_b_rand import AtoB
from PIL import Image
from rdqn_policy import Agent

import wandb

SAVE_PATH = "meta_dqn/saves/"
# LOAD_PATH = "ood/saves/dqn_cuda:4 copy 2.pkl"
LOAD_PATH = None


@dataclass
class Experiment_Config:
    notes = "DQN_policy_test"
    task = AtoB
    agent = Agent
    rl_iters = 10000000
    rl_task_args = {"render_mode": "rgb_array"}
    device = "cuda:0"
    gather_length = 1000
    eps_train: float = 0.3
    eps_gather: float = 1.0
    eps_gather_loaded: float = 0.05
    train_every: int = 10
    render_every: int = 250


@dataclass
class dqn_hyperpars:
    memory_length: int = int(5e5)
    gamma: float = 0.975
    layer_size: int = 256
    learning_rate: float = 3e-5
    n_optimizer_steps: int = 400
    target_update_interval: int = 4
    minibatch_size: int = 15000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device")
    parser.add_argument("--load")
    args = parser.parse_args()

    config = Experiment_Config()
    device = args.device if args.device else config.device
    config.device = device
    torch.set_default_device(device)

    if hasattr(config, "seed"):
        torch.manual_seed(config.seed)
    task = config.task
    task0 = task(**config.rl_task_args)

    model = config.agent(task0.observation_space.shape[0], task0.action_space.n, **asdict(dqn_hyperpars())).to(device)
    if LOAD_PATH is not None:
        model.load_state_dict(torch.load(LOAD_PATH))
        eps = config.eps_gather_loaded
    else:
        eps = config.eps_gather
        # pars = model.model.state_dict()
        # model.target_model.load_state_dict(pars)

    print("Using device:", device)
    print("Config:", config)

    e_cfg = asdict(Experiment_Config())
    e_cfg = {("experiment_" + k): v for k, v in e_cfg.items()}
    dqn_cfg = asdict(dqn_hyperpars())
    dqn_cfg = {("dqn_" + k): v for k, v in dqn_cfg.items()}
    total_cfg = {**e_cfg, **dqn_cfg}

    wandb.init(
        project="NPDE",
        name=f"NPDE_{device}_{config.notes}",
        config=total_cfg,
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
                if LOAD_PATH is None:
                    action = model.act_epsilon(obs_torch)
                else:
                    action = (
                        model.act_epsilon(obs_torch)
                        if (torch.rand(1) < eps) and not env_instance.test
                        else model.act_policy(obs_torch)
                    )

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
    eps = config.eps_train
    # Training
    ep0 = ep
    for ep in range(ep0, config.rl_iters):
        env_instance.test = ep % 200 == 0
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
                if ep >= config.render_every and ep % (config.render_every + j) == 0:
                    img = env_instance.render()
                    images.append(Image.fromarray(img.astype("uint8")))
            if env_instance.test:
                img = env_instance.render()
                images.append(Image.fromarray(img.astype("uint8")))

            if done:
                if not env_instance.test:
                    if ep % config.train_every == 0:
                        model.update()
                        torch.save(model.state_dict(), f=f"ood/saves/dqn_{device}.pkl")
                        wandb.log(model.train_log, step=ep)

                    wandb.log({"RL Score:": score}, step=ep)
                    wandb.log({"Compute time (episode)": time.time() - t0}, step=ep)
                    for j in range(4):
                        if ep > 0 and ep % (config.render_every + j) == 0:
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
