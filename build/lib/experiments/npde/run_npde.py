import argparse
import time
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path

import torch

# from tasks import WaterMazeEasy
from env.grid_a_to_b import AtoB
from npde_policy import NPDE_Agent
from rdqn_policy import Agent

import wandb

LOAD_PATH = "ood/saves/dqn_cuda:4 copy.pkl"


@dataclass
class Experiment_Config:
    notes = "NPDE_policy_test"
    task = AtoB
    agent = Agent
    rl_iters = 10000000
    rl_task_args = {"render_mode": "rgb_array"}
    device = "cuda:1"
    gather_length = 2000


@dataclass
class dqn_hyperpars:
    memory_length: int = 100000
    gamma: float = 0.975
    layer_size: int = 256
    learning_rate: float = 3e-5
    n_optimizer_steps: int = 500
    target_update_interval: int = 10


class npde_hyperpars:
    n_training_iters: int = 5
    batch_size: int = 1024
    memory_length: int = 4096 * 4
    lr: float = 1e-4
    weight_decay: float = 0
    min_lateral_sample: int = 10
    max_lateral_sample: int = 256
    solving_steps: int = 10
    topk: int = 10


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

    # Load the model
    model = config.agent(task0.observation_space.shape[0], task0.action_space.n, **asdict(dqn_hyperpars())).to(device)
    model.load_state_dict(torch.load(LOAD_PATH))
    model_npde = NPDE_Agent(model.model, npde_hyperpars())

    print("Using device:", device)
    print("Config:", config)

    wandb.init(
        project="NPDE",
        name=f"NPDE_Test_{device}_{config.notes}",
        settings=wandb.Settings(
            log_internal=str(Path(__file__).parent / "wandb" / "null"),
            _disable_stats=True,
            _disable_meta=True,
        ),
    )
    # define which metrics will be plotted against it
    wandb.define_metric("MSE step")
    wandb.define_metric("Test step")
    wandb.define_metric("Test step (Null)")
    wandb.define_metric("MSE", step_metric="MSE step")
    wandb.define_metric("Test score", step_metric="Test step")
    wandb.define_metric("Test score (Null)", step_metric="Test step")

    # Gather data:
    wandb_step_mse = 0
    wandb_step_test = 0
    wandb_step_test_null = 0

    while True:
        for _ in range(10):
            # print("Gathering data")
            env_instance = task(**config.rl_task_args)
            env_instance.test = False

            model_npde.clear_memory()
            # Training
            while not model_npde.mem_q.wrapped:
                # print("Training ep", ep)
                t0_reset = time.time()
                obs = env_instance.reset()
                obs_torch = torch.FloatTensor(obs).to(device).requires_grad_(False)
                score = 0
                t0 = time.time()
                model.rnn.h *= 0
                for step in count():
                    t0_step = time.time()
                    with torch.no_grad():
                        action = model.act_policy(obs_torch)
                        q = model.model(model.rnn.h)
                        obs, reward, done, _ = env_instance.step(action)
                        obs_torch = torch.FloatTensor(obs).to(device)
                    model.remember(action, reward, done)
                    model_npde.collect(model.rnn.h.clone(), q.clone())
                    score += reward
                    if done:
                        break

            # print("Training NPDE")
            for _ in range(model_npde.n_training_iters):
                mse = model_npde.train()

                train_log = {
                    "MSE step": wandb_step_mse,
                    "MSE": mse,
                }
                wandb.log(train_log)
                wandb_step_mse += 1

        # print("Testing NPDE")
        env_instance = task(**config.rl_task_args)
        env_instance.test = True
        for ep in range(20):
            t0_reset = time.time()
            obs = env_instance.reset()
            obs_torch = torch.FloatTensor(obs).to(device).requires_grad_(False)
            score = 0
            t0 = time.time()
            model.rnn.h *= 0

            for step in count():
                t0_step = time.time()
                with torch.no_grad():
                    _ = model.act_policy(obs_torch)
                    action = model_npde(model.rnn.h.unsqueeze(0)).argmax(-1)
                    obs, reward, done, _ = env_instance.step(action)
                    obs_torch = torch.FloatTensor(obs).to(device)
                    model.remember(action, reward, done)
                score += reward
                if done:
                    test_log = {
                        "Test step": wandb_step_test,
                        "Test score": score,
                        "Compute time (episode) (Test)": time.time() - t0,
                    }
                    wandb.log(test_log)
                    wandb_step_test += 1
                    break

        # print("Testing NPDE")
        env_instance = task(**config.rl_task_args)
        env_instance.test = True
        for ep in range(20):
            t0_reset = time.time()
            obs = env_instance.reset()
            obs_torch = torch.FloatTensor(obs).to(device).requires_grad_(False)
            score = 0
            t0 = time.time()
            model.rnn.h *= 0
            for step in count():
                t0_step = time.time()
                with torch.no_grad():
                    action = model.act_policy(obs_torch)
                    obs, reward, done, _ = env_instance.step(action)
                    obs_torch = torch.FloatTensor(obs).to(device)
                    model.remember(action, reward, done)
                score += reward
                if done:
                    test_log = {
                        "Test step (Null)": wandb_step_test_null,
                        "Test score (Null)": score,
                        "Compute time (episode) (Test, Null)": time.time() - t0,
                    }
                    wandb.log(test_log)
                    wandb_step_test_null += 1
                    break
