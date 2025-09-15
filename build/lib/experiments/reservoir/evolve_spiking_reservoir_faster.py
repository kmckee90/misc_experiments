import random
import time
from collections import OrderedDict

import evotorch.algorithms as algorithms
import ray

# import numpy as np
import torch
import torch.nn as nn
from evotorch.logging import StdOutLogger, WandbLogger

# import torch.nn.functional as F
# from evotorch import Problem
from evotorch.neuroevolution import NEProblem

# from evotorch.tools import device_of
import base_model as base_model
from reservoirs.test_memory_length_function import (
    # MemoryLengthDiagnosticData,
    # TestMemoryLength,
    TestMemoryLengthLeastSquares,
)

# TestMemoryLengthPregen,
from reservoirs.spiking_module_faster import Cortex


"""This goal of this script is to get a basic handle on evolving a neural network architecture and some arbitrarily defined set of hyperparameters.
This will be used to optimize the basic reservoir agent to learn as fast as possible."""
RNN_SIZE = 1024

POP_SIZE = 96
TOURNAMENT_SIZE = 5
NUM_ELITES = 2
MUT_SD = 0.1
MUT_P = 0.1
NUM_GPU = 12
NUM_ACTORS = 12

ray.init(address="set-falcon.astera-infra.com:6379")


def transform_unit_params(x):
    spiking_pars_evo_min = OrderedDict(
        {
            # Pars to evolve:
            "decay": 0.9,
            "firing_threshold": 0.25,
            "reset_point": -0.5,
            "input_split": 0.00,
            # Random vars
            "drop_prob": 0.0,
            "lower_threshold": -0.5,
            "exc_local_scale": 0,
            "inh_local_scale": -10,
            # Architecture
            "kernel_size_exc_local": 3,
            "kernel_dilation_exc_local": 1,
            "kernel_size_inh_local": 3,
            "kernel_dilation_inh_local": 1,
            # Input params: how inputs perturb the reservoir spatially
            "input_mask_prob_fine": 0.01,
            "input_mask_prob_coarse": 0.01,
            "input_scale": 0.05,
        }
    )

    spiking_pars_evo_max = OrderedDict(
        {
            # Pars to evolve:
            "decay": 0.99999,
            "firing_threshold": 0.99999,
            "reset_point": 0,
            "input_split": 0.5,
            # Random vars
            "drop_prob": 0.99,
            "lower_threshold": -0.001,
            "exc_local_scale": 10,
            "inh_local_scale": 0,
            # Architecture
            "kernel_size_exc_local": 13,
            "kernel_dilation_exc_local": 5,
            "kernel_size_inh_local": 13,
            "kernel_dilation_inh_local": 5,
            # Input params: how inputs perturb the reservoir spatially
            "input_mask_prob_fine": 1.0,
            "input_mask_prob_coarse": 1.0,
            "input_scale": 2.0,
        }
    )

    integer_odd_pars = [
        "kernel_size_exc_local",
        "kernel_size_inh_local",
    ]
    integer_pars = [
        "kernel_dilation_inh_local",
        "kernel_dilation_exc_local",
    ]

    params_raw = x.clone()
    params_raw[params_raw < 0] = 0
    params_raw[params_raw > 1] = 1

    mins = torch.tensor(list(spiking_pars_evo_min.values())).to(params_raw.device)
    maxs = torch.tensor(list(spiking_pars_evo_max.values())).to(params_raw.device)
    tparams = params_raw * (maxs - mins) + mins
    params_dict = {k: v.item() for k, v in zip(spiking_pars_evo_min.keys(), tparams)}

    for par in integer_pars:
        params_dict[par] = int(params_dict[par])

    for par in integer_odd_pars:
        params_dict[par] = int(params_dict[par] // 2 * 2 + 1)  # Get odd num one down or same

    return OrderedDict(params_dict)


# Params is a tensor
def problem_wrapper(params):
    spiking_pars = OrderedDict(
        {
            "dim": 64,
            "input_square_dim": 32,
            "internal_channels": 1,
            # Pars to evolve:
            "decay": 0.99,
            "firing_threshold": 0.99,
            "reset_point": -0.12,
            "input_split": 0.1,
            # Random vars
            "drop_prob": 0.2,
            "lower_threshold": -0.09,
            "exc_local_scale": 4,
            "inh_local_scale": -2,
            # Architecture
            "kernel_size_exc_local": 5,
            "kernel_dilation_exc_local": 1,
            "kernel_size_inh_local": 5,
            "kernel_dilation_inh_local": 1,
            # Input params: how inputs perturb the reservoir spatially
            "input_mask_prob_fine": 0.25,
            "input_mask_prob_coarse": 0.5,
            "input_scale": 0.5,
        }
    )
    # Transform the input parameters
    pars = params.params.clone()

    tpars = transform_unit_params(pars)
    spiking_pars.update(tpars)
    # model = torch.load("base_model_spiking_evo.pklk", map_location=pars.device)

    data = torch.load("spiking_evo_data.pklk", map_location=pars.device)

    model = Cortex(
        input_dim=data[0].shape[1],
        output_size=RNN_SIZE,
        **spiking_pars,
    ).to(pars.device)

    score = TestMemoryLengthLeastSquares(
        model,
        data,
        device=pars.device,
    )
    penalty = (pars > 1).sum() + (pars < 0).sum()
    return score.mean().item() + penalty.item()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    class paramsModule(nn.Module):
        def __init__(self, len=15):
            super().__init__()
            self.params = nn.Parameter(torch.rand(len))

    parmod = paramsModule(15)

    problem = NEProblem(
        "min",
        parmod,
        problem_wrapper,
        initial_bounds=(0, 1),
        num_actors=NUM_ACTORS,
        num_gpus_per_actor=NUM_GPU / NUM_ACTORS,
        device="cpu",
    )

    searcher = algorithms.CMAES(
        problem,
        popsize=POP_SIZE,
        stdev_init=0.25,
        center_init=0.5,
    )

    log_interval = 1
    WandbLogger(searcher)
    StdOutLogger(searcher)
    print("Running evolution.")
    for _ in range(10000):
        t0 = time.time()
        searcher.run(log_interval)
        print("Generation time:", time.time() - t0)

        best = searcher.status["best"].values.clone()
        best_dict = transform_unit_params(best)
        longest_key = max(len(key) for key in best_dict)
        print("\nALL TIME BEST:")
        for key, value in best_dict.items():
            key = f'"{key}"'
            print(f"\t{key:<{longest_key}} : {round(value,ndigits=5)},")

        with open("evospike_best.txt", "w") as file:
            for key, value in best_dict.items():
                key = f'"{key}"'
                file.write(f"\t{key:<{longest_key}} : {round(value, ndigits=5)},\n")
        print(best)

        print("\nPOPULATION BEST:")
        pop_best = searcher.status["pop_best"].values.clone()
        best_dict = transform_unit_params(pop_best)
        longest_key = max(len(key) for key in best_dict)
        for key, value in best_dict.items():
            key = f'"{key}"'
            print(f"\t{key:<{longest_key}} : {round(value,ndigits=5)},")

        with open("evospike_pop_best.txt", "w") as file:
            for key, value in best_dict.items():
                key = f'"{key}"'
                file.write(f"\t{key:<{longest_key}} : {round(value, ndigits=5)},\n")

        print(pop_best)
