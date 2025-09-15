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
from reservoirs.tanh_module import Res2D

"""This goal of this script is to get a basic handle on evolving a neural network architecture and some arbitrarily defined set of hyperparameters.
This will be used to optimize the basic reservoir agent to learn as fast as possible."""
RNN_SIZE = 2048

POP_SIZE = 24
TOURNAMENT_SIZE = 5
NUM_ELITES = 2
MUT_SD = 0.1
MUT_P = 0.1
NUM_GPU = 4
NUM_ACTORS = 12

# ray.init(address="set-falcon.astera-infra.com:6379")


def transform_unit_params(x):
    tanh_pars_evo_min = OrderedDict(
        {
            "seed": 0,
            "dim": 5,
            "n_channel": 1,
            "input_square_dim": 3,
            "output_square_dim": 0,
            "internal_channels": 1,
            "input_scale": 0.1,
            "s1": -1.0,
            "s2": -1.0,
            "k1": 3,
            "k2": 3,
            "d1": 1,
            "d2": 1,
            "p1": 0.2,
            "p2": 0.2,
            "p_mask_coarse": 0.0,
            "p_mask_fine": 0.0,
        }
    )

    tanh_pars_evo_max = OrderedDict(
        {
            "seed": 99999,
            "dim": 7,
            "n_channel": 3,
            "input_square_dim": 5,
            "output_square_dim": 5,
            "internal_channels": 12,
            "input_scale": 10,
            "s1": 1.0,
            "s2": 1.0,
            "k1": 3,
            "k2": 3,
            "d1": 11,
            "d2": 31,
            "p1": 1.0,
            "p2": 1.0,
            "p_mask_coarse": 1.0,
            "p_mask_fine": 1.0,
        }
    )

    integer_odd_pars = [
        "k1",
        "k2",
    ]
    integer_pars = [
        "seed",
        "d1",
        "d2",
        "n_channel",
        "internal_channels",
        "dim",
        "input_square_dim",
        "output_square_dim",
    ]
    integer_2powx_pars = [
        "dim",
        "input_square_dim",
        "output_square_dim",
    ]

    params_raw = x.clone()
    params_raw[params_raw < 0] = 0
    params_raw[params_raw > 1] = 1

    mins = torch.tensor(list(tanh_pars_evo_min.values())).to(params_raw.device)
    maxs = torch.tensor(list(tanh_pars_evo_max.values())).to(params_raw.device)
    tparams = params_raw * (maxs - mins) + mins
    params_dict = {k: v.item() for k, v in zip(tanh_pars_evo_min.keys(), tparams)}

    for par in integer_pars:
        params_dict[par] = int(params_dict[par])

    for par in integer_odd_pars:
        params_dict[par] = int(params_dict[par] // 2 * 2 + 1)  # Get odd num one down or same

    for par in integer_2powx_pars:
        params_dict[par] = int(2 ** params_dict[par])

    return OrderedDict(params_dict)


# transform_unit_params(torch.zeros(14))

# transform_unit_params(torch.zeros(14) + 1)
# RNN_SIZE // (32**2)


# Params is a tensor
def problem_wrapper(params):
    tanh_pars = OrderedDict(
        {
            "seed": 12345,
            "dim": 64,
            "n_channel": 1,
            "input_square_dim": 16,
            "output_square_dim": 32,
            "internal_channels": 1,
            "input_scale": 1,
            "s1": 1.0,
            "s2": 0.0,
            "k1": 3,
            "k2": 3,
            "d1": 1,
            "d2": 31,
            "p1": 0.5,
            "p2": 0.5,
        }
    )
    # Transform the input parameters
    pars = params.params.clone()

    tpars = transform_unit_params(pars)
    tanh_pars.update(tpars)
    # model = torch.load("base_model_tanh_evo.pklk", map_location=pars.device)

    data = torch.load("spiking_evo_data.pklk", map_location=pars.device)

    # for seg in [0, 50, -10]:
    # print(data[seg].std(0))

    model = Res2D(
        input_dim=data[0].shape[1],
        output_size=RNN_SIZE // (tanh_pars["output_square_dim"] ** 2),
        **tanh_pars,
    ).to(pars.device)

    score = TestMemoryLengthLeastSquares(
        model,
        data,
        device=pars.device,
    )
    penalty = (pars > 1).sum() + (pars < 0).sum()
    print(score)
    return score.mean().item() + penalty.item()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load", action="store_true")

    args = parser.parse_args()

    load_vectors = [
        torch.tensor(
            [
                0.4662,
                0.5985,
                0.7606,
                0.0629,
                0.3299,
                0.0432,
                0.0058,
                0.0065,
                0.4995,
                0.2341,
                0.6014,
                0.4536,
                0.6369,
                0.1063,
                0.3910,
                0.7799,
                0.4484,
            ]
        )
    ]

    n_pars = 17

    class paramsModule(nn.Module):
        def __init__(self, len=n_pars):
            super().__init__()
            self.params = nn.Parameter(torch.rand(len))

    parmod = paramsModule(n_pars)

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

    # Load:
    if args.load:
        for i, vec in enumerate(load_vectors):
            print("Loading vector...")
            searcher.step()
            searcher.population.set_values(values=vec, solutions=i)
            searcher.problem.evaluate(searcher.population[i])
            print("Loaded vector evaluated.")

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
