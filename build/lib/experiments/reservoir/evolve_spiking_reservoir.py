import time
from collections import OrderedDict

import evotorch.algorithms as algorithms

# import numpy as np
import torch
import torch.nn as nn
from evotorch.logging import StdOutLogger, WandbLogger

# import torch.nn.functional as F
# from evotorch import Problem
from evotorch.neuroevolution import NEProblem
from evotorch.tools import device_of

import base_model as base_model
from reservoirs.test_memory_length_function import TestMemoryLength
from reservoirs.spiking_module import Cortex

"""This goal of this script is to get a basic handle on evolving a neural network architecture and some arbitrarily defined set of hyperparameters.
This will be used to optimize the basic reservoir agent to learn as fast as possible."""

# Fixed eval  parameters
ITERS = 1000
INTERVALS = (1, 10, 50, 100, 250)
INPUT_LEN = 8
SEQ_LEN = 256
LR = 3e-4
HID_SIZE = 256
RNN_SIZE = 1024


POP_SIZE = 32
TOURNAMENT_SIZE = 4
NUM_ELITES = 1
MUT_SD = 0.2
MUT_P = 1.0
# Set up a base model to load to save time


def transform_unit_params(params_raw):
    spiking_pars_evo_min = OrderedDict(
        {
            "decay": 0.5,
            "firing_threshold": 0.5,
            "reset_point": -0.5,
            "input_split": 0.0,
            # Random vars
            "drop_prob_min": 0.0,
            "drop_prob_max": 0.0,
            "lower_threshold_min": -0.2,
            "lower_threshold_max": -0.2,
            "exc_local_scale_min": 0,
            "exc_local_scale_max": 0,
            "inh_local_scale_min": -10,
            "inh_local_scale_max": -10,
            "exc_global_scale_min": 0,
            "exc_global_scale_max": 0,
            # Architecture
            "kernel_size_exc_local": 3,
            "kernel_dilation_exc_local": 1,
            "kernel_size_inh_local": 3,
            "kernel_dilation_inh_local": 1,
            "kernel_size_exc_global": 5,
            "kernel_dilation_exc_global": 1,
            # Input params: how inputs perturb the reservoir spatially
            "input_mask_prob_fine": 0.0,
            "input_mask_prob_coarse": 0.0,
            "input_scale": 0,
        }
    )

    spiking_pars_evo_max = OrderedDict(
        {
            "decay": 0.99999,
            "firing_threshold": 0.99999,
            "reset_point": 0.0,
            "input_split": 0.5,
            # Random vars
            "drop_prob_min": 1.0,
            "drop_prob_max": 1.0,
            "lower_threshold_min": 0,
            "lower_threshold_max": 0,
            "exc_local_scale_min": 10,
            "exc_local_scale_max": 10,
            "inh_local_scale_min": 0,
            "inh_local_scale_max": 0,
            "exc_global_scale_min": 10,
            "exc_global_scale_max": 10,
            # Architecture
            "kernel_size_exc_local": 21,
            "kernel_dilation_exc_local": 5,
            "kernel_size_inh_local": 21,
            "kernel_dilation_inh_local": 5,
            "kernel_size_exc_global": 21,
            "kernel_dilation_exc_global": 5,
            # Input params: how inputs perturb the reservoir spatially
            "input_mask_prob_fine": 1.0,
            "input_mask_prob_coarse": 1.0,
            "input_scale": 5,
        }
    )

    integer_odd_pars = [
        "kernel_size_exc_local",
        "kernel_size_inh_local",
        "kernel_size_exc_global",
    ]
    integer_pars = [
        "kernel_dilation_exc_global",
        "kernel_dilation_inh_local",
        "kernel_dilation_exc_local",
    ]

    params_raw = torch.sigmoid(params_raw)
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
            "decay": 0.99,
            "firing_threshold": 0.99,
            "reset_point": -0.10,
            "input_split": 0.10,
            # Random vars
            "drop_prob_min": 0.8,
            "drop_prob_max": 0.8,
            "lower_threshold_min": -0.0925,
            "lower_threshold_max": -0.0925,
            "exc_local_scale_min": 4,
            "exc_local_scale_max": 4,
            "inh_local_scale_min": -3,
            "inh_local_scale_max": -3,
            "exc_global_scale_min": 1,
            "exc_global_scale_max": 1,
            # Architecture
            "kernel_size_exc_local": 7,
            "kernel_dilation_exc_local": 1,
            "kernel_size_inh_local": 3,
            "kernel_dilation_inh_local": 1,
            "kernel_size_exc_global": 5,
            "kernel_dilation_exc_global": 2,
            # Input params: how inputs perturb the reservoir spatially
            "input_square_dim": 32,
            "internal_channels": 1,
            "input_mask_prob_fine": 0.2,
            "input_mask_prob_coarse": 1.0,
            "input_scale": 1,
        }
    )
    # Transform the input parameters
    pars = params.params.clone()

    tpars = transform_unit_params(pars)
    spiking_pars.update(tpars)
    model = torch.load("base_model_spiking_evo.pkl", map_location=pars.device)
    model.rnn = Cortex(
        input_dim=INPUT_LEN * len(INTERVALS),
        output_channels=RNN_SIZE,
        output_kernel_size=spiking_pars["dim"],
        **spiking_pars,
    ).to(pars.device)

    score = TestMemoryLength(
        model,
        device=pars.device,
        iters=ITERS,
        intervals=INTERVALS,
        input_length=INPUT_LEN,
        seq_len=SEQ_LEN,
        lr=LR,
    )
    return score.mean().item()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # This model will get loaded in problem_wrapper
    model1 = base_model.Model1(INPUT_LEN * len(INTERVALS), INPUT_LEN * len(INTERVALS), HID_SIZE, RNN_SIZE)
    torch.save(model1, "base_model_spiking_evo.pkl")

    class paramsModule(nn.Module):
        def __init__(self, len=23):
            super().__init__()
            self.params = nn.Parameter(torch.rand(len))

    # parmod = paramsModule(23).to("cuda:0")
    # problem_wrapper(parmod)
    # t0 = time.time()
    # problem_wrapper(parmod)
    # print("CUDA:0 problem function run time:", time.time() - t0)

    # parmod = paramsModule(23).to("cpu")
    # problem_wrapper(parmod)
    # t0 = time.time()
    # problem_wrapper(parmod)
    # print("CPU problem function run time:", time.time() - t0)
    parmod = paramsModule(23)

    problem = NEProblem(
        "min",
        parmod,
        problem_wrapper,
        num_actors=POP_SIZE,
        num_gpus_per_actor=4 / POP_SIZE,
        device="cpu",
    )

    searcher = algorithms.Cosyne(
        problem,
        popsize=POP_SIZE,
        tournament_size=TOURNAMENT_SIZE,
        num_elites=NUM_ELITES,
        mutation_stdev=MUT_SD,
        mutation_probability=MUT_P,
    )

    searcher.step()

    log_interval = 3
    WandbLogger(searcher)
    print("Running evolution.")
    for _ in range(10000):
        t0 = time.time()
        searcher.run(log_interval)
        print("Generation time:", time.time() - t0)

        best_dict = transform_unit_params(searcher.status["best"].values.clone())
        longest_key = max(len(key) for key in best_dict)
        for key, value in best_dict.items():
            key = f'"{key}"'
            print(f"\t{key:<{longest_key}} : {round(value,ndigits=5)},")
