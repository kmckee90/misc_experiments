# import numpy as np
import time

import torch
import torch.nn.functional as F

# from evotorch.tools import device_of
from modules import ReservoirBatch
from reservoirs.tanh_module import Res2D
from reservoirs.test_memory_length_function import MLDD_Linear, MLDD_Nonlinear  # noqa: F401

"""This goal of this script is to get a basic handle on evolving a neural network architecture and some arbitrarily defined set of hyperparameters.
This will be used to optimize the basic reservoir agent to learn as fast as possible."""
ITERS = 80
# INTERVALS = (1, 10, 30, 50, 70, 90, 110, 120)
INTERVALS = (2, 4, 16, 32, 64, 128)

INPUT_LEN = 4
SEQ_LEN = 256
RNN_SIZE = 1024 * 2
SEED = 1234

device = "cuda:0"

print("Generating data.")
data = []
data_generator = MLDD_Linear(INPUT_LEN, INTERVALS, 1, 0.5, SEQ_LEN)
_ = data_generator.gen_data(SEQ_LEN)
_ = data_generator.gen_data(SEQ_LEN)

for _ in range(ITERS):
    data.append(data_generator.gen_data(SEQ_LEN))
print("Finished generating data.")


# print("Loading data")
# data = torch.load("spiking_evo_data.pklk", map_location=device)


print("Data std. deviation:")
for i in [ITERS - 10, ITERS - 50, 0]:
    print(data[i].std(0))

spiking_pars = {
    "dim": 128,
    "input_square_dim": 32,
    "internal_channels": 1,
    "decay": 0.92461,
    "firing_threshold": 0.52287,
    "reset_point": -0.42122,
    "input_split": 0.19294,
    "drop_prob": 0.00194,
    "lower_threshold": -0.31826,
    "exc_local_scale": 0.85558,
    "inh_local_scale": -1.92456,
    "kernel_size_exc_local": 9,
    "kernel_dilation_exc_local": 3,
    "kernel_size_inh_local": 3,
    "kernel_dilation_inh_local": 3,
    "input_mask_prob_fine": 1,
    "input_mask_prob_coarse": 1,
    "input_scale": 0.1,
}

scaler = 2
Res2D_pars = {
    "seed": 46618,
    "dim": 64,
    "n_channel": 2,
    "input_square_dim": 8,
    "output_square_dim": 1,
    "internal_channels": 1,
    "input_scale": 0.15772,
    "s1": -0.987,
    "s2": -0.00104,
    "k1": 3,
    "k2": 3,
    "d1": 5,
    "d2": 20,
    "p1": 0.28507,
    "p2": 0.51284,
    "p_mask_coarse": 0.77992,
    "p_mask_fine": 0.44837,
}

model1 = Res2D(
    input_dim=data[0].shape[1],
    output_size=RNN_SIZE // (Res2D_pars["output_square_dim"] ** 2),
    **Res2D_pars,
).to(device)

torch.save(model1, "ESH_Conv2D_Evolved.pkl")

model2 = ReservoirBatch(data[0].shape[1], hidden_size=RNN_SIZE).to(device)


models = [model1, model2]

X = data
training_sets = X[: (ITERS - 10)]
test_sets = X[(ITERS - 10) :]
training_data = torch.cat(training_sets, 0).to(device)
test_data = torch.cat(test_sets, 0).to(device)

for model in models:
    print("Training", type(model))
    training_data_input = training_data[:-1]
    training_data_output = training_data[1:]

    # Training
    Z = model(training_data_input)
    print(Z.shape)
    Z = torch.cat([torch.ones(Z.shape[0], 1).to(device), Z], 1)
    W = torch.pinverse(Z) @ training_data_output

    # Testing
    test_data_input = test_data[:-1]
    test_data_output = test_data[1:]

    t0 = time.time()
    Z = model(test_data_input)
    Z = torch.cat([torch.ones(Z.shape[0], 1).to(device), Z], 1)
    y_pred = Z @ W
    run_time = time.time() - t0
    print("Run time:", run_time)

    ploss1 = torch.zeros(len(INTERVALS)).to(device)
    for j in range(len(INTERVALS)):
        interval = range(j * INPUT_LEN, (j + 1) * INPUT_LEN)
        ploss1[j] = F.mse_loss(y_pred[:, interval], test_data_output[:, interval])

    print(ploss1)


# (y_pred - test_data_output).pow(2).mean(0)
