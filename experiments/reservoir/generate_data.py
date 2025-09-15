from collections import OrderedDict

import torch
import torch.nn as nn

from reservoirs.test_memory_length_function import MemoryLengthDiagnosticData, MLDD_Linear, MLDD_Nonlinear

ITERS = 80
INTERVALS = (2, 4, 16, 32, 64, 128)
INPUT_LEN = 4
SEQ_LEN = 256
RNN_SIZE = 1024
SEED = 1234

torch.manual_seed(SEED)
print("Generating data.")
data = []
data_generator = MLDD_Linear(INPUT_LEN, INTERVALS, 1, 0.5, SEQ_LEN)
_ = data_generator.gen_data(SEQ_LEN)
_ = data_generator.gen_data(SEQ_LEN)

for _ in range(ITERS):
    data.append(data_generator.gen_data(SEQ_LEN))
print("Finished generating data.")

torch.save(data, "spiking_evo_data.pklk")
