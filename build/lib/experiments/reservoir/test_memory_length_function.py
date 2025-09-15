from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

"""This goal of this document is to get a basic handle on evolving a neural network architecture and some arbitrarily defined set of hyperparameters.
This will be used to optimize the basic reservoir agent to learn as fast as possible."""


class MemoryLengthDiagnosticData:
    def __init__(
        self,
        sub_state_length,
        dependent_intervals,
        noise_interval=10,
        noise=0.1,
        hist_length=256,
    ):
        self.hist_len = hist_length
        self.hist = deque([], maxlen=self.hist_len)

        self.sub_state_length = sub_state_length
        self.dependent_intervals = dependent_intervals
        self.state_length = self.sub_state_length * len(dependent_intervals)
        self.data_model = nn.Sequential(
            nn.Linear(self.state_length, self.sub_state_length),
            nn.Tanh(),
        )

        self.norm = nn.LayerNorm((self.sub_state_length,))

        for _ in range(self.hist_len):
            self.hist.appendleft(torch.randn(self.state_length))

        self.noise = noise
        self.noise_interval = noise_interval

    def gen_data(self, n):
        with torch.no_grad():
            for _ in range(n):
                old_y = []
                new_y = []
                for interval in self.dependent_intervals:
                    old_y = self.hist[interval]
                    new_y.append(self.norm(self.data_model(old_y).squeeze()))
                s = torch.cat(new_y)
                x = torch.randn_like(s)
                y = (1 - self.noise) * s + self.noise * x
                self.hist.appendleft(y)
            data = torch.flip(torch.stack(list(self.hist))[:n], (0,))
        return data


class MLDD_Linear:
    def __init__(
        self,
        sub_state_length,
        dependent_intervals,
        noise_interval=10,
        noise=0.1,
        hist_length=256,
    ):
        self.hist_len = hist_length
        self.hist = deque([], maxlen=self.hist_len)

        self.sub_state_length = sub_state_length
        self.dependent_intervals = dependent_intervals
        self.state_length = self.sub_state_length * len(dependent_intervals)

        for _ in range(self.hist_len):
            self.hist.appendleft(torch.randn(self.state_length))

        self.noise = noise

    def gen_data(self, n):
        with torch.no_grad():
            for _ in range(n):
                old_y = []
                new_y = []
                for j, interval in enumerate(self.dependent_intervals):
                    elements = range(j * self.sub_state_length, (j + 1) * self.sub_state_length)
                    old_y = self.hist[interval][elements]
                    new_y.append(old_y)
                s = torch.cat(new_y)
                y = 0.75**0.5 * s + self.noise * torch.randn_like(s)
                self.hist.appendleft(y)
            data = torch.flip(torch.stack(list(self.hist))[:n], (0,))
        return data


def count_pars(model):
    count = 0
    for par in model.parameters():
        if par.requires_grad:
            count += len(par.view(-1))
    return count


class MLDD_Nonlinear:
    def __init__(
        self,
        sub_state_length,
        dependent_intervals,
        noise_interval=10,
        noise=0.1,
        hist_length=256,
    ):
        self.hist_len = hist_length
        self.hist = deque([], maxlen=self.hist_len)

        self.sub_state_length = sub_state_length
        self.dependent_intervals = dependent_intervals
        self.state_length = self.sub_state_length * len(dependent_intervals)
        self.data_model = nn.Sequential(
            nn.Linear(self.state_length, self.sub_state_length * 10),
            nn.LayerNorm((self.sub_state_length * 10,)),
            nn.Tanh(),
            nn.Linear(self.sub_state_length * 10, self.sub_state_length * 10),
            nn.LayerNorm((self.sub_state_length * 10,)),
            nn.Tanh(),
            nn.Linear(self.sub_state_length * 10, self.sub_state_length * 10),
            nn.LayerNorm((self.sub_state_length * 10,)),
            nn.Tanh(),
            nn.Linear(self.sub_state_length * 10, self.sub_state_length),
            nn.LayerNorm((self.sub_state_length,)),
        )

        self.norm = nn.LayerNorm((self.sub_state_length,))

        for _ in range(self.hist_len):
            self.hist.appendleft(torch.randn(self.state_length))

        self.noise = noise
        self.noise_interval = noise_interval

    def gen_data(self, n):
        with torch.no_grad():
            for _ in range(n):
                old_y = []
                new_y = []
                for interval in self.dependent_intervals:
                    old_y = self.hist[interval]
                    new_y.append(self.data_model(old_y).squeeze())
                s = torch.cat(new_y)
                x = torch.randn_like(s)
                y = 0.75**0.5 * s + self.noise * x
                self.hist.appendleft(y)
            data = torch.flip(torch.stack(list(self.hist))[:n], (0,))
        return data


def TestMemoryLength(
    model,
    device,
    iters=50000,
    intervals=(1, 10, 100, 250),
    input_length=8,
    seq_len=1024,
    lr=3e-4,
):
    data_generator = MemoryLengthDiagnosticData(
        input_length,
        dependent_intervals=intervals,
        noise_interval=1,
        noise=0.5,
        hist_length=seq_len,
    )
    _ = data_generator.gen_data(seq_len)
    m1 = model
    optimizer1 = torch.optim.Adam(m1.parameters(), lr=lr)
    for _ in range(iters):
        y = data_generator.gen_data(seq_len).to(device)
        y_prev = y[:-1]
        y_targ = y[1:]

        optimizer1.zero_grad()
        pred1 = m1(y_prev)
        loss1 = F.mse_loss(pred1, y_targ)
        loss1.backward()
        optimizer1.step()

        ploss1 = torch.zeros(len(intervals))
        for j in range(len(intervals)):
            interval = range(j * input_length, (j + 1) * input_length)
            ploss1[j] = F.mse_loss(pred1[:, interval], y_targ[:, interval])

    return ploss1


def TestMemoryLengthPregen(
    model,
    data,
    device,
    iters=5000,
    intervals=(1, 10, 100, 250),
    input_length=8,
    seq_len=1024,
    lr=3e-4,
):
    m1 = model
    optimizer1 = torch.optim.Adam(m1.parameters(), lr=lr)
    for i in range(iters):
        y = data[i % len(data)]
        y_prev = y[:-1]
        y_targ = y[1:]

        optimizer1.zero_grad()
        pred1 = m1(y_prev)
        loss1 = F.mse_loss(pred1, y_targ)
        loss1.backward()
        optimizer1.step()

        ploss1 = torch.zeros(len(intervals))
        for j in range(len(intervals)):
            interval = range(j * input_length, (j + 1) * input_length)
            ploss1[j] = F.mse_loss(pred1[:, interval], y_targ[:, interval])

    return ploss1


def TestMemoryLengthLeastSquares(
    model,
    data,
    device,
    num_intervals=6,
    elem_per_interval=4,
):
    X = data
    training_sets = X[:-10]  # 70: 33gb 100: 47gb
    test_sets = X[-10:]

    training_data = torch.cat(training_sets, 0).to(device)
    training_data_input = training_data[:-1]
    training_data_output = training_data[1:]

    # Training
    Z = model(training_data_input)
    Z = torch.cat([torch.ones(Z.shape[0], 1).to(device), Z], 1)
    try:
        W = torch.pinverse(Z) @ training_data_output
    except:  # noqa: E722
        return torch.ones(num_intervals)

    # Testing
    test_data = torch.cat(test_sets, 0).to(device)
    test_data_input = test_data[:-1]
    test_data_output = test_data[1:]

    Z = model(test_data_input)
    Z = torch.cat([torch.ones(Z.shape[0], 1).to(device), Z], 1)
    y_pred = Z @ W

    ploss1 = torch.zeros(num_intervals).to(device)
    for j in range(num_intervals):
        interval = range(j * elem_per_interval, (j + 1) * elem_per_interval)
        ploss1[j] = F.mse_loss(y_pred[:, interval], test_data_output[:, interval])

    return ploss1
