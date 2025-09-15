import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import utils


class Reservoir(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        spectral_radius=1.0,
        connection_probability=1,
        forward_iters=16,
        input_gain=2.5,
        bias_gain=1,
    ):
        super().__init__()
        self.rnn = nn.RNNCell(input_size, hidden_size)
        self.register_buffer("h", torch.zeros(hidden_size))
        torch.nn.init.xavier_uniform_(self.rnn.weight_hh)
        torch.nn.init.xavier_normal_(self.rnn.weight_ih, gain=input_gain)
        self.rnn.bias_hh = nn.Parameter(torch.zeros_like(self.rnn.bias_hh))
        self.rnn.bias_ih = nn.Parameter(bias_gain * torch.randn_like(self.rnn.bias_ih))
        utils.sparsify_weights(self.rnn, connection_probability=connection_probability)
        utils.set_spectral_radius(self.rnn, hidden_size, radius=spectral_radius)
        self.forward_iters = forward_iters

    def forward(self, x):
        for _ in range(self.forward_iters):
            self.h = self.rnn(x, self.h)
        return self.h


def ridgeinv(X, L=1):
    I = torch.eye(X.shape[1], dtype=torch.double)
    X = X.double()
    pinv = torch.inverse(X.T @ X + L * I) @ X.T
    return pinv.float()


def pinv2(X, L=1):
    X = X.double()
    pinv = torch.inverse(X.T @ X) @ X.T
    return pinv.float()


torch.manual_seed(123)

# Generate a complex nonlinear sequence
d = 1
s = 10000
x_test = 5 * (torch.rand(s, d) * 2 - 1)
x = torch.randn(s, d) * 1
repl = torch.randint(0, s, (s // 100,))
x[repl] = x_test[repl]
y = torch.sin(2 * torch.pi * x / 5) + torch.sin(2 * torch.pi * x / 1) + torch.sin(2 * torch.pi * x / 0.5)
x_test = 5 * (torch.rand(s, d) * 2 - 1)
y_test = (
    torch.sin(2 * torch.pi * x_test / 5) + torch.sin(2 * torch.pi * x_test / 1) + torch.sin(2 * torch.pi * x_test / 0.5)
)

k = 64
rnn = Reservoir(1, k)
rnn.h = rnn.h * torch.ones(s, 1)

rnn.h = rnn.h * 0
z = rnn(x)


W = ridgeinv(z, L=0.0000001) @ y
pred = z @ W

train_mse = torch.nn.functional.mse_loss(pred, y)


plt.scatter(x, y, s=0.5)
plt.scatter(x, pred.detach().cpu(), s=0.5)
plt.xlim(-5, 5)
plt.ylim(-3, 3)
plt.savefig("reservoir_multistep_train.png", dpi=300)
plt.close()


rnn.h = rnn.h * 0
z = rnn(x_test)
pred = z @ W
test_mse = torch.nn.functional.mse_loss(pred, y_test)


plt.scatter(x_test, y_test, s=0.5)
plt.scatter(x_test, pred.detach().cpu(), s=0.5)
plt.xlim(-5, 5)
plt.ylim(-3, 3)
plt.savefig("reservoir_multistep_test.png", dpi=300)
plt.close()
print(train_mse.item(), test_mse.item())


# To test: test MSE | steps, connection prob,
