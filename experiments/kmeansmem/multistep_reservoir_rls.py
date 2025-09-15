import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import utils


class RLS(nn.Module):
    def __init__(self, n_features, n_outputs, delta=1.0, lambda_=1.0):
        super().__init__()
        self.n_features = n_features + 1  # Add one for the bias term
        self.n_outputs = n_outputs
        self.lambda_ = lambda_
        self.delta = delta
        self.register_buffer("P", (1.0 / delta) * torch.eye(self.n_features))
        self.register_buffer("beta", torch.zeros(self.n_features, n_outputs))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Ensure x is 2D
        ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=self.beta.device)
        x_aug = torch.cat([ones, x], dim=1)
        pred = torch.mm(x_aug, self.beta)
        return pred

    def update(self, x, y):
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)  # Ensure x is 2D
            if y.dim() == 1:
                y = y.unsqueeze(0)  # Ensure y is 2D
            # Augment x with a column of ones to account for the bias term.
            ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
            x_aug = torch.cat([ones, x], dim=1)

            x_aug = x_aug.double()
            y = y.double()
            self.P = self.P.double()
            self.beta = self.beta.double()

            # Process each observation.
            for xi, yi in zip(x_aug, y):
                xi = xi.unsqueeze(1)  # Convert to column vector
                yi = yi.unsqueeze(1)  # Convert to column vector
                Px = torch.mm(self.P, xi)
                error = yi.T - torch.mm(xi.t(), self.beta)
                gain = Px / (self.lambda_ + torch.mm(xi.t(), Px))
                self.P = (self.P - torch.mm(gain, Px.t())) / self.lambda_
                self.beta += torch.mm(gain, error)

            self.beta = self.beta.float()

    def reset(self):
        self.P = (1.0 / self.delta) * torch.eye(self.n_features, device=self.P.device)
        self.beta = torch.zeros(self.n_features, self.n_outputs, device=self.P.device)


class Reservoir(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        spectral_radius=1.0,
        connection_probability=1,
        forward_iters=2,
        input_gain=10,
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
rls = RLS(k, 1, 0.001, 1)

rnn.h = rnn.h * 0
z = rnn(x)


# Fit model
rls.update(z, y)
W = rls.beta

# Prediction
ones = torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
z_aug = torch.cat([ones, z], dim=1)
pred = z_aug @ W

# Output
train_mse = torch.nn.functional.mse_loss(pred, y)


plt.scatter(x, y, s=0.5)
plt.scatter(x, pred.detach().cpu(), s=0.5)
plt.xlim(-5, 5)
plt.ylim(-3, 3)
plt.savefig("reservoir_multistep_train.png", dpi=300)
plt.close()


rnn.h = rnn.h * 0
z = rnn(x_test)


ones = torch.ones(z.shape[0], 1, dtype=z.dtype, device=z.device)
z_aug = torch.cat([ones, z], dim=1)
pred = z_aug @ W
test_mse = torch.nn.functional.mse_loss(pred, y_test)


plt.scatter(x_test, y_test, s=0.5)
plt.scatter(x_test, pred.detach().cpu(), s=0.5)
plt.xlim(-5, 5)
plt.ylim(-3, 3)
plt.savefig("reservoir_multistep_test.png", dpi=300)
plt.close()
print(train_mse.item(), test_mse.item())


# To test: test MSE | steps, connection prob,
