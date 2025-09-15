import torch
import torch.nn as nn

with torch.no_grad():
    D_x = 8
    D_z = 1
    min_lateral_sample = 128
    max_lateral_sample = 512
    batch_size = 64
    n = 15000

    zfunc = nn.Identity(D_x, D_z)
    # xfunc = nn.Linear(D_u, D_x)
    # zfunc = nn.Linear(D_x, D_z)

    def dgf(x):
        d = x.shape[-1]
        z = torch.zeros(len(x))
        for i in range(d):
            z += torch.sin(torch.pi * x[:, i])
        return (z - z.mean()) / z.std()

    # The data has an ODE structure over z:
    x = torch.randn(n, D_x)  # Coordinates in R1
    z = dgf(x)  # ODE z(u)


class mlp(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)


class mlp_large(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)


# Now, given only X, can we model the state space of z?
x_to_z = mlp(D_x, D_z)


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Vz = mlp_large(D_z, D_z)
        self.dx = mlp(D_x, 1)


m = model()
optim_F = torch.optim.Adam(x_to_z.parameters(), lr=3e-4)
optim_L = torch.optim.Adam(m.parameters(), lr=3e-4)
mse_mean_L = 0
mse_mean_F = 0

for i in range(3000):
    # Process forward: Autoencode x as z.
    idx_F = torch.randperm(len(x))[:batch_size]
    xt = x[idx_F]
    zt = x_to_z(xt)

    # Process lateral
    n_lateral_sample = torch.randint(min_lateral_sample, max_lateral_sample, ()).item()
    idx_L = torch.randperm(len(x))[:n_lateral_sample]
    x_sample = x[idx_L]
    z_sample = z[idx_L].unsqueeze(1)

    # Alternative approach
    dx_input = xt.unsqueeze(1) - x_sample.unsqueeze(0)
    dx = m.dx(dx_input).squeeze()
    zz = dx @ m.Vz(z_sample) / n_lateral_sample

    # Loss
    mse_F = torch.nn.functional.mse_loss(zt.squeeze(), z[idx_F].squeeze())
    mse_L = torch.nn.functional.mse_loss(zz.squeeze(), z[idx_F].squeeze())
    mse_F.backward()
    mse_L.backward()
    optim_F.step()
    optim_L.step()
    optim_F.zero_grad()
    optim_L.zero_grad()

    mse_mean_F += mse_F.item() / 100
    mse_mean_L += mse_L.item() / 100
    if i % 100 == 0:
        print(i, mse_mean_F, mse_mean_L)
        mse_mean_F = 0
        mse_mean_L = 0


print("Testing Zero-Shot: Using large sample of in-distribution data")
# The data has an ODE structure:
n = 1000
xtest = torch.randn(n, D_x) + 2  # Coordinates in R1
ztest = dgf(xtest)  # ODE z(u)
n_lateral_sample = 64

batch_size = 1
with torch.no_grad():
    # Inference?
    mse_forward = 0
    mse_lateral = 0
    mse_null = 0
    for i in range(len(xtest)):
        # Process forward
        idx_F = torch.randperm(len(xtest))[:batch_size]
        xt = xtest[idx_F]
        zt = x_to_z(xt)
        mse_forward += torch.nn.functional.mse_loss(zt.squeeze(), ztest[idx_F].squeeze())

        # Process lateral
        idx_L = torch.randperm(len(x))[:n_lateral_sample]
        x_sample = x[idx_L]
        z_sample = z[idx_L].unsqueeze(1)

        # Alternative approach
        dx_input = xt.unsqueeze(1) - x_sample.unsqueeze(0)
        dx = m.dx(dx_input).squeeze()
        zz = dx @ m.Vz(z_sample) / n_lateral_sample

        mse_lateral += torch.nn.functional.mse_loss(zz.squeeze(), ztest[idx_F].squeeze())
        mse_null += torch.nn.functional.mse_loss(
            x_to_z(xtest[(idx_F + 1) % len(xtest)]).squeeze(), ztest[idx_F].squeeze()
        )

    print(
        i,
        f"Forward:{mse_forward.item() / len(xtest)}, \tLateral:{mse_lateral.item() / len(xtest)}, \tNull:{mse_null / len(xtest)}",
    )


print("Testing Few-Shot: Using small sample of out-of-distribution data")
# The data has an ODE structure:
n = 1000
xtest = torch.randn(n, D_x) + 2  # Coordinates in R1
ztest = dgf(xtest)  # ODE z(u)
batch_size = 1
n_lateral_sample = 16

with torch.no_grad():
    # Inference?
    mse_forward = 0
    mse_lateral = 0
    mse_null = 0
    for i in range(len(xtest)):
        # Process forward
        idx_F = torch.randperm(len(xtest))[:batch_size]
        xt = xtest[idx_F]
        zt = x_to_z(xt)

        # Process lateral
        idx_L = torch.randperm(len(xtest))[:n_lateral_sample]
        x_sample = xtest[idx_L]
        z_sample = ztest[idx_L].unsqueeze(1)

        # Alternative approach
        dx_input = xt.unsqueeze(1) - x_sample.unsqueeze(0)
        dx = m.dx(dx_input).squeeze()
        zz = dx @ m.Vz(z_sample) / n_lateral_sample

        mse_forward += torch.nn.functional.mse_loss(zt.squeeze(), ztest[idx_F].squeeze())
        mse_lateral += torch.nn.functional.mse_loss(zz.squeeze(), ztest[idx_F].squeeze())
        mse_null += torch.nn.functional.mse_loss(
            x_to_z(xtest[(idx_F + 1) % len(xtest)]).squeeze(), ztest[idx_F].squeeze()
        )

    print(
        i,
        f"Forward:{mse_forward.item() / len(xtest)}, \tLateral:{mse_lateral.item() / len(xtest)}, \tNull:{mse_null / len(xtest)}",
    )

print("Testing Many-Shot: Using large sample of out-of-distribution data")
# The data has an ODE structure:
n = 1000
xtest = torch.randn(n, D_x) + 2  # Coordinates in R1
ztest = dgf(xtest)  # ODE z(u)
batch_size = 1
n_lateral_sample = 64

with torch.no_grad():
    # Inference?
    mse_forward = 0
    mse_lateral = 0
    mse_null = 0
    for i in range(len(xtest)):
        # Process forward
        idx_F = torch.randperm(len(xtest))[:batch_size]
        xt = xtest[idx_F]
        zt = x_to_z(xt)

        # Process lateral
        idx_L = torch.randperm(len(xtest))[:n_lateral_sample]
        x_sample = xtest[idx_L]
        z_sample = ztest[idx_L].unsqueeze(1)

        # Alternative approach
        dx_input = xt.unsqueeze(1) - x_sample.unsqueeze(0)
        dx = m.dx(dx_input).squeeze()
        zz = dx @ m.Vz(z_sample) / n_lateral_sample

        mse_forward += torch.nn.functional.mse_loss(zt.squeeze(), ztest[idx_F].squeeze())
        mse_lateral += torch.nn.functional.mse_loss(zz.squeeze(), ztest[idx_F].squeeze())
        mse_null += torch.nn.functional.mse_loss(
            x_to_z(xtest[(idx_F + 1) % len(xtest)]).squeeze(), ztest[idx_F].squeeze()
        )

    print(
        i,
        f"Forward:{mse_forward.item() / len(xtest)}, \tLateral:{mse_lateral.item() / len(xtest)}, \tNull:{mse_null / len(xtest)}",
    )
