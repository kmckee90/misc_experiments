import torch
import torch.nn as nn

# This ones uses pytorch's built in jacobian function to supply the first derivatives to the dynamics model.
# It also selects only the closest sample from a larger sample, rather than averaging.


with torch.no_grad():
    D_x = 2
    D_z = 2
    min_lateral_sample = 1
    max_lateral_sample = 1024
    batch_size = 4096
    n = 15000
    n_training_iters = 5000

    zfunc = nn.Identity(D_x, D_z)
    # zfunc = nn.Linear(D_x, D_z)

    # def dgf(x):
    #     d = x.shape[-1]
    #     z = torch.zeros(len(x))
    #     for i in range(d):
    #         z += torch.sin(torch.pi * x[:, i])
    #     return (z - z.mean()) / z.std()

    # # The data has an ODE structure over z:
    # def dgf(x):
    #     z = torch.sin(torch.pi * zfunc(x))
    #     z = (z - z.mean(0)) / z.std(0)
    #     return z

    def dgf(x):
        z = torch.sin(torch.pi * zfunc(x))
        z = (z - z.mean(0)) / z.std(0)
        return z

    x = torch.randn(n, D_x)  # Coordinates in R1
    z = dgf(x)  # ODE z(u)


class mlp(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512):
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
        self.Vz = mlp(D_z, D_z)
        self.dx = mlp(D_x, 1)


class model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.z = mlp(D_x + D_z + D_x * D_z, D_z)
        self.dz = mlp(D_x + D_z + D_x * D_z, D_z * D_x)

    def forward(self, dx, z, dz):
        x = torch.cat((dx, z, dz), -1)
        return self.z(x), self.dz(x)


znet = model2()

m = model()
optim_F = torch.optim.Adam(x_to_z.parameters(), lr=3e-4, weight_decay=1e-7)
optim_L = torch.optim.Adam(znet.parameters(), lr=3e-4, weight_decay=1e-7)
mse_mean_L = 0
mse_mean_F = 0


# def x_to_z_grad(x):
#     return torch.autograd.functional.jacobian(x_to_z, x)

x_to_z_grad = torch.vmap(torch.func.jacrev(x_to_z))

for i in range(n_training_iters):
    # Process forward:
    idx_F = torch.randperm(len(x))[:batch_size]
    xt = x[idx_F]
    zt = x_to_z(xt)
    dzt = x_to_z_grad(xt).reshape(batch_size, -1)

    # Process lateral using forward net to get jacobian
    n_lateral_sample = torch.randint(min_lateral_sample, max_lateral_sample + 1, ()).item()
    idx_L = torch.randperm(len(x))[:n_lateral_sample]
    x_sample = x[idx_L]

    # Get just closest sample:
    dx = xt.unsqueeze(1) - x_sample.unsqueeze(0)
    closest = torch.norm(dx, dim=-1).argmin(-1)
    x_sample = x_sample[closest]
    z_sample = x_to_z(x_sample)
    dz_sample = x_to_z_grad(x_sample).reshape(batch_size, -1)
    dx_sample = torch.take_along_dim(dx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze()
    zz, dz = znet(dx_sample, z_sample, dz_sample)
    # Loss
    # breakpoint()
    mse_F = torch.nn.functional.mse_loss(zt.squeeze(), z[idx_F].squeeze())
    mse_L_z = torch.nn.functional.mse_loss(zz.squeeze(), z[idx_F].squeeze())
    mse_L_dz = torch.nn.functional.mse_loss(dz, dzt.detach())
    mse_L = mse_L_z + mse_L_dz
    mse_F.backward()
    mse_L.backward()
    optim_F.step()
    optim_L.step()
    optim_F.zero_grad()
    optim_L.zero_grad()

    mse_mean_F += mse_F.item() / 100
    mse_mean_L += mse_L_z.item() / 100
    if i % 100 == 0:
        print(i, mse_mean_F, mse_mean_L)
        mse_mean_F = 0
        mse_mean_L = 0


print("Testing Zero-Shot: Using large sample of in-distribution data")
# The data has an ODE structure:
n = 1000
xtest = torch.randn(n, D_x) + 2  # Coordinates in R1
ztest = dgf(xtest)  # ODE z(u)
n_lateral_sample = max_lateral_sample

batch_size = 2
for n_steps in range(1, 10):
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
            dzt = x_to_z_grad(xt).reshape(batch_size, -1)
            mse_forward += torch.nn.functional.mse_loss(zt.squeeze(), ztest[idx_F].squeeze())

            # Process lateral using forward net to get jacobian
            n_lateral_sample = torch.randint(min_lateral_sample, max_lateral_sample + 1, ()).item()
            idx_L = torch.randperm(len(x))[:n_lateral_sample]
            x_sample = x[idx_L]

            # Get just closest sample:
            dx = xt.unsqueeze(1) - x_sample.unsqueeze(0)
            closest = torch.norm(dx, dim=-1).argmin(-1)
            x_sample = x_sample[closest]
            z_sample = x_to_z(x_sample)
            dx_sample = torch.take_along_dim(dx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze() / n_steps
            dz_sample = x_to_z_grad(x_sample).reshape(batch_size, -1)
            for step in range(n_steps):
                z_sample, dz_sample = znet(dx_sample, z_sample, dz_sample)
            zz = z_sample

            mse_lateral += torch.nn.functional.mse_loss(zz.squeeze(), ztest[idx_F].squeeze())
            mse_null += torch.nn.functional.mse_loss(
                x_to_z(xtest[(idx_F + 1) % len(xtest)]).squeeze(), ztest[idx_F].squeeze()
            )

        print(
            i,
            n_steps,
            f"Forward:{mse_forward.item() / len(xtest)}, \tLateral:{mse_lateral.item() / len(xtest)}, \tNull:{mse_null / len(xtest)}",
        )
