import torch
import torch.nn as nn
from spherical_dist import sample_hypersphere

# This ones uses pytorch's built in jacobian function to supply the first derivatives to the dynamics model.
# It also selects only the closest sample from a larger sample, rather than averaging.

torch.set_default_device("cuda:1")

with torch.no_grad():
    CATEGORICAL = False
    D_x = 5
    D_z = 2
    batch_size = 4096
    n = 15000
    n_training_iters = 10000
    min_lateral_sample = 1
    max_lateral_sample = 1024

    zfunc = nn.Linear(D_x, D_z)
    # yfunc = nn.Linear(D_z, D_z)

    # zfunc = nn.Identity(D_x, D_z)
    yfunc = nn.Identity(D_z, D_z)

    # def dgf(x):
    #     d = x.shape[-1]
    #     z = torch.zeros(len(x))
    #     for i in range(d):
    #         z += torch.sin(torch.pi * x[:, i])
    #     return (z - z.mean()) / z.std()

    # # The data has an ODE structure over z:
    def dgf(x):
        z = zfunc(x)
        z = torch.sin((1 + torch.rand(D_z)) * torch.pi * z)
        y = yfunc(z)
        y = torch.argmax(y, -1) if CATEGORICAL else y
        return y

    x = torch.randn(n, D_x)  # Coordinates in R1
    z = dgf(x)  # ODE z(u)


# import torch.nn.init as init

afunc = nn.SiLU


class mlp(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            afunc(),
            nn.Linear(hidden_size, hidden_size),
            afunc(),
            nn.Linear(hidden_size, hidden_size),
            afunc(),
            nn.Linear(hidden_size, hidden_size),
            afunc(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)


# Now, given only X, can we model the state space of z?
x_to_z = mlp(D_x, D_z)


class model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.z = mlp(D_x + D_z + D_x * D_z, D_z)
        self.dz = mlp(D_x + D_z + D_x * D_z, D_z * D_x)

    def forward(self, dx, z, dz):
        x = torch.cat((dx, z, dz), -1)
        return self.z(x), self.dz(x)


znet = model2()

optim_F = torch.optim.Adam(x_to_z.parameters(), lr=1e-3)
optim_L = torch.optim.Adam(znet.parameters(), lr=1e-3)
mse_mean_L = 0
mse_mean_F = 0
sum_L = 0
sum_F = 0


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
    if CATEGORICAL:
        mse_F = torch.nn.functional.cross_entropy(zt.squeeze(), z[idx_F].squeeze())
        mse_L_z = torch.nn.functional.cross_entropy(zz.squeeze(), z[idx_F].squeeze())
        # mse_F = torch.nn.functional.mse_loss(zt.squeeze(), torch.nn.functional.one_hot(z[idx_F].squeeze()).float())
        # mse_L_z = torch.nn.functional.mse_loss(zz.squeeze(), torch.nn.functional.one_hot(z[idx_F].squeeze()).float())
        sum_F += (zt.argmax(-1).squeeze() == z[idx_F].squeeze()).sum().item() / (100 * batch_size)
        sum_L += (zz.argmax(-1).squeeze() == z[idx_F].squeeze()).sum().item() / (100 * batch_size)
    else:
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
        print(i, mse_mean_F, mse_mean_L, "\tFwd:", sum_F, "\tLat:", sum_L)
        mse_mean_F = 0
        mse_mean_L = 0
        sum_F = 0
        sum_L = 0

print("Testing Zero-Shot: Using large sample of in-distribution data")
# The data has an ODE structure:
batch_size = 4096
n_lateral_sample = max_lateral_sample


for rad in (2, 3, 4, 5, 6, 7, 8, 9, 10):
    print("RAD=", rad)
    xtest = sample_hypersphere((batch_size, D_x), radius=rad, thickness=0.1)
    ztest = dgf(xtest)  # ODE z(u)
    for n_steps in range(1, 16):
        with torch.no_grad():
            # Process forward
            xt = xtest
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
            dx_sample = torch.take_along_dim(dx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze() / n_steps
            dz_sample = x_to_z_grad(x_sample).reshape(batch_size, -1)
            for step in range(n_steps):
                z_sample, dz_sample = znet(dx_sample, z_sample, dz_sample)
            zz = z_sample

            if CATEGORICAL:
                n_correct_fwd = (zt.argmax(-1).squeeze() == ztest.squeeze()).sum().item() / batch_size
                n_correct_lateral = (zz.argmax(-1).squeeze() == ztest.squeeze()).sum().item() / batch_size

                print(
                    i,
                    n_steps,
                    f"Forward:{n_correct_fwd}, \tLateral:{n_correct_lateral}",
                )
            else:
                mse_forward = torch.nn.functional.mse_loss(zt.squeeze(), ztest.squeeze())
                mse_lateral = torch.nn.functional.mse_loss(zz.squeeze(), ztest.squeeze())
                mse_null = torch.nn.functional.mse_loss(x_to_z(0 * xtest).squeeze(), ztest.squeeze())

                print(
                    n_steps,
                    f"Forward:{mse_forward.item()}, \tLateral:{mse_lateral.item()}, \tNull:{mse_null}",
                )
