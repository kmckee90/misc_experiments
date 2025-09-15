import torch
import torch.nn as nn
from spherical_dist import sample_hypersphere

# This one uses the forward function to create a train of equidistant points opposite the target.
# It can model very complex functions by conditioning on them.
# The issue: we don't really know what order the dynamical system is.
# By providing a large number, we don't need to know exactly.
# Using the jacobian limits us to second order nonlinear systems, however flexible that is.

torch.set_default_device("cuda:2")

with torch.no_grad():
    CATEGORICAL = False
    D_x = 2
    D_z = 2
    batch_size = 4096
    n = 15000
    n_training_iters = 1000
    min_lateral_sample = 1
    max_lateral_sample = 1024
    K = 8

    # zfunc = nn.Linear(D_x, D_z)
    zfunc = nn.Identity(D_x, D_z)

    # def dgf(x):
    #     d = x.shape[-1]
    #     z = torch.zeros(len(x))
    #     for i in range(d):
    #         z += torch.sin(torch.pi * x[:, i])
    #     return (z - z.mean()) / z.std()

    # # The data has an ODE structure over z:
    def dgf(x):
        z = torch.sin(torch.pi * x)
        z = torch.argmax(z, -1) if CATEGORICAL else z
        return z

    x = torch.randn(n, D_x)  # Coordinates in R1
    z = dgf(x)  # ODE z(u)


class mlp(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=1024):
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


def onehot(x, type="mode"):
    return (
        torch.distributions.OneHotCategorical(logits=x).mode
        if type == "mode"
        else torch.distributions.OneHotCategorical(logits=x).sample()
    )


# Now, given only X, can we model the state space of z?
x_to_z = mlp(D_x, D_z)


class model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.z = mlp(K * D_z + D_x, D_z)

    def forward(self, dx, zl):
        x = torch.cat((dx, zl), -1)
        z = self.z(x)
        new_zl = torch.cat((z, zl[..., : -z.shape[-1]]), -1)
        return z, new_zl


znet = model3()

optim_F = torch.optim.Adam(x_to_z.parameters(), lr=3e-4, weight_decay=1e-7)
optim_L = torch.optim.Adam(znet.parameters(), lr=3e-4, weight_decay=1e-7)
mse_mean_L = 0
mse_mean_F = 0
sum_L = 0
sum_F = 0
for i in range(n_training_iters):
    # Process forward:
    idx_F = torch.randperm(n)[:batch_size]
    xt = x[idx_F]
    zt = x_to_z(xt)

    # Process lateral using forward net to get jacobian
    n_lateral_sample = torch.randint(min_lateral_sample, max_lateral_sample + 1, ()).item()
    idx_L = torch.randperm(n)[:n_lateral_sample]
    x_sample = x[idx_L]

    # Get just closest sample:
    dx = xt.unsqueeze(1) - x_sample.unsqueeze(0)
    closest = torch.norm(dx, dim=-1).argmin(-1)
    x_sample = x_sample[closest]
    dx = torch.take_along_dim(dx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze()
    zl = [x_to_z(x_sample - k * dx) for k in range(K)]
    zl = torch.cat(zl, -1)
    zz, zl = znet(dx, zl)

    # Loss
    if CATEGORICAL:
        mse_F = torch.nn.functional.cross_entropy(zt.squeeze(), z[idx_F].squeeze())
        mse_L = torch.nn.functional.cross_entropy(zz.squeeze(), z[idx_F].squeeze())
        # mse_F = torch.nn.functional.mse_loss(zt.squeeze(), torch.nn.functional.one_hot(z[idx_F].squeeze()).float())
        # mse_L = torch.nn.functional.mse_loss(zz.squeeze(), torch.nn.functional.one_hot(z[idx_F].squeeze()).float())
        sum_F += (zt.argmax(-1).squeeze() == z[idx_F].squeeze()).sum().item() / (100 * batch_size)
        sum_L += (zz.argmax(-1).squeeze() == z[idx_F].squeeze()).sum().item() / (100 * batch_size)
    else:
        mse_F = torch.nn.functional.mse_loss(zt.squeeze(), z[idx_F].squeeze())
        mse_L = torch.nn.functional.mse_loss(zz.squeeze(), z[idx_F].squeeze())
        mse_mean_F += mse_F.item() / 100
        mse_mean_L += mse_L.item() / 100

    mse_F.backward()
    mse_L.backward()
    optim_F.step()
    optim_L.step()
    optim_F.zero_grad()
    optim_L.zero_grad()

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

for rad in (2, 3, 4, 5, 6, 7):
    print("RAD=", rad)
    xtest = sample_hypersphere((batch_size, D_x), radius=rad, thickness=0.1)
    ztest = dgf(xtest)  # ODE z(u)
    for n_steps in range(1, 16):
        with torch.no_grad():
            # Process forward:
            xt = xtest
            zt = x_to_z(xt)

            # Get lateral sample
            n_lateral_sample = torch.randint(min_lateral_sample, max_lateral_sample + 1, ()).item()
            idx_L = torch.randperm(n)[:n_lateral_sample]
            x_sample = x[idx_L]

            # Get just closest sample:
            dx = xt.unsqueeze(1) - x_sample.unsqueeze(0)
            closest = torch.norm(dx, dim=-1).argmin(-1)
            x_sample = x_sample[closest]
            dx = torch.take_along_dim(dx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze() / n_steps
            zl = [x_to_z(x_sample - k * dx) for k in range(K)]
            zl = torch.cat(zl, -1)
            for recstep in range(n_steps):
                zz, zl = znet(dx, zl)

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
                mse_null = torch.nn.functional.mse_loss(x_to_z(torch.randn_like(xtest) * 0.5), ztest.squeeze())
                print(
                    i,
                    n_steps,
                    f"Forward:{mse_forward.item()}, \tLateral:{mse_lateral.item()}, \tNull:{mse_null}",
                )
