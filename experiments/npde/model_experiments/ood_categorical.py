import torch
import torch.nn as nn

torch.set_default_device("cuda:1")

D_x = 3
D_z = 3
min_lateral_sample = 1024
max_lateral_sample = 1024
batch_size = 4096 * 4
n = 4096 * 100
n_training_iters = 1000
zfunc = nn.Identity(D_x, D_z)
lr = 1e-4


# # The data has an ODE structure over z:
def dgf(x):
    z = torch.sin(torch.pi * zfunc(x) / torch.tensor([2.0, 1.0, 0.5]))
    y = z.argmax(-1)
    return y


x = 0.5 * torch.randn(n, D_x)  # Coordinates in R1
z = dgf(x)  # ODE z(u)
[(z == i).sum().item() for i in range(D_z)]


class mlpc(nn.Module):
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
            nn.Softmax(),
        )

    def forward(self, x):
        return self.model(x)


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


# Now, given only X, can we model the state space of z?
x_to_z = mlp(D_x, D_z)
z_to_y = mlp(D_z, D_z)


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.z = mlp(D_x + D_z + D_x * D_z, D_z)
        self.dz = mlp(D_x + D_z + D_x * D_z, D_z * D_x)

    def forward(self, dx, z, dz):
        x = torch.cat((dx, z, dz), -1)
        return self.z(x), self.dz(x)


fwpars = [
    {"params": x_to_z.parameters()},
    {"params": z_to_y.parameters()},
]

znet = model()
optim_F = torch.optim.Adam(fwpars, lr=lr)
optim_L = torch.optim.Adam(znet.parameters(), lr=lr)
mse_mean_L = 0
mse_mean_F = 0

x_to_z_grad = torch.vmap(torch.func.jacrev(x_to_z))

for i in range(n_training_iters):
    # Process forward:
    idx_F = torch.randperm(len(x))[:batch_size]
    xt = x[idx_F]
    zt = x_to_z(xt)
    dzt = x_to_z_grad(xt).reshape(batch_size, -1)
    # yt = z_to_y(zt)
    yt = zt

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
    # yy = z_to_y(zz)
    yy = zz

    # Loss
    mse_F = torch.nn.functional.cross_entropy(yt.squeeze(), z[idx_F].squeeze())
    mse_L_z = torch.nn.functional.cross_entropy(yy.squeeze(), z[idx_F].squeeze())
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

        print(
            "Forward:",
            (yt.argmax(-1).squeeze() == z[idx_F]).sum().item() / batch_size,
            "\t",
            "Lateral",
            (yy.argmax(-1).squeeze() == z[idx_F]).sum().item() / batch_size,
        )


print("Testing Zero-Shot: Using large sample of in-distribution data")
# The data has an ODE structure:
batch_size = 4096 * 4

xtest = 10 * (torch.rand(n, D_x) * 2 - 1)  # + torch.tensor([5.0, -3, -1])  # Coordinates in R1
ztest = dgf(xtest)  # ODE z(u)
n_lateral_sample = max_lateral_sample

for n_steps in range(1, 13, 1):
    with torch.no_grad():
        # Inference?
        mse_forward = 0
        mse_lateral = 0
        mse_null = 0
        num_correct_forward = 0
        num_correct_lateral = 0
        for i in range(1):
            # Process forward
            idx_F = torch.randperm(len(xtest))[:batch_size]
            xt = xtest[idx_F]
            zt = x_to_z(xt)
            dzt = x_to_z_grad(xt).reshape(batch_size, -1)
            # yt = z_to_y(zt)
            yt = zt

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
            # yy = z_to_y(zz)
            yy = zz

            # mse_forward += torch.nn.functional.cross_entropy(yt.squeeze(), ztest[idx_F].squeeze())
            # mse_lateral += torch.nn.functional.cross_entropy(yy.squeeze(), ztest[idx_F].squeeze())
            # mse_null += torch.nn.functional.cross_entropy(
            #     z_to_y(x_to_z(xtest[(idx_F + 1) % len(xtest)])).squeeze(), ztest[idx_F].squeeze()
            # )
            num_correct_forward += (yt.argmax(-1).squeeze() == ztest[idx_F]).sum().item()
            num_correct_lateral += (yy.argmax(-1).squeeze() == ztest[idx_F]).sum().item()

        # print(
        #     i,
        #     n_steps,
        #     f"Forward:{mse_forward.item() / len(xtest)}, \tLateral:{mse_lateral.item() / len(xtest)}, \tNull:{mse_null / len(xtest)}",
        # )

        print(step, "Fwd:", num_correct_forward / (batch_size), "\t", "Lat", num_correct_lateral / (batch_size))
