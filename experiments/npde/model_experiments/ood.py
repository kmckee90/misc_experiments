import torch
import torch.nn as nn

with torch.no_grad():
    D_x = 4
    D_z = 8
    n_lateral_sample = 10
    lateral_batch_size = 1

    xfunc = nn.Linear(D_z, D_x).requires_grad_(False)
    zfunc = nn.Linear(D_x, D_z).requires_grad_(False)

    # The data has an ODE structure:
    n = 5000
    x = torch.rand(n, D_x)  # Coordinates in R1
    z = torch.sin(torch.pi * zfunc(x))  # ODE z(u)
    # x = x + 0.01 * torch.randn_like(x)


class mlp(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)


# Now, given only X, can we model the state space of z?
class model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.x_to_u = mlp(D_x, 1)
        self.x_to_z = mlp(D_x, D_z)
        self.z_to_x = mlp(D_z, D_x)
        self.Qz = mlp(D_z, D_z)
        self.Kz = mlp(D_z, D_z)
        self.Vz = mlp(D_z, D_z)
        self.Qu = mlp(D_x, D_z)
        self.Ku = mlp(D_x, D_z)
        self.Vu = mlp(D_x, D_z)
        self.zzfunc = nn.MultiheadAttention(D_z, 1)


m = model()
# Re-estimate z laterally:
# Take 'x' in question:
# Autoencode:

optim = torch.optim.Adam(m.parameters(), lr=3e-5)

for i in range(len(x)):
    # Process forward
    rand_idx_targ = torch.randperm(len(x))[:lateral_batch_size]
    xt = x[rand_idx_targ]
    zt = m.x_to_z(xt)
    x_est = m.z_to_x(zt)
    # ut = m.x_to_u(xt)
    mse_forward = torch.nn.functional.mse_loss(x_est, xt)

    # Process lateral
    rand_idx = torch.randperm(len(x))[:n_lateral_sample]
    x_sample = x[rand_idx]
    z_est_sample = m.x_to_z(x_sample)
    # u_est_sample = m.x_to_u(x_sample)
    zz, _ = m.zzfunc(
        m.Qu(x_sample),
        m.Ku(xt - x_sample),
        m.Vz(z_est_sample.detach()),
        need_weights=False,
    )
    zz = zz.sum(0)
    mse_lateral_z = torch.nn.functional.mse_loss(zz.squeeze(), zt.squeeze().detach())
    with torch.no_grad():
        mse_lateral_x = torch.nn.functional.mse_loss(m.z_to_x(zz).squeeze(), xt.squeeze())

    # Opt
    loss = mse_forward + mse_lateral_z
    optim.zero_grad()
    loss.backward()
    optim.step()
    if i % 100 == 0:
        print(i, mse_forward.item(), mse_lateral_x.item(), mse_lateral_z.item())


print("Testing")
# The data has an ODE structure:
n = 1000
xtest = torch.rand(n, D_x) + 0.5  # Coordinates in R1
z = torch.sin(torch.pi * zfunc(xtest))  # ODE z(u)
# xtest = xfunc(z)  # Observations x(u) which is actually x(z(u))
# xtest = xtest + 0.01 * torch.randn_like(xtest)


with torch.no_grad():
    # Inference?
    mse_forward = 0
    mse_lateral = 0
    for i in range(len(xtest)):
        # Process forward
        rand_idx_targ = torch.randperm(len(xtest))[:lateral_batch_size]
        xt = xtest[rand_idx_targ]
        zt = m.x_to_z(xt)
        # ut = m.x_to_u(xt)
        x_est = m.z_to_x(zt)
        mse_forward += torch.nn.functional.mse_loss(x_est.squeeze(), xt.squeeze())

        # Process lateral
        rand_idx = torch.randperm(len(x))[:n_lateral_sample]
        x_sample = x[rand_idx]
        z_est_sample = m.x_to_z(x_sample)
        # u_est_sample = m.x_to_u(x_sample)
        zz, _ = m.zzfunc(
            m.Qu(x_sample),
            m.Ku(xt - x_sample),
            m.Vz(z_est_sample.detach()),
            need_weights=False,
        )
        zz = zz.sum(0)
        x_est_lat = m.z_to_x(zz)
        mse_lateral += torch.nn.functional.mse_loss(x_est_lat.squeeze(), xt.squeeze())
    print(i, f"Forward:{mse_forward.item() / len(xtest)}, \tLateral:{mse_lateral.item() / len(xtest)}")
