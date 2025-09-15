import torch
import torch.nn as nn
from spherical_dist import sample_hypersphere

# This ones uses pytorch's built in jacobian function to supply the first derivatives to the dynamics model.
# It also selects only the closest sample from a larger sample, rather than averaging.

# X -> U
# Y -> V
# d2V(U)/d2U = f(V, dV)

torch.set_default_device("cuda:5")

D_u = 2
D_v = 2

batch_size = 4096
n = 15000
n_training_iters = 1000
n_forward_training_iters = 10000
min_lateral_sample = 10
max_lateral_sample = 256
vfunc = nn.Linear(D_u, D_v)
vfunc.weight.data = nn.Parameter(2 * (torch.rand_like(vfunc.weight.data) * 2 - 1))
# vfunc.weight.data = nn.Parameter(2 * torch.randn_like(vfunc.weight.data))

# vfunc = nn.Identity(D_u, D_v)


def dgf(u):
    v = torch.sin(torch.pi * vfunc(u))
    return u, v


with torch.no_grad():
    u = torch.randn(n, D_u)
    x, y = dgf(u)

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


class lateral_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = mlp(D_u + D_v + D_u * D_v, D_v)
        self.dv = mlp(D_u + D_v + D_u * D_v, D_v * D_u)

    def forward(self, du, v, dv):
        u = torch.cat((du, v, dv), -1)
        return self.v(u), self.dv(u)


u_to_v = mlp(D_u, D_v, 1024)
u_to_dv = torch.vmap(torch.func.jacrev(u_to_v))

vnet = lateral_model()
optim_xy = torch.optim.Adam(u_to_v.parameters(), lr=3e-4, weight_decay=1e-6)

mse_mean_xy = 0
mse_mean_uv = 0
print("Training forward net")
for i in range(n_forward_training_iters):
    # Get all main variables of interest:
    idx_F = torch.randperm(n)[:batch_size]
    x_train = x[idx_F]
    y_train = y[idx_F]
    ypred = u_to_v(x_train)
    mse = torch.nn.functional.mse_loss(ypred, y_train.detach())
    # Lateral:
    optim_xy.zero_grad()
    mse.backward()
    optim_xy.step()
    mse_mean_uv += mse.item() / 100
    if i % 100 == 0:
        print(i, mse_mean_uv)
        mse_mean_uv = 0

optim_uv = torch.optim.Adam(vnet.parameters(), lr=3e-4, weight_decay=1e-6)

while True:
    print("Training PDE")
    mse_mean_uv = 0
    for i in range(n_training_iters):
        # Get all main variables of interest:
        idx_F = torch.randperm(n)[:batch_size]
        x_train = x[idx_F]
        y_train = y[idx_F]

        # Prepare Lateral
        with torch.no_grad():
            n_lateral_sample = torch.randint(min_lateral_sample, max_lateral_sample + 1, ()).item()
            idx_L = torch.randperm(n)[:n_lateral_sample]
            x_sample = x[idx_L]
            dx = x_train.unsqueeze(1) - x_sample.unsqueeze(0)
            closest = torch.norm(dx, dim=-1).argmin(-1)
            x_sample = x_sample[closest]
            dx_sample = torch.take_along_dim(dx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze()

        # Using the forward here but not training it with this. (Could tho)
        y_sample = u_to_v(x_sample)
        dy_sample = u_to_dv(x_sample).reshape(batch_size, -1)

        # Process lateral
        yy, dypred = vnet(dx_sample.detach(), y_sample.detach(), dy_sample.detach())
        mse_uv = torch.nn.functional.mse_loss(yy.squeeze(), y_train.squeeze())

        # Target for training vnet's dV output
        dy = u_to_dv(x_train).reshape(batch_size, -1)
        mse_uv = mse_uv + torch.nn.functional.mse_loss(dypred.squeeze(), dy.squeeze().detach())

        # Lateral:
        mse = mse_uv
        optim_uv.zero_grad()
        mse_uv.backward()
        optim_uv.step()

        mse_mean_uv += mse_uv.item() / 100
        if i % 100 == 0:
            print(i, mse_mean_uv)
            mse_mean_uv = 0

    ### Test
    print("TEST:")
    with torch.no_grad():
        radii = range(0, 16)
        n_steps = range(10, 11)
        for rad in radii:
            best_step = -1
            best_score = torch.inf
            for steps in n_steps:
                u_test = sample_hypersphere((batch_size, D_u), radius=rad, thickness=0.1)
                x_test, y_test = dgf(u_test)  # ODE v(u)

                # Prepare Lateral
                n_lateral_sample = torch.randint(min_lateral_sample, max_lateral_sample + 1, ()).item()
                idx_L = torch.randperm(n)[:n_lateral_sample]
                x_sample = x[idx_L]
                dx = x_test.unsqueeze(1) - x_sample.unsqueeze(0)
                closest = torch.norm(dx, dim=-1).argmin(-1)
                # norms = torch.norm(dx, dim=-1)
                # closest = torch.topk(norms, 10, -1).indices[..., 0]
                x_sample = x_sample[closest]
                dx_sample = torch.take_along_dim(dx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze() / steps

                # lateral inference
                y_sample = u_to_v(x_sample)
                dy_sample = u_to_dv(x_sample).reshape(batch_size, -1)
                for _ in range(steps):
                    y_sample, dy_sample = vnet(dx_sample.detach(), y_sample.detach(), dy_sample.detach())

                # Lateral:
                mse_y = torch.nn.functional.mse_loss(u_to_v(x_test), y_test.squeeze())
                mse_null = torch.nn.functional.mse_loss(u_to_v(torch.randn_like(x_test)), y_test.squeeze())
                mse_uv = torch.nn.functional.mse_loss(y_sample.squeeze(), y_test.squeeze().detach())

                best_step = steps if mse_uv < best_score else best_step
                best_score = mse_uv if mse_uv < best_score else best_score

                # print(f"{steps} Steps:", "\tFw:", mse_y.item(), "\tLat:", mse_uv.item(), "\tNull:", mse_null.item())
            print(
                "Radius",
                rad,
                "Steps:",
                best_step,
                "Best:",
                best_score.item(),
                "Fwd:",
                mse_y.item(),
                "Null:",
                mse_null.item(),
            )
