import torch
import torch.nn as nn
from spherical_dist import sample_hypersphere

# This ones uses pytorch's built in jacobian function to supply the first derivatives to the dynamics model.
# It also selects only the closest sample from a larger sample, rather than averaging.

# X -> U
# Y -> V
# d2V(U)/d2U = f(V, dV)

torch.set_default_device("cuda:2")

D_x = 32
D_y = 32
D_u = 8
D_v = 8

batch_size = 4096
n = 15000
n_training_iters = 1000
min_lateral_sample = 10
max_lateral_sample = 128

xfunc = nn.Linear(D_u, D_x)
yfunc = nn.Linear(D_v, D_y)
vfunc = nn.Identity(D_u, D_v)

# Model dimensions
mD_u = 16
mD_v = 16


def dgf(u):
    v = torch.sin(torch.pi * u / 2)
    x = xfunc(u)
    y = yfunc(v)
    return x, y


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


class mlp_small(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
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
        self.v = mlp(mD_u + mD_v + mD_u * mD_v, mD_v)
        self.dv = mlp(mD_u + mD_v + mD_u * mD_v, mD_v * mD_u)

    def forward(self, du, v, dv):
        u = torch.cat((du, v, dv), -1)
        return self.v(u), self.dv(u)


class preprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_to_u = mlp_small(D_x, mD_u)
        self.u_to_v = mlp(mD_u, mD_v)
        self.v_to_y = mlp_small(mD_v, D_y)
        self.u_to_dv = torch.vmap(torch.func.jacrev(self.u_to_v))

    def forward(self, x):
        u = self.x_to_u(x)
        v = self.u_to_v(u)
        y = self.v_to_y(v)
        return u, v, y


vnet = lateral_model()
xynet = preprocessor()


pars = [
    {"params": xynet.parameters(), "lr": 1e-4, "weight_decay": 1e-5},
    {"params": vnet.parameters(), "lr": 1e-4, "weight_decay": 1e-5},
]
optim = torch.optim.Adam(pars)
while True:
    print("Training PDE")
    mse_mean_xy = 0
    mse_mean_uv = 0
    for i in range(n_training_iters):
        # Get all main variables of interest:
        idx_F = torch.randperm(n)[:batch_size]
        x_train = x[idx_F]
        y_train = y[idx_F]
        Ux, Vu, Yv = xynet(x_train)

        # Prepare Lateral
        with torch.no_grad():
            n_lateral_sample = torch.randint(min_lateral_sample, max_lateral_sample + 1, ()).item()
            idx_L = torch.randperm(n)[:n_lateral_sample]

        Ux_sample = xynet.x_to_u(x[idx_L])

        with torch.no_grad():
            dUx = Ux.unsqueeze(1) - Ux_sample.unsqueeze(0)
            closest = torch.norm(dUx, dim=-1).argmin(-1)
            Ux_sample = Ux_sample[closest]
            dUx_sample = torch.take_along_dim(dUx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze()
            dVu_sample = xynet.u_to_dv(Ux_sample).reshape(batch_size, -1)
            dVu = xynet.u_to_dv(Ux).reshape(batch_size, -1)

        # Process lateral
        Vu_sample = xynet.u_to_v(Ux_sample)
        Vv, dVv = vnet(dUx_sample.detach(), Vu_sample, dVu_sample.detach())

        # Target for training vnet's dV output

        # Autoencoders:
        mse_uv_v = torch.nn.functional.mse_loss(Vv.squeeze(), Vu.squeeze().detach())
        mse_uv_dv = torch.nn.functional.mse_loss(dVv.squeeze(), dVu.squeeze().detach())
        mse_xy = torch.nn.functional.mse_loss(Yv.squeeze(), y_train.squeeze())

        mse = 0.05 * (mse_uv_v + mse_uv_dv) + mse_xy
        # Lateral:
        optim.zero_grad()
        mse.backward()
        optim.step()

        mse_mean_uv += mse_uv_v.item() / 100
        mse_mean_xy += mse_xy.item() / 100
        if i % 100 == 0:
            print(i, mse_mean_xy, mse_mean_uv)
            mse_mean_uv = 0
            mse_mean_xy = 0

    ### Test
    batch_size = 4096

    print("TEST:")
    with torch.no_grad():
        radii = range(4, 16)
        n_steps = range(10, 11)
        for rad in radii:
            best_step = -1
            best_score = torch.inf
            # print("Radius:", rad)
            for steps in n_steps:
                u_test = sample_hypersphere((batch_size, D_u), radius=rad, thickness=0.1)
                x_test, y_test = dgf(u_test)  # ODE v(u)

                # Get all main variables of interest:
                Ux, Vu, Yv = xynet(x_test)
                _, _, ynull = xynet(torch.randn_like(x_test) * 0.1)

                mse_y = torch.nn.functional.mse_loss(Yv.squeeze(), y_test.squeeze())

                # Lateral
                with torch.no_grad():
                    n_lateral_sample = torch.randint(min_lateral_sample, max_lateral_sample + 1, ()).item()
                    idx_L = torch.randperm(n)[:n_lateral_sample]
                    Ux_sample = xynet.x_to_u(x[idx_L])
                    dUx = Ux.unsqueeze(1) - Ux_sample.unsqueeze(0)
                    # closest = torch.norm(dUx, dim=-1).argmin(-1)

                    norms = torch.norm(dUx, dim=-1)
                    closest = torch.topk(norms, 10, -1).indices[..., 0]

                    Ux_sample = Ux_sample[closest]
                    dUx_sample = torch.take_along_dim(dUx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze() / steps
                    Vu_sample = xynet.u_to_v(Ux_sample.detach())
                    dVu_sample = xynet.u_to_dv(Ux_sample).reshape(batch_size, -1)

                for _ in range(steps):
                    Vu_sample, dVu_sample = vnet(dUx_sample.detach(), Vu_sample.detach(), dVu_sample)
                Yvu = xynet.v_to_y(Vu_sample.detach())

                # Lateral:
                mse_uv = torch.nn.functional.mse_loss(Yvu.squeeze(), y_test.squeeze())
                mse_null = torch.nn.functional.mse_loss(ynull.squeeze(), y_test.squeeze())

                best_step = steps if mse_uv < best_score else best_step
                best_score = mse_uv if mse_uv < best_score else best_score

                # print(f"{steps} Steps:", "\tFw:", mse_y.item(), "\tLat:", mse_uv.item(), "\tNull:", mse_null.item())

            print(
                "Radius:",
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
