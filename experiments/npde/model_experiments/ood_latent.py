import torch
import torch.nn as nn
from spherical_dist import sample_hypersphere

# This ones uses pytorch's built in jacobian function to supply the first derivatives to the dynamics model.
# It also selects only the closest sample from a larger sample, rather than averaging.

# X -> U
# Y -> V
# d2V(U)/d2U = f(V, dV)

torch.set_default_device("cuda:2")

D_x = 7 * 7 * 3
D_y = 3
D_u = 8
D_v = 3

batch_size = 4096
n = 15000
n_training_iters = 1000
n_training_iters_autoencoder = 10000

min_lateral_sample = 10
max_lateral_sample = 256

xfunc = nn.Linear(D_u, D_x)
yfunc = nn.Linear(D_v, D_y)
vfunc = nn.Linear(D_u, D_v)

vfunc.weight.data = nn.Parameter(2 * (torch.rand_like(vfunc.weight.data) * 2 - 1))

# Model dimensions
mD_u = 16
mD_v = 16


def dgf(u):
    v = torch.sin(torch.pi * vfunc(u))
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
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
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
        self.v = mlp(mD_u + mD_v + mD_u * mD_v, mD_v)
        self.dv = mlp(mD_u + mD_v + mD_u * mD_v, mD_v * mD_u)

    def forward(self, du, v, dv):
        u = torch.cat((du, v, dv), -1)
        return self.v(u), self.dv(u)


class autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.encode = mlp_small(input_size, latent_size, hidden_size)
        self.decode = mlp_small(latent_size, input_size, hidden_size)

    def forward(self, x):
        encoding = self.encode(x)
        reconstruction = self.decode(encoding)
        return encoding, reconstruction


class preprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.U = autoencoder(D_x, 256, mD_u)
        self.V = autoencoder(D_y, 256, mD_v)
        self.u_to_v = mlp(mD_u, mD_v)
        self.u_to_dv = torch.vmap(torch.func.jacrev(self.u_to_v))

    def forward(self, x, y):
        Ux, Xu = self.U(x)
        Vy, Yv = self.V(y)
        return Ux, Vy, Xu, Yv


vnet = lateral_model()
xynet = preprocessor()


optim_xy = torch.optim.Adam(xynet.parameters(), lr=3e-4, weight_decay=1e-6)
optim_uv = torch.optim.Adam(vnet.parameters(), lr=3e-4, weight_decay=1e-6)

mse_mean_xy = 0
mse_mean_uv = 0

print("Pretraining AEs")
for i in range(n_training_iters_autoencoder):
    # Get all main variables of interest:
    idx_F = torch.randperm(n)[:batch_size]
    x_train = x[idx_F]
    y_train = y[idx_F]

    # Only use the reconstructions
    Ux, Vy, Xu, Yv = xynet(x_train, y_train)

    Vu = xynet.u_to_v(Ux.detach())

    # Autoencoders:
    mse_uv = torch.nn.functional.mse_loss(Vu.squeeze(), Vy.squeeze().detach())
    mse_xy = torch.nn.functional.mse_loss(Xu.squeeze(), x_train.squeeze())
    mse_xy = mse_xy + torch.nn.functional.mse_loss(Yv.squeeze(), y_train.squeeze())
    mse = mse_xy + mse_uv

    optim_xy.zero_grad()
    mse.backward()
    optim_xy.step()

    mse_mean_xy += mse_xy.item() / 100
    mse_mean_uv += mse_uv.item() / 100
    if i % 100 == 0:
        print(i, mse_mean_xy, mse_mean_uv)
        mse_mean_xy = 0
        mse_mean_uv = 0

while True:
    print("Training PDE")
    mse_mean_uv = 0
    for i in range(n_training_iters):
        # Get all main variables of interest:
        idx_F = torch.randperm(n)[:batch_size]
        x_train = x[idx_F]
        y_train = y[idx_F]
        Ux, Vy, _, _ = xynet(x_train, y_train)

        # Prepare Lateral
        with torch.no_grad():
            n_lateral_sample = torch.randint(min_lateral_sample, max_lateral_sample + 1, ()).item()
            idx_L = torch.randperm(n)[:n_lateral_sample]
            Ux_sample, Xu_sample = xynet.U(x[idx_L])
            dUx = Ux.unsqueeze(1) - Ux_sample.unsqueeze(0)

            closest = torch.norm(dUx, dim=-1).argmin(-1)

            # norms = torch.norm(dUx, dim=-1)
            # closest = torch.topk(norms, 4, -1).indices[..., 0]

            Ux_sample = Ux_sample[closest]
            dUx_sample = torch.take_along_dim(dUx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze()

            # Using the forward here but not training it with this. (Could tho)
            Vu_sample = xynet.u_to_v(Ux_sample.detach())
            dVu_sample = xynet.u_to_dv(Ux_sample).reshape(batch_size, -1)

        # Process and train forward
        # Vu = xynet.u_to_v(Ux.detach())
        # mse_uv = torch.nn.functional.mse_loss(Vu.squeeze(), Vy.squeeze().detach())

        # Process lateral
        Vv, dVv = vnet(dUx_sample.detach(), Vu_sample.detach(), dVu_sample.detach())
        mse_uv = torch.nn.functional.mse_loss(Vv.squeeze(), Vy.squeeze().detach())

        # Target for training vnet's dV output
        dVu = xynet.u_to_dv(Ux).reshape(batch_size, -1)
        mse_uv = mse_uv + torch.nn.functional.mse_loss(dVv.squeeze(), dVu.squeeze().detach())

        # Lateral:
        optim_uv.zero_grad()
        mse_uv.backward()
        optim_uv.step()

        mse_mean_uv += mse_uv.item() / 100
        if i % 100 == 0:
            print(i, mse_mean_uv)
            mse_mean_uv = 0

    ### Test
    batch_size = 4096

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

                # Get all main variables of interest:
                Ux, Vy, Xu, Yv = xynet(x_test, y_test)

                # The total feed-forward path:
                ypred = xynet.V.decode(xynet.u_to_v(xynet.U.encode(x_test)))
                ynull = xynet.V.decode(xynet.u_to_v(xynet.U.encode(torch.randn_like(x_test) * 0.01)))

                mse_y = torch.nn.functional.mse_loss(ypred.squeeze(), y_test.squeeze())
                mse_null = torch.nn.functional.mse_loss(ynull.squeeze(), y_test.squeeze())

                # Lateral
                n_lateral_sample = torch.randint(min_lateral_sample, max_lateral_sample + 1, ()).item()
                idx_L = torch.randperm(n)[:n_lateral_sample]
                Ux_sample, Xu_sample = xynet.U(x[idx_L])
                dUx = Ux.unsqueeze(1) - Ux_sample.unsqueeze(0)

                # closest = torch.norm(dUx, dim=-1).argmin(-1)
                norms = torch.norm(dUx, dim=-1)
                closest = torch.topk(norms, 10, -1).indices[..., 0]

                Ux_sample = Ux_sample[closest]
                dVu_sample = xynet.u_to_dv(Ux_sample).reshape(batch_size, -1)
                dUx_sample = torch.take_along_dim(dUx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze() / steps

                Vu = xynet.u_to_v(Ux.detach())
                Vu_sample = xynet.u_to_v(Ux_sample.detach())
                dVu = xynet.u_to_dv(Ux).reshape(batch_size, -1)

                for _ in range(steps):
                    Vu_sample, dVu_sample = vnet(dUx_sample.detach(), Vu_sample.detach(), dVu_sample.detach())
                Yvu = xynet.V.decode(Vu_sample)

                # Lateral:
                mse_uv = torch.nn.functional.mse_loss(Yvu.squeeze(), y_test.squeeze().detach())
                mse_null = torch.nn.functional.mse_loss(ynull.squeeze(), y_test.squeeze())

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
