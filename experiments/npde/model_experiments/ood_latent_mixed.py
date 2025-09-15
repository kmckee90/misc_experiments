import torch
import torch.nn as nn
from spherical_dist import sample_hypersphere

# This ones uses pytorch's built in jacobian function to supply the first derivatives to the dynamics model.
# It also selects only the closest sample from a larger sample, rather than averaging.

# X -> U
# Y -> V
# d2V(U)/d2U = f(V, dV)

torch.set_default_device("cuda:1")

D_x = 256
D_y = 256
D_u = 2
D_v = 2

batch_size = 4096
n = 15000
n_training_iters = 1000
min_lateral_sample = 1
max_lateral_sample = 1024

xfunc = nn.Linear(D_u, D_x)
yfunc = nn.Linear(D_v, D_y)


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
        self.v = mlp(D_u + D_v + D_u * D_v, D_v)
        self.dv = mlp(D_u + D_v + D_u * D_v, D_v * D_u)
        self.u_to_v = mlp(D_u, D_v)
        self.u_to_dv = torch.vmap(torch.func.jacrev(self.u_to_v))

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
        self.U = autoencoder(D_x, 256, D_u)
        self.V = autoencoder(D_y, 256, D_v)

    def forward(self, x, y):
        Ux, Xu = self.U(x)
        Vy, Yv = self.V(y)
        return Ux, Vy, Xu, Yv


vnet = lateral_model()
xynet = preprocessor()


pars = [
    {"params": xynet.parameters(), "lr": 3e-4, "weight_decay": 1e-4},
    {"params": vnet.parameters(), "lr": 3e-4},
]

optim = torch.optim.Adam(pars)


print("Training PDE")
mse_mean = 0
for i in range(n_training_iters):
    # Get all main variables of interest:
    idx_F = torch.randperm(n)[:batch_size]
    x_train = x[idx_F]
    y_train = y[idx_F]
    Ux, Vy, Xu, Yv = xynet(x_train, y_train)

    # Prepare Lateral
    with torch.no_grad():
        n_lateral_sample = torch.randint(min_lateral_sample, max_lateral_sample + 1, ()).item()
        idx_L = torch.randperm(n)[:n_lateral_sample]
        Ux_sample, Xu_sample = xynet.U(x[idx_L])
        dUx = Ux.unsqueeze(1) - Ux_sample.unsqueeze(0)
        closest = torch.norm(dUx, dim=-1).argmin(-1)
        Ux_sample = Ux_sample[closest]
        dUx_sample = torch.take_along_dim(dUx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze()
        dVu_sample = vnet.u_to_dv(Ux_sample).reshape(batch_size, -1)

    # Process and train forward
    Vu = vnet.u_to_v(Ux.detach())

    # Using the forward here but not training it with this. (Could tho)
    Vu_sample = vnet.u_to_v(Ux_sample.detach())

    # Process lateral
    Vv, dVv = vnet(dUx_sample.detach(), Vu_sample.detach(), dVu_sample.detach())

    # Target for training vnet's dV output
    dVu = vnet.u_to_dv(Ux).reshape(batch_size, -1)

    mse = 0
    mse = mse + torch.nn.functional.mse_loss(Xu.squeeze(), x_train.squeeze())
    mse = mse + torch.nn.functional.mse_loss(Yv.squeeze(), y_train.squeeze())
    mse = mse + torch.nn.functional.mse_loss(Vu.squeeze(), Vy.squeeze().detach())
    mse = mse + torch.nn.functional.mse_loss(Vv.squeeze(), Vy.squeeze().detach())
    mse = mse + torch.nn.functional.mse_loss(dVv.squeeze(), dVu.squeeze().detach())

    # Lateral:
    optim.zero_grad()
    mse.backward()
    optim.step()

    mse_mean += mse.item() / 100
    if i % 100 == 0:
        print(i, mse_mean)
        mse_mean = 0


### Test
batch_size = 4096 * 3

print("TEST:")
with torch.no_grad():
    radii = range(3, 11)
    n_steps = range(1, 31)
    for rad in radii:
        best_step = -1
        best_score = torch.inf
        print("Radius:", rad)
        for steps in n_steps:
            u_test = sample_hypersphere((batch_size, D_u), radius=rad, thickness=0.1)
            x_test, y_test = dgf(u_test)  # ODE v(u)

            # Get all main variables of interest:
            Ux, Vy, Xu, Yv = xynet(x_test, y_test)

            # The total feed-forward path:
            ypred = xynet.V.decode(vnet.u_to_v(xynet.U.encode(x_test)))
            ynull = xynet.V.decode(vnet.u_to_v(xynet.U.encode(torch.randn_like(x_test) * 0.01)))

            mse_y = torch.nn.functional.mse_loss(ypred.squeeze(), y_test.squeeze())
            mse_null = torch.nn.functional.mse_loss(ynull.squeeze(), y_test.squeeze())

            # Lateral
            n_lateral_sample = torch.randint(min_lateral_sample, max_lateral_sample + 1, ()).item()
            idx_L = torch.randperm(n)[:n_lateral_sample]
            Ux_sample, Xu_sample = xynet.U(x[idx_L])
            dUx = Ux.unsqueeze(1) - Ux_sample.unsqueeze(0)
            closest = torch.norm(dUx, dim=-1).argmin(-1)
            Ux_sample = Ux_sample[closest]
            dVu_sample = vnet.u_to_dv(Ux_sample).reshape(batch_size, -1)
            dUx_sample = torch.take_along_dim(dUx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze() / steps

            Vu = vnet.u_to_v(Ux.detach())
            Vu_sample = vnet.u_to_v(Ux_sample.detach())
            dVu = vnet.u_to_dv(Ux).reshape(batch_size, -1)

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
            "Steps:",
            best_step,
            "Best:",
            best_score.item(),
            "Fwd:",
            mse_y.item(),
            "Null:",
            mse_null.item(),
        )
