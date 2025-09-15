import torch
import torch.nn as nn
from model_experiments.spherical_dist import sample_hypersphere

# This ones uses pytorch's built in jacobian function to supply the first derivatives to the dynamics model.
# It also selects only the closest sample from a larger sample, rather than averaging.

# X -> U
# Y -> V
# d2V(U)/d2U = f(V, dV)


class CircularBuffer:
    def __init__(self, size, capacity, dtype=torch.float):
        self.capacity = capacity
        self.buffer = torch.zeros(self.capacity, size, dtype=dtype)
        self.idx = 0
        self.wrapped = False

    def append(self, x):
        self.buffer[self.idx] = x
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.wrapped = True

    def last(self):
        return self.buffer[(self.idx - 1) % self.capacity]

    def __call__(self, *args, **kwds):
        if not self.wrapped:
            return self.buffer[: self.idx]
        else:
            return self.buffer


afunc = nn.SiLU


class mlp(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=2048):
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


class autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.encode = mlp_small(input_size, latent_size, hidden_size)
        self.decode = mlp_small(latent_size, input_size, hidden_size)

    def forward(self, x):
        encoding = self.encode(x)
        reconstruction = self.decode(encoding)
        return encoding, reconstruction


class lateral_model(nn.Module):
    def __init__(self, D_x, D_q):
        super().__init__()
        self.q = mlp(D_x + D_q + D_x * D_q, D_q)
        self.dq = mlp(D_x + D_q + D_x * D_q, D_q * D_x)

    def forward(self, dx, q, dq):
        x = torch.cat((dx, q, dq), -1)
        return self.q(x), self.dq(x)


class NPDE_Agent(nn.Module):
    def __init__(self, forward_func, config):
        super().__init__()
        self.q = forward_func
        self.dq = torch.vmap(torch.func.jacrev(self.q))
        self.D_x = self.q[0].in_features
        self.D_q = self.q[-1].out_features
        self.q_q = lateral_model(self.D_x, self.D_q)
        self.memory_length = config.memory_length
        self.batch_size = config.batch_size
        self.max_lateral_sample = config.max_lateral_sample
        self.min_lateral_sample = config.min_lateral_sample
        self.solving_steps = config.solving_steps
        self.n_training_iters = config.n_training_iters
        self.topk = config.topk

        self.mem_q = CircularBuffer(self.D_q, self.memory_length, dtype=torch.float32)
        self.mem_state = CircularBuffer(self.D_x, self.memory_length, dtype=torch.float32)

        self.optim = torch.optim.Adam(self.q_q.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def forward(self, x):
        with torch.no_grad():
            n_lateral_sample = torch.randint(self.min_lateral_sample, self.max_lateral_sample + 1, ()).item()
            idx_L = torch.randperm(self.memory_length)[:n_lateral_sample]
            x_sample = self.mem_state()[idx_L]

            dx = x.unsqueeze(1) - x_sample.unsqueeze(0)
            # closest = torch.norm(dx, dim=-1).argmin(-1)

            norms = torch.norm(dx, dim=-1)
            closest = torch.topk(norms, self.topk, -1).indices[..., 0]

            x_sample = x_sample[closest]
            dx_sample = (
                torch.take_along_dim(dx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze() / self.solving_steps
            )

            # lateral inference
            q_sample = self.q(x_sample)
            dq_sample = self.dq(x_sample).reshape(x_sample.shape[0], -1)
            for _ in range(self.solving_steps):
                q_sample, dq_sample = self.q_q(
                    dx_sample.detach().squeeze(), q_sample.detach().squeeze(), dq_sample.detach().squeeze()
                )
            return q_sample

    def collect(self, state, q):
        self.mem_state.append(state.detach())
        self.mem_q.append(q.detach())

    def clear_memory(self):
        self.mem_q = CircularBuffer(self.D_q, self.memory_length, dtype=torch.float32)
        self.mem_state = CircularBuffer(self.D_x, self.memory_length, dtype=torch.float32)

    def train(self):
        # mse_mean_uv = 0
        # Get all main variables of interest:
        # Prepare Lateral
        with torch.no_grad():
            idx_F = torch.randperm(self.memory_length)[: self.batch_size]
            x_train = self.mem_state()[idx_F]
            q_train = self.mem_q()[idx_F]

            n_lateral_sample = torch.randint(self.min_lateral_sample, self.max_lateral_sample + 1, ()).item()
            idx_L = torch.randperm(self.memory_length)[:n_lateral_sample]
            x_sample = self.mem_state()[idx_L]
            dx = x_train.unsqueeze(1) - x_sample.unsqueeze(0)
            closest = torch.norm(dx, dim=-1).argmin(-1)
            x_sample = x_sample[closest]
            dx_sample = torch.take_along_dim(dx, dim=1, indices=closest.reshape(-1, 1, 1)).squeeze()

        # Using the forward here but not training it with this. (Could tho)
        q_sample = self.q(x_sample)
        dq_sample = self.dq(x_sample).reshape(self.batch_size, -1)

        # Process lateral
        qq, dqpred = self.q_q(dx_sample.detach(), q_sample.detach(), dq_sample.detach())
        mse_uv = torch.nn.functional.mse_loss(qq.squeeze(), q_train.squeeze())

        # Target for training vnet's dV output
        dq = self.dq(x_train).reshape(self.batch_size, -1)
        mse_uv = mse_uv + torch.nn.functional.mse_loss(dqpred.squeeze(), dq.squeeze().detach())

        # Lateral:
        self.optim.zero_grad()
        mse_uv.backward()
        self.optim.step()

        # mse_mean_uv += mse_uv.item() / self.n_training_iters

        return mse_uv


if __name__ == "__main__":
    torch.set_default_device("cuda:3")

    class npde_hyperpars:
        n_training_iters: int = 1000
        batch_size: int = 1024
        memory_length: int = 4096 * 4
        lr: float = 1e-4
        weight_decay: float = 0
        min_lateral_sample: int = 10
        max_lateral_sample: int = 256
        solving_steps: int = 10
        topk: int = 10

    config = npde_hyperpars()

    D_u = 2
    D_v = 2

    n = 15000
    fw_training_iters = 3000
    vfunc = nn.Linear(D_u, D_v)
    vfunc.weight.data = nn.Parameter(2 * (torch.rand_like(vfunc.weight.data) * 2 - 1))
    vfunc = nn.Identity()

    # def dgf(u):
    #     v = torch.sin(torch.pi * vfunc(u) / 2)
    #     return u, v

    def dgf(u):
        v = torch.sin(torch.pi * vfunc(u)) + torch.sin(2 * torch.pi * vfunc(u)) + torch.sin(3 * torch.pi * vfunc(u))
        return u, v

    with torch.no_grad():
        u = torch.randn(n, D_u)
        x, y = dgf(u)

    # First need to get a forward function acting as "q"
    hidden_size = 1024
    qnet = nn.Sequential(
        nn.Linear(D_u, hidden_size),
        afunc(),
        nn.Linear(hidden_size, hidden_size),
        afunc(),
        nn.Linear(hidden_size, hidden_size),
        afunc(),
        nn.Linear(hidden_size, hidden_size),
        afunc(),
        nn.Linear(hidden_size, D_v),
    )

    optim_xy = torch.optim.Adam(qnet.parameters(), lr=3e-4, weight_decay=1e-6)

    mse_mean = 0
    print("Training forward net")
    for i in range(fw_training_iters):
        idx_F = torch.randperm(n)[:4096]
        x_train = x[idx_F]
        y_train = y[idx_F]
        ypred = qnet(x_train)
        mse = torch.nn.functional.mse_loss(ypred, y_train.detach())
        optim_xy.zero_grad()
        mse.backward()
        optim_xy.step()
        mse_mean += mse.item() / 100
        if i % 100 == 0:
            print(i, mse_mean)
            mse_mean = 0

    # instantiate the model
    model = NPDE_Agent(qnet, npde_hyperpars())

    # Train lateral
    for i in range(model.memory_length):
        # Get all main variables of interest:
        x_train = x[i % n]
        y_train = y[i % n]
        model.collect(x_train, y_train)

    while True:
        print("TRAIN")
        mean_mse = 0
        for i in range(config.n_training_iters):
            mse = model.train()
            mean_mse += mse.item() / 100
            if i >= 100 and i % 100 == 0:
                print(i, "MSE", mean_mse)
                mean_mse = 0

        # Test lateral
        print("TEST:")
        batch_size = 4096
        with torch.no_grad():
            radii = range(0, 16)
            n_steps = range(10, 11)
            for rad in radii:
                best_step = -1
                best_score = torch.inf
                for steps in n_steps:
                    u_test = sample_hypersphere((batch_size, D_u), radius=rad, thickness=0.1)
                    x_test, y_test = dgf(u_test)  # ODE v(u)
                    q_pred = model(x_test)

                    mse_y = torch.nn.functional.mse_loss(model.q(x_test), y_test.squeeze())
                    mse_null = torch.nn.functional.mse_loss(model.q(torch.randn_like(x_test)), y_test.squeeze())
                    mse_uv = torch.nn.functional.mse_loss(q_pred.squeeze(), y_test.squeeze().detach())

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
