import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import evolutionary.modules as M


def savefig(file):
    plt.savefig(file, bbox_inches="tight", pad_inches=0.1, dpi=300)


def set_spectral_radius(rnn, layer_size, radius=0.9):
    with torch.no_grad():
        W = rnn.weight_hh.cpu().detach()[-layer_size:]
        eigs = torch.abs(torch.real(torch.linalg.eigvals(W)))
        maxEig = eigs.max()
        rnn.weight_hh[-layer_size:] *= radius / maxEig


def sparsify_weights(rnn, connection_probability=0.1):
    with torch.no_grad():
        a1 = torch.zeros_like(rnn.weight_hh).bernoulli_(connection_probability)
        a2 = torch.zeros_like(rnn.weight_ih).bernoulli_(connection_probability)
        rnn.weight_hh *= a1
        rnn.weight_ih *= a2


def random_band_matrix(n, m, k):
    with torch.no_grad():
        matrix = torch.zeros(n, m)
        for i in range(-k, k + 1):
            diag_size = min(n, m, n - abs(i), m - abs(i))
            random_vector = 2 * torch.rand(diag_size) - 1
            diag_matrix = torch.diag(random_vector, diagonal=i)
            matrix += diag_matrix
    return matrix


def createRollingMask(input_size, num_unique, num_shared):
    with torch.no_grad():
        wmask = torch.tensor(torch.kron(torch.eye(input_size), torch.ones((num_unique + num_shared, 1))))
        wmask_ex = wmask.clone().detach()
        for i in range(1, num_shared + 1):
            wmask_ex += wmask.roll(i, 0)
        wmask_ex = (wmask_ex > 0).int().float()
    return wmask_ex


class Reservoir(nn.Module):
    def __init__(self, input_size, hidden_size, spectral_radius=0.9, connection_probability=0.1):
        super().__init__()
        self.rnn = nn.RNNCell(input_size, hidden_size).requires_grad_(False)
        self.h = torch.zeros(hidden_size)
        self.hidden_size = hidden_size
        torch.nn.init.xavier_uniform_(self.rnn.weight_hh, gain=1)
        torch.nn.init.xavier_normal_(self.rnn.weight_ih, gain=1)
        sparsify_weights(self.rnn, connection_probability=connection_probability)
        set_spectral_radius(self.rnn, hidden_size, radius=spectral_radius)

    def forward(self, x):
        self.h = self.rnn(x, self.h)
        return self.h


class Reservoir_Local(nn.Module):
    def __init__(
        self,
        input_size,
        num_unique=10,
        num_shared=5,
        spectral_radius=1,
        reach=10,
        input_conn_prob=0.5,
        local_conn_prob=0.5,
        local_weight_scale=1,
        global_conn_prob=0.01,
        global_weight_scale=1,
    ):
        super().__init__()

        wmask = createRollingMask(input_size, num_unique, num_shared)
        self.hidden_size = wmask.shape[0]

        self.rnn = nn.RNNCell(input_size, self.hidden_size).requires_grad_(False)
        self.h = torch.zeros(self.hidden_size)
        self.rnn.weight_hh = nn.Parameter(
            random_band_matrix(self.hidden_size, self.hidden_size, reach) * local_weight_scale
        ).requires_grad_(False)
        self.rnn.weight_hh *= torch.bernoulli(self.rnn.weight_hh, local_conn_prob)

        mask_local = (self.rnn.weight_hh == 0).float()

        newCons_hh = (
            (2 * torch.rand_like(self.rnn.weight_hh) - 1)
            * torch.bernoulli(self.rnn.weight_hh, global_conn_prob)
            * global_weight_scale
            * mask_local
        )
        self.rnn.weight_hh += newCons_hh

        torch.nn.init.xavier_uniform_(self.rnn.weight_ih, gain=50)
        self.rnn.weight_ih *= torch.bernoulli(self.rnn.weight_ih, input_conn_prob)
        self.rnn.bias_hh.zero_()
        self.rnn.bias_ih.zero_()

        self.rnn.weight_ih *= wmask

        set_spectral_radius(self.rnn, self.hidden_size, radius=spectral_radius)

    # #RNN
    def forward(self, x):
        self.h = self.rnn(x, self.h)
        return self.h


input_size = 20
input_scale = 5
res = Reservoir_Local(
    input_size,
    num_unique=10,
    num_shared=5,
    spectral_radius=1.0,
    reach=15,
    local_conn_prob=1,
    global_conn_prob=0.025,
    local_weight_scale=1,
    global_weight_scale=1,
)

# Self connections
# res.rnn.weight_hh *= 1-torch.eye(res.rnn.hidden_size)
# res.rnn.weight_hh += torch.eye(res.rnn.hidden_size)*0.1

plt.imshow(res.rnn.weight_hh)
# plt.show()
savefig("res_weight_hh.png")


res.h *= 0
x = torch.randn(input_size) * input_scale
z = res(x)
vecs = []
for i in range(250):
    z = res(x * 0)
    vecs.append(z.unsqueeze(0))
out = torch.cat(vecs).squeeze().numpy()

plt.imshow(out.T)
savefig("res_activity.png")
# plt.show()


class Reservoir_Stochastic_Local(nn.Module):
    def __init__(
        self,
        input_size,
        num_unique=10,
        num_shared=5,
        spectral_radius=1,
        reach=10,
        input_conn_prob=0.5,
        local_conn_prob=0.5,
        local_weight_scale=1,
        global_conn_prob=0.01,
        global_weight_scale=1,
    ):
        super().__init__()

        wmask = createRollingMask(input_size, num_unique, num_shared)
        self.hidden_size = wmask.shape[0]

        self.rnn = nn.RNNCell(input_size, self.hidden_size).requires_grad_(False)
        self.h = torch.zeros(self.hidden_size)
        self.rnn.weight_hh = nn.Parameter(
            random_band_matrix(self.hidden_size, self.hidden_size, reach) * local_weight_scale
        ).requires_grad_(False)
        self.rnn.weight_hh *= torch.bernoulli(self.rnn.weight_hh, local_conn_prob)

        mask_local = (self.rnn.weight_hh == 0).float()

        newCons_hh = (
            (2 * torch.rand_like(self.rnn.weight_hh) - 1)
            * torch.bernoulli(self.rnn.weight_hh, global_conn_prob)
            * global_weight_scale
            * mask_local
        )
        self.rnn.weight_hh += newCons_hh

        torch.nn.init.xavier_uniform_(self.rnn.weight_ih, gain=50)
        self.rnn.weight_ih *= torch.bernoulli(self.rnn.weight_ih, input_conn_prob)
        self.rnn.bias_hh.zero_()
        self.rnn.bias_ih.zero_()

        self.rnn.weight_ih *= wmask

        set_spectral_radius(self.rnn, self.hidden_size, radius=spectral_radius)

    # #RNN
    def forward(self, x):
        self.h = self.rnn(x, self.h)
        return self.h


input_size = 10
res = Reservoir_Stochastic_Local(
    input_size,
    num_unique=15,
    num_shared=5,
    spectral_radius=0.95,
    reach=15,
    local_conn_prob=0.4,
    global_conn_prob=0.01,
    local_weight_scale=1,
    global_weight_scale=1,
)

bits = 20
bit_scale = 0
input_scale = 5

bin = M.WorkingMemoryBernoulli(res.hidden_size, 256, bits)
b = torch.zeros(bits)

res.h *= 0
x = torch.randn(input_size) * input_scale
# x[:bits] = b*bit_scale

z = res(x)
b = bin(z).mode
# x[:bits] = b*bit_scale

vecs = []
bit_vecs = []
for i in range(250):
    z = res(x)
    b = bin(z).mode
    x *= 0
    res.h = 0.2 * res.h / res.h.abs().mean()
    # x[:bits] = b*bit_scale
    vecs.append(z.unsqueeze(0))
    bit_vecs.append(b.unsqueeze(0))
out = torch.cat(vecs).squeeze().numpy()
bit_out = torch.cat(bit_vecs).squeeze().numpy()

plt.imshow(out.T)
savefig("res_activity.png")
plt.imshow(bit_out.T)
savefig("bit_activity.png")
