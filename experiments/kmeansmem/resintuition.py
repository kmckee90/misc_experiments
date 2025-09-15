import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt


from modules import Reservoir, Reservoir_Local

d0 = 1
torch.manual_seed(12345)
rnn = Reservoir(d0, 32, spectral_radius=1, connection_probability=1).requires_grad_(False)
# rnn = Reservoir_Local(d0, 10, num_shared=0).requires_grad_(False)

k = 1
s = 3
r = 5

x = torch.randn(s, d0)

# x = 2 * torch.tensor([2 * (j / k) - 1])
x = torch.cat([x] * r)
# x = x + x.roll(1)

y = -x

# x = -x
# y = -y

for j in range(k):
    rnn.h *= 0
    H = []
    # x = torch.randn(d0)

    for i in range(s * r):
        h = rnn(y[i % s])
        u = h.clone().detach().cpu()
        H.append(u)
        if i == 0:
            plt.scatter(h[8], h[9])

    # for i in range(s * r):
    #     h = rnn(x[i])
    #     u = h.clone().detach().cpu()
    #     H.append(u)

    # for i in range(s * r):
    #     h = rnn(y[i % s])
    #     u = h.clone().detach().cpu()
    #     H.append(u)

    v = torch.stack(H)

    plt.plot(v[:, 8], v[:, 9], linewidth=0.25)

# for j in range(k):
#     rnn.h *= 0
#     H = []
#     # x = torch.randn(d0)

#     for i in range(s * r):
#         h = rnn(x[i % s])
#         u = h.clone().detach().cpu()
#         H.append(u)
#         if i == 0:
#             plt.scatter(h[0], h[1])

#     for i in range(s * r):
#         h = rnn(y[i % s])
#         u = h.clone().detach().cpu()
#         H.append(u)
#     v = torch.stack(H)

#     plt.plot(v[:, 0], v[:, 1], linewidth=0.25)

# plt.xlim(-1.5, 1.5)
# plt.ylim(-1.5, 1.5)

plt.savefig("resintuit.png", dpi=300)
plt.close()
