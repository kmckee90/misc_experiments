import time

import pygame
import torch
import torch.nn as nn

N = 256  # Example size
n_channel = 1
pygame.init()
size = (N, N)
scaled_size = (512, 512)
screen = pygame.display.set_mode(scaled_size)
pygame.display.set_caption("2D Reservoir animations")

ln = nn.LayerNorm(1 * N * N).requires_grad_(False)


def norm(x):
    for i in range(x.shape[1]):
        x[0, i] = ln(x[0, i].view(-1)).view(1, N, N)
    return x


p_input = 0.001
input_upsample = 1
c_prob = 1
kernel_size = 3
radius = 1
nfunc_scale = 1
bias_scale = 0.0

# kern_np = mexican_hat_kernel(kernel_size)
# kern = 0.5*torch.tensor(kern_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# kern += kern.min()
# plt.imshow(kern_np)
# plt.show()

strength = 0.1505
kern = strength * torch.ones(kernel_size, kernel_size)
kern[1, 1] = -strength
kern = torch.stack([kern, torch.zeros_like(kern), torch.zeros_like(kern)]).unsqueeze(1)
kern[1, 0, :, :] = torch.ones(kernel_size, kernel_size)
kern[1, :, 1, 1] = 0.0

# kern[1,0,1,:] = 1
# kern[1,0,:,1] = -1
# kern[1,0,1,1] = 0

# kern[2,0,1,:] = -1
# kern[2,0,:,1] = 1
# kern[2,0,1,1] = 0


def nfunc(x):
    return torch.tanh(x)


with torch.no_grad():
    conv1 = nn.Conv2d(
        n_channel, 3, kernel_size, stride=1, padding=kernel_size // 2, bias=False, padding_mode="circular", groups=1
    )
    conv1.weight = nn.Parameter(kern)

    up = nn.Upsample(scale_factor=(input_upsample, input_upsample))

    bias1 = bias_scale * 1 * up(torch.randn(1, n_channel, N // input_upsample, N // input_upsample))

    time.sleep(1)
    U = torch.randn(1, n_channel, N // input_upsample, N // input_upsample)
    X = torch.zeros(1, n_channel, N, N)

    for i in range(10000):
        print(i)
        U = torch.randn(1, n_channel, N, N) if i % 5 == 0 else torch.zeros_like(U)

        U = U * U.bernoulli(10 * p_input)
        # U[:,1:,:,:]=0
        # U0 = up(U)

        X = nfunc(0.95 * conv1(X)) + 1 * U

        Z = X.clone().squeeze()
        X = X[:, 0, :, :]
        # Z = torch.stack([Z,Z,-Z])

        # Z -= Z.min()
        # Z /= Z.max()
        frame = Z.permute(1, 2, 0).numpy()
        frame = (frame + 1) / 2 * 255
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        scaled_surface = pygame.transform.scale(surface, scaled_size)
        _ = screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        time.sleep(0.01)
