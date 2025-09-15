import time

import pygame
import torch
import torch.nn as nn

N = 64  # Example size
n_channel = 3
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

strength = 1 / kernel_size**2
kern = strength * torch.ones(kernel_size, kernel_size)
kern[1, 1] = 1.1 * strength
kern = torch.stack([kern, kern, kern]).unsqueeze(1)


def nfunc(x):
    return torch.softmax(2.8 * x.abs(), 1)


up = nn.Upsample(scale_factor=(input_upsample, input_upsample))
bias1 = bias_scale * 1 * up(torch.randn(1, n_channel, N // input_upsample, N // input_upsample))


with torch.no_grad():
    conv1 = nn.Conv2d(
        n_channel,
        n_channel,
        kernel_size,
        stride=1,
        padding=kernel_size // 2,
        bias=False,
        padding_mode="circular",
        groups=3,
    )
    conv1.weight = nn.Parameter(kern)

    time.sleep(1)
    X = torch.zeros(1, n_channel, N, N)
    U = torch.zeros_like(X)
    i = 0
    for _ in range(10000):
        i += 1
        if i == 600:
            i = 0
        print(i)
        U = (
            torch.randn(1, n_channel, N, N) * torch.zeros(1, n_channel, N, N).bernoulli(p_input)
            if i < 200 or i > 400
            else torch.zeros_like(U)
        )
        # U[:,1:,:,:]=0
        # U0 = up(U)

        X = nfunc(conv1(X)) + 10 * U

        Z = X.clone().squeeze()

        # X = X[:,0,:,:]
        # Z = torch.stack([Z,Z,-Z])

        # Z -= Z.min()
        # Z /= Z.max()
        frame = Z.permute(1, 2, 0).numpy()
        frame = frame * 255
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        scaled_surface = pygame.transform.scale(surface, scaled_size)
        _ = screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        time.sleep(0.01)
