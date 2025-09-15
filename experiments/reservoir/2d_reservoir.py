import time

# import matplotlib.pyplot as plt
# import numpy as np
import pygame
import torch
import torch.nn as nn

N = 50  # Example size
pygame.init()
size = (N, N)
scaled_size = (512, 512)
screen = pygame.display.set_mode(scaled_size)
pygame.display.set_caption("2D Reservoir animations")

W = torch.zeros((N**2, N**2))
for i in range(N):
    for j in range(N):
        k = i * N + j  # Flattened index
        neighbors = [
            (i - 1, j - 1),
            (i - 1, j),
            (i - 1, j + 1),
            (i, j - 1),
            (i, j + 1),
            (i + 1, j - 1),
            (i + 1, j),
            (i + 1, j + 1),
        ]
        for ni, nj in neighbors:
            if 0 <= ni < N and 0 <= nj < N:
                nk = ni * N + nj
                W[k, nk] = 1  # or some weight

W = W * torch.rand_like(W)
W.fill_diagonal_(-1.0)

conv = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3, bias=False).requires_grad_(False)

conv.weight = nn.Parameter(conv.weight / torch.linalg.eigvals(conv.weight).real.abs().max())
# W[W == 0] = W[W == 0].bernoulli(0.025)

radius = 1.0
eigs = torch.abs(torch.real(torch.linalg.eigvals(W)))
maxEig = eigs.max()
W *= radius / maxEig

ln = nn.LayerNorm(N * N).requires_grad_(False)


p_input = 0.02
X0 = torch.zeros(N, N).bernoulli(p_input) - torch.zeros(N, N).bernoulli(p_input)
X = X0.clone().unsqueeze(0)
U = X0.clone().unsqueeze(0)


for i in range(1000):
    U = (
        torch.zeros_like(U).bernoulli(p_input) - torch.zeros_like(U).bernoulli(p_input)
        if i % 50 == 0
        else torch.zeros_like(U)
    )

    # X = torch.tanh(X.view(-1) @ W).view(N, N) + U
    X = torch.tanh(conv(X)) + U

    # X = ln(X.view(-1)).view(1, N, N)

    Z = X.clone().squeeze()
    Z[0, 0] = -1
    Z[0, -1] = 1
    frame = torch.stack((Z, Z * 0.5, -Z)).permute(1, 2, 0).numpy()
    frame = frame + 1
    frame = frame * (255 / 2)
    surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    scaled_surface = pygame.transform.scale(surface, scaled_size)
    _ = screen.blit(scaled_surface, (0, 0))
    pygame.display.flip()
    time.sleep(0.01)


N = 256  # Example size
pygame.init()
size = (N, N)
scaled_size = (1024, 1024)
screen = pygame.display.set_mode(scaled_size)
pygame.display.set_caption("2D Reservoir animations")


c_prob = 1
kernel_size = 3
radius = 0.4
p_input = 0.01

with torch.no_grad():
    conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
    conv.weight = nn.Parameter(torch.ones_like(conv.weight))
    conv.weight = nn.Parameter(radius * conv.weight / torch.linalg.eigvals(conv.weight).real.abs().max())
    conv.weight = nn.Parameter(conv.weight * conv.weight.bernoulli(c_prob))

    X0 = torch.zeros(N, N).bernoulli(p_input) - torch.zeros(N, N).bernoulli(p_input)
    X = X0.clone().unsqueeze(0)
    U = X0.clone().unsqueeze(0)

    for i in range(1000):
        U = (
            torch.zeros_like(U).bernoulli(p_input) - torch.zeros_like(U).bernoulli(p_input)
            if i % 1 == 0
            else torch.zeros_like(U)
        )

        # X = torch.tanh(X.view(-1) @ W).view(N, N) + U
        X = 0.8 * torch.tanh(conv(X))

        Z = X.clone().squeeze()

        X = X + U
        # X = ln(X.view(-1)).view(1, N, N)

        Z[0, 0] = -1
        Z[0, -1] = 1
        frame = torch.stack((Z, Z * 0.5, -Z)).permute(1, 2, 0).numpy()
        frame = frame + 1
        frame = frame * (255 / 2)
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        scaled_surface = pygame.transform.scale(surface, scaled_size)
        _ = screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        time.sleep(0.01)


N = 256  # Example size
pygame.init()
size = (N, N)
scaled_size = (512, 512)
screen = pygame.display.set_mode(scaled_size)
pygame.display.set_caption("2D Reservoir animations")

ln = nn.LayerNorm(3 * N * N).requires_grad_(False)

c_prob = 0.1
kernel_size = 31
radius = 1
p_input = 0.0025

conv = nn.Conv2d(3, 3, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
conv.weight = nn.Parameter(conv.weight * conv.weight.bernoulli(c_prob))

up = nn.Upsample(scale_factor=(8, 8))

with torch.no_grad():
    # conv.weight = nn.Parameter(torch.ones_like(conv.weight))
    # conv.weight = nn.Parameter(radius * conv.weight / torch.linalg.eigvals(conv.weight).real.abs().max())

    X0 = torch.zeros(3, N, N).bernoulli(p_input) - torch.zeros(3, N, N).bernoulli(p_input)
    X = X0.clone().unsqueeze(0)
    U = torch.zeros(3, N // 8, N // 8)

    for i in range(1000):
        U = (
            torch.zeros_like(U).bernoulli(p_input) - torch.zeros_like(U).bernoulli(p_input)
            if i % 20 == 0
            else torch.zeros_like(U)
        )

        # X = torch.tanh(X.view(-1) @ W).view(N, N) + U
        X = torch.tanh(conv(X))
        X = ln(X.view(-1)).view(3, N, N)

        Z = X.clone().squeeze()

        X = X + 200 * up(U.unsqueeze(0)).squeeze()

        Z = Z - Z.min()
        Z = Z / Z.max()
        frame = Z.permute(1, 2, 0).numpy()
        # frame = frame + 1
        frame = frame * 255
        surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        scaled_surface = pygame.transform.scale(surface, scaled_size)
        _ = screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        time.sleep(0.001)
