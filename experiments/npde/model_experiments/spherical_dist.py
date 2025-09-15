import torch


def sample_hypersphere(size, radius, thickness=0.1):
    samples = torch.randn(size)
    norms = torch.norm(samples, dim=-1, keepdim=True)
    radius = radius + torch.randn_like(norms) * thickness
    points = radius * samples / norms
    return points
