import torch
import torch.nn as nn

# Structure
# Cortex > Cortical Reservoir > Lateral connections > Conv2d, LayerNorm
#        > Cortical Input > Conv2dTranspose
#        > Conv2d


class LateralConnections(nn.Module):
    def __init__(
        self, n_channel, kernel_size, padding_mode="circular", conn_exp=0.0, conn_prob=1.0, dilation=1, groups=1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            n_channel,
            n_channel,
            kernel_size,
            stride=1,
            padding=dilation * (kernel_size - 1) // 2,
            bias=False,
            padding_mode=padding_mode,
            groups=groups,
            dilation=dilation,
        )
        self.conv.weight = nn.Parameter(self.conv.weight.abs() ** conn_exp * self.conv.weight.bernoulli(conn_prob))
        wsums = self.conv.weight.sum((1, 2, 3)).reshape(-1, 1, 1, 1)
        self.conv.weight = nn.Parameter(self.conv.weight / wsums)

    def forward(self, x):
        return self.conv(x)


class CorticalReservoirSimpler(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dim = kwargs["dim"]
        self.n_channel = kwargs["n_channel"]
        self.conv1_scale = kwargs["s1"]
        self.conv2_scale = kwargs["s2"]
        self.conv1 = LateralConnections(
            self.n_channel,
            kernel_size=kwargs["k1"],
            conn_prob=kwargs["p1"],
            dilation=kwargs["d1"],
            conn_exp=1,
            groups=self.n_channel,
        )
        self.conv2 = LateralConnections(
            self.n_channel,
            kernel_size=kwargs["k2"],
            conn_prob=kwargs["p2"],
            dilation=kwargs["d2"],
            conn_exp=1,
            groups=self.n_channel,
        )

        self.register_buffer("Z", torch.zeros(1, self.n_channel, self.dim, self.dim))

    def forward(self, x):
        self.Z = torch.tanh(self.compute_lateral(self.Z) + x)
        return self.Z.reshape(self.n_channel, self.dim, self.dim)

    def compute_lateral(self, x):
        z = self.conv1_scale * self.conv1(x) + self.conv2_scale * self.conv2(x)
        return z


class CorticalInput(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        self.input_square_dim = kwargs["input_square_dim"]
        self.internal_channels = kwargs["internal_channels"]
        self.input_scale = kwargs["input_scale"]
        self.square_embed = nn.Linear(input_dim, self.internal_channels * self.input_square_dim**2, bias=False)
        self.kernel_size = kwargs["dim"] // self.input_square_dim

        self.deconv = nn.ConvTranspose2d(
            in_channels=self.internal_channels,
            out_channels=kwargs["n_channel"],
            kernel_size=self.kernel_size,
            stride=self.kernel_size,
            bias=False,
        )

        # This localizes inputs
        # W0 = torch.zeros_like(self.square_embed.weight)
        # W0.fill_diagonal_(1)
        # W0 = W0[torch.randperm(W0.shape[0])]
        # self.square_embed.weight = nn.Parameter(W0)

        self.register_buffer(
            "input_mask_coarse",
            torch.zeros(1, 1, self.input_square_dim, self.input_square_dim).bernoulli(kwargs["p_mask_coarse"]),
        )
        self.register_buffer(
            "input_mask_fine",
            torch.zeros(1, 1, kwargs["dim"], kwargs["dim"]).bernoulli(kwargs["p_mask_fine"]),
        )

    def forward(self, x):
        u = self.square_embed(x)
        u = u.reshape(-1, self.internal_channels, self.input_square_dim, self.input_square_dim)
        u = u * self.input_mask_coarse
        u = self.deconv(u)
        u = self.input_scale * u * self.input_mask_fine
        return u


class Res2D(nn.Module):
    def __init__(self, input_dim, output_size, **kwargs):
        super().__init__()
        torch.manual_seed(kwargs["seed"])
        self.cortex_input = CorticalInput(input_dim, **kwargs).requires_grad_(False)
        self.cortex_res = CorticalReservoirSimpler(**kwargs).requires_grad_(False)
        self.cortical_output = torch.nn.Conv2d(
            kwargs["n_channel"],
            output_size // (kwargs["output_square_dim"] ** 2),
            kernel_size=kwargs["dim"] // kwargs["output_square_dim"],
            stride=kwargs["dim"] // kwargs["output_square_dim"],
        )
        torch.manual_seed(torch.randint(0, 999999, (1,)).item())

    def forward(self, X):
        with torch.no_grad():
            X = self.cortex_input(X)
            outputs = []
            for i in range(X.shape[-4]):
                x = self.cortex_res(X[i])
                outputs.append(x)
            outputs = torch.stack(outputs)
        outputs = self.cortical_output(outputs).reshape(outputs.shape[0], -1)
        return outputs


if __name__ == "__main__":
    import random
    import time

    import pygame

    Res2D_pars = {
        "seed": 12346,
        "dim": 64,
        "n_channel": 2,
        "input_square_dim": 1,
        "output_square_dim": 1,
        "internal_channels": 3,
        "input_scale": 0.11388,
        "s1": 0.00123,
        "s2": 0.9391,
        "k1": 3,
        "k2": 3,
        "d1": 3,
        "d2": 17,
        "p1": 0.47016,
        "p2": 0.34443,
        "p_mask_coarse": 1,
        "p_mask_fine": 0.1,
    }

    time_step = 0.1
    scaled_size = (1024, 1024)
    input_dim = 12
    input_frequency = 1

    pygame.init()
    size = (Res2D_pars["dim"], Res2D_pars["dim"])
    screen = pygame.display.set_mode(scaled_size)
    pygame.display.set_caption("2D Reservoir animations")

    model = Res2D(
        input_dim,
        output_size=16,
        **Res2D_pars,
    ).to("cpu")

    data = torch.load("spiking_evo_data.pklk", map_location="cpu")

    X = torch.cat(data[:10], 0)
    with torch.no_grad():
        for i in range(10000):
            print(i)
            y = model(X[i] * (i < 300))

            # Pygame visualization
            _ = y.clone().cpu().squeeze()
            Z = model.cortex_res.Z.squeeze()  # torch.stack([Y * 0, Y, Y])
            # Z = Z / Z.abs().max()
            # Z = torch.softmax(30 * Z, 0)

            frame = Z.permute(1, 2, 0).numpy()
            frame = (frame + 1) / 2 * 255
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            scaled_surface = pygame.transform.scale(surface, scaled_size)
            _ = screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()
            time.sleep(time_step)
