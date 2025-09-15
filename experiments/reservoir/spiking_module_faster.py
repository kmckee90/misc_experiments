import torch
import torch.nn as nn

# Structure
# Cortex > Cortical Reservoir > Lateral connections > Conv2d, LayerNorm
#        > Cortical Input > Conv2dTranspose
#        > Conv2d


class CorticalReservoirSimpler(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dim = kwargs["dim"]
        self.n_channel = 1

        self.decay = kwargs["decay"]
        self.firing_threshold = kwargs["firing_threshold"]
        self.reset_point = kwargs["reset_point"]
        self.input_split = kwargs["input_split"]

        self.lower_threshold = kwargs["lower_threshold"]
        self.exc_local_scale = kwargs["exc_local_scale"]
        self.inh_local_scale = kwargs["inh_local_scale"]

        k1 = kwargs["kernel_size_exc_local"]
        d1 = kwargs["kernel_dilation_exc_local"]
        self.lat_exc = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=k1,
            stride=1,
            padding=d1 * (k1 - 1) // 2,
            bias=False,
            padding_mode="circular",
            groups=1,
            dilation=d1,
        )
        self.lat_exc.weight = nn.Parameter(self.lat_exc.weight**0)
        self.lat_exc.weight = nn.Parameter(self.lat_exc.weight / self.lat_exc.weight.sum())

        k1 = kwargs["kernel_size_inh_local"]
        d1 = kwargs["kernel_dilation_inh_local"]
        self.lat_inh = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=k1,
            stride=1,
            padding=d1 * (k1 - 1) // 2,
            bias=False,
            padding_mode="circular",
            groups=1,
            dilation=d1,
        )
        self.lat_inh.weight = nn.Parameter(self.lat_inh.weight**0)
        self.lat_inh.weight = nn.Parameter(self.lat_inh.weight / self.lat_inh.weight.sum())
        self.drop = nn.Dropout(kwargs["drop_prob"])

        self.register_buffer("V", torch.zeros(1, self.n_channel, self.dim, self.dim))
        self.register_buffer("S", self.V.clone())

    def forward(self, x):
        self.V = self.decay * self.V + self.input_split * x

        # Compute lateral conductance with old spikes
        active = self.lower_threshold <= self.V
        self.V[active] += ((1 - self.input_split) * x + self.compute_lateral(self.S))[active]
        self.V[self.V > 1] = 1  # Cap values at 1

        # New spikes
        Sbool = self.firing_threshold < self.V
        Sbool = self.drop(Sbool.float()).bool()
        self.V[Sbool] = self.reset_point
        self.S = Sbool.float()

        return self.V

    def compute_lateral(self, s):
        z = self.exc_local_scale * self.lat_exc(s)
        z += self.inh_local_scale * self.lat_inh(s)
        return z


class CorticalInput(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        self.input_square_dim = kwargs["input_square_dim"]
        self.internal_channels = 1
        self.input_mask_prob_fine = kwargs["input_mask_prob_fine"]
        self.input_mask_prob_coarse = kwargs["input_mask_prob_coarse"]
        self.input_scale = kwargs["input_scale"]
        self.output_square_dim = kwargs["dim"]
        self.square_embed = nn.Linear(input_dim, self.internal_channels * self.input_square_dim**2, bias=False)
        self.deconv = nn.ConvTranspose2d(
            in_channels=self.internal_channels,
            out_channels=1,
            kernel_size=self.output_square_dim // self.input_square_dim,
            stride=self.output_square_dim // self.input_square_dim,
            bias=False,
        )

        # This localizes inputs
        W0 = torch.zeros_like(self.square_embed.weight)
        W0.fill_diagonal_(1)
        W0 = W0[torch.randperm(W0.shape[0])]
        self.square_embed.weight = nn.Parameter(W0)

        self.deconv.weight = nn.Parameter(self.deconv.weight**0)
        self.norm = nn.LayerNorm((1, 1, self.output_square_dim, self.output_square_dim))

        self.register_buffer(
            "input_mask_fine",
            torch.zeros(1, 1, self.output_square_dim, self.output_square_dim).bernoulli(self.input_mask_prob_fine),
        )
        self.register_buffer(
            "input_mask_coarse",
            torch.zeros(1, 1, self.input_square_dim, self.input_square_dim).bernoulli(self.input_mask_prob_coarse),
        )

    def forward(self, x):
        u = self.square_embed(x)
        u = u.reshape(-1, self.internal_channels, self.input_square_dim, self.input_square_dim)
        u = u * self.input_mask_coarse
        u = self.deconv(u)
        u = self.input_scale * torch.tanh(u) * self.input_mask_fine
        return u


class Cortex(nn.Module):
    def __init__(self, input_dim, output_size, **kwargs):
        super().__init__()
        self.cortex_input = CorticalInput(input_dim, **kwargs).requires_grad_(False)
        self.cortex_res = CorticalReservoirSimpler(**kwargs).requires_grad_(False)
        self.cortical_output = torch.nn.Conv2d(2, output_size, kernel_size=kwargs["dim"])

    def forward(self, X):
        with torch.no_grad():
            X = self.cortex_input(X)
            outputs = []
            for i in range(X.shape[-4]):
                for _ in range(1):
                    _ = self.cortex_res(X[i])
                outputs.append(torch.stack([self.cortex_res.V.squeeze(), self.cortex_res.S.float().squeeze()]))
            outputs = torch.stack(outputs)
        outputs = self.cortical_output(outputs).squeeze()
        return outputs


spiking_pars = {
    "dim": 64,
    "input_square_dim": 16,
    "internal_channels": 1,
    "decay": 0.99,
    "firing_threshold": 0.99,
    "reset_point": -0.1,
    "input_split": 0.1,
    "drop_prob": 0.5,
    "lower_threshold": -0.090,
    "exc_local_scale": 6,
    "inh_local_scale": -3,
    "kernel_size_exc_local": 5,
    "kernel_dilation_exc_local": 1,
    "kernel_size_inh_local": 5,
    "kernel_dilation_inh_local": 1,
    "input_mask_prob_fine": 0.5,
    "input_mask_prob_coarse": 0.25,
    "input_scale": 0.5,
}


if __name__ == "__main__":
    import time

    import pygame

    time_step = 0.05
    scaled_size = (1024, 1024)
    input_dim = 12
    input_frequency = 1

    pygame.init()
    size = (spiking_pars["dim"], spiking_pars["dim"])
    screen = pygame.display.set_mode(scaled_size)
    pygame.display.set_caption("2D Reservoir animations")

    model = Cortex(
        input_dim,
        output_size=16,
        **spiking_pars,
    ).to("cpu")

    data = torch.load("spiking_evo_data.pklk", map_location="cpu")

    X = torch.cat(data[:10], 0)
    with torch.no_grad():
        for i in range(10000):
            print(i)
            y = model(X[i] * (i < 300))

            # Pygame visualization
            Vp = model.cortex_res.V.clone().cpu().squeeze()
            Vp[Vp < 0] = 0
            Sp = model.cortex_res.S.clone().cpu().float().squeeze()
            Z = torch.stack([Vp * 0, Sp, Vp])

            frame = Z.permute(1, 2, 0).numpy()
            frame = frame * 255
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            scaled_surface = pygame.transform.scale(surface, scaled_size)
            _ = screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()
            time.sleep(time_step)
