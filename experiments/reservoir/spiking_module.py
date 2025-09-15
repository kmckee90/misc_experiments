import torch
import torch.nn as nn

# Structure
# Cortex > Cortical Reservoir > Lateral connections > Conv2d, LayerNorm
#        > Cortical Input > Conv2dTranspose
#        > Conv2d


class LateralConnections(nn.Module):
    def __init__(
        self,
        n_channel,
        kernel_size,
        padding_mode="circular",
        conn_exp=0.0,
        conn_prob=1.0,
        dilation=1,
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
            groups=1,
            dilation=dilation,
        )
        self.conv.weight = nn.Parameter(self.conv.weight.abs() ** conn_exp * self.conv.weight.bernoulli(conn_prob))
        if conn_prob < 1.0:
            rot_weights = self.conv.weight.clone() * 0
            for i in range(0, 4):
                rot_weights += torch.rot90(self.conv.weight.clone(), k=i)
        self.conv.weight = nn.Parameter(self.conv.weight / self.conv.weight.sum())

    def forward(self, x):
        return self.conv(x)


class CorticalReservoir(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.dim = kwargs["dim"]
        self.n_channel = 1

        self.decay = kwargs["decay"]
        self.firing_threshold = kwargs["firing_threshold"]
        self.reset_point = kwargs["reset_point"]
        self.input_split = kwargs["input_split"]

        self.drop_prob = (kwargs["drop_prob_min"], kwargs["drop_prob_max"])
        self.lower_threshold = (kwargs["lower_threshold_min"], kwargs["lower_threshold_max"])
        self.exc_local_scale = (kwargs["exc_local_scale_min"], kwargs["exc_local_scale_max"])
        self.inh_local_scale = (kwargs["inh_local_scale_min"], kwargs["inh_local_scale_max"])
        self.exc_global_scale = (kwargs["exc_global_scale_min"], kwargs["exc_global_scale_max"])

        # Lateral connections: Random elements tend to be better for larger kernels, too much bias in small kernels.
        self.lat_exc = LateralConnections(
            n_channel=1,
            kernel_size=kwargs["kernel_size_exc_local"],
            conn_exp=0.0,
            conn_prob=1.0,
            dilation=kwargs["kernel_dilation_exc_local"],
        )
        self.lat_inh = LateralConnections(
            n_channel=1,
            kernel_size=kwargs["kernel_size_inh_local"],
            conn_exp=0.0,
            conn_prob=1.0,
            dilation=kwargs["kernel_dilation_inh_local"],
        )
        self.lat_exc_far = LateralConnections(
            n_channel=1,
            kernel_size=kwargs["kernel_size_exc_global"],
            conn_exp=0.0,
            conn_prob=1.0,
            dilation=kwargs["kernel_dilation_exc_global"],
        )

        # Voltage and spike. Voltage is not literal... several accumulating factors such as ligand stores.
        self.register_buffer("V", torch.zeros(1, self.n_channel, self.dim, self.dim))
        self.register_buffer("S", self.V.clone())

        # Modulators are uniformly distributed between the given ranges.
        self.register_buffer(
            "m_exc_local_scale",
            self.exc_local_scale[0]
            + (self.exc_local_scale[1] - self.exc_local_scale[0]) * (torch.rand(1, 1, self.dim, self.dim)),
        )
        self.register_buffer(
            "m_inh_local_scale",
            self.inh_local_scale[0]
            + (self.inh_local_scale[1] - self.inh_local_scale[0]) * (torch.rand(1, 1, self.dim, self.dim)),
        )
        self.register_buffer(
            "m_exc_global_scale",
            self.exc_global_scale[0]
            + (self.exc_global_scale[1] - self.exc_global_scale[0]) * (torch.rand(1, 1, self.dim, self.dim)),
        )
        self.register_buffer(
            "m_p_drop",
            self.drop_prob[0] + (self.drop_prob[1] - self.drop_prob[0]) * (torch.rand(1, 1, self.dim, self.dim)),
        )
        self.register_buffer(
            "m_lower_threshold",
            self.lower_threshold[0]
            + (self.lower_threshold[1] - self.lower_threshold[0]) * (torch.rand(1, 1, self.dim, self.dim)),
        )

    def forward(self, h):
        Sfloat = self.S.float()
        self.V = self.decay * self.V + self.input_split * h
        self.V[self.m_lower_threshold <= self.V] += ((1 - self.input_split) * h + self.compute_lateral(Sfloat))[
            self.m_lower_threshold <= self.V
        ]
        self.V[self.V > 1] = 1
        self.S = self.firing_threshold < self.V

        self.drop = 1 - torch.bernoulli(self.m_p_drop)
        self.S = (self.drop * self.S).bool()
        self.V[self.S] = self.reset_point
        return self.V

    def compute_lateral(self, s):
        z = self.m_exc_local_scale * self.lat_exc(s)
        z += self.m_inh_local_scale * self.lat_inh(s)
        z += self.m_exc_global_scale * self.lat_exc_far(s)
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
        u = u.reshape(1, self.internal_channels, self.input_square_dim, self.input_square_dim)
        u = u * self.input_mask_coarse
        u = self.deconv(u)
        u = self.input_scale * torch.tanh(u) * self.input_mask_fine
        return u


class Cortex(nn.Module):
    def __init__(self, input_dim, output_channels=16, output_kernel_size=64, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.reservoir_dim = kwargs["dim"]

        self.cortex_input = CorticalInput(self.input_dim, **kwargs).requires_grad_(False)
        self.cortex_res = CorticalReservoir(**kwargs).requires_grad_(False)

        # Trainable
        self.output_kernel_size = output_kernel_size
        self.output_channels = output_channels
        self.conv_out = nn.Conv2d(
            in_channels=2,
            out_channels=self.output_channels,
            kernel_size=self.output_kernel_size,
            stride=self.output_kernel_size,  # self.output_kernel_size // 2,
            padding=0,  # self.output_kernel_size // 2,
            padding_mode="circular",
        )
        self.conv_out_shape = len(self.conv_out(torch.zeros(1, 2, self.reservoir_dim, self.reservoir_dim)).flatten())

    def forward(self, X):
        outputs = []
        for i in range(X.shape[-2]):
            with torch.no_grad():
                x = X[i]
                self.input = self.cortex_input(x)
                for _ in range(1):  # sub-iters
                    _ = self.cortex_res(self.input)
                x = torch.stack([self.cortex_res.V.squeeze(), self.cortex_res.S.float().squeeze()])
            y = self.conv_out(x.unsqueeze(0)).flatten()
            outputs.append(y)
        return torch.stack(outputs)


if __name__ == "__main__":
    import time

    import pygame

    spiking_pars = {
        "dim": 256,
        "decay": 0.99,
        "firing_threshold": 0.99,
        "reset_point": -0.12,
        "input_split": 0.1,
        # Random vars
        "drop_prob_min": 0.2,
        "drop_prob_max": 0.3,
        "lower_threshold_min": -0.09,
        "lower_threshold_max": -0.09,
        "exc_local_scale_min": 4,
        "exc_local_scale_max": 8,
        "inh_local_scale_min": -2,
        "inh_local_scale_max": -1,
        "exc_global_scale_min": 1,
        "exc_global_scale_max": 2,
        # Architecture
        "kernel_size_exc_local": 5,
        "kernel_dilation_exc_local": 1,
        "kernel_size_exc_global": 19,
        "kernel_dilation_exc_global": 4,
        "kernel_size_inh_local": 5,
        "kernel_dilation_inh_local": 1,
        # Input params: how inputs perturb the reservoir spatially
        "input_square_dim": 16,
        "internal_channels": 1,
        "input_mask_prob_fine": 0.25,
        "input_mask_prob_coarse": 0.5,
        "input_scale": 0.5,
    }

    time_step = 0.00
    scaled_size = (1024, 1024)
    input_dim = 16
    input_frequency = 1

    pygame.init()
    size = (spiking_pars["dim"], spiking_pars["dim"])
    screen = pygame.display.set_mode(scaled_size)
    pygame.display.set_caption("2D Reservoir animations")

    model = Cortex(
        input_dim,
        output_channels=16,
        output_kernel_size=64,
        **spiking_pars,
    )

    with torch.no_grad():
        for i in range(10000):
            print(i)
            if i % input_frequency == 0:  # and i < 500:
                X = torch.randn(input_dim)
            else:
                X *= 0
            y = model(X)

            # Pygame visualization
            Vp = model.cortex_res.V.clone().cpu().squeeze()
            Vp[Vp < 0] = 0
            Sp = model.cortex_res.S.clone().cpu().float().squeeze()
            Xp = model.input.abs().clone().cpu().squeeze() * 20
            Z = torch.stack([Xp * 0, Sp, Vp])

            frame = Z.permute(1, 2, 0).numpy()
            frame = frame * 255
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            scaled_surface = pygame.transform.scale(surface, scaled_size)
            _ = screen.blit(scaled_surface, (0, 0))
            pygame.display.flip()
            time.sleep(time_step)
