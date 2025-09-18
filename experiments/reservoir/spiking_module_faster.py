import torch
import torch.nn as nn
from PIL import Image
import numpy as np
# Structure
# Cortex > Cortical Reservoir > Lateral connections > Conv2d, LayerNorm
#        > Cortical Input > Conv2dTranspose
#        > Conv2d
DEFAULT_DEVICE = "cuda"
torch.set_default_device(DEFAULT_DEVICE)

class PermutedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, H, W, **conv_kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)
        N = H * W
        self.register_buffer('perm', torch.randperm(N))
        inv = torch.empty_like(self.perm)
        inv[self.perm] = torch.arange(N)
        self.register_buffer('inv', inv)
        self.H, self.W = H, W

    def _permute(self, x, perm):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)[:, :, perm]
        return x.view(B, C, H, W)

    def forward(self, x):
        x = self._permute(x, self.perm)
        x = self.conv(x)
        return self._permute(x, self.inv)

    def __getattr__(self, name):
        if name != 'conv' and hasattr(self, 'conv') and hasattr(self.conv, name):
            return getattr(self.conv, name)
        return super().__getattr__(name)


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
        self.exc_global_scale = kwargs["exc_global_scale"]
        self.inh_global_scale = kwargs["inh_global_scale"]

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


        k1 = kwargs["kernel_size_exc_global"]
        d1 = kwargs["kernel_dilation_exc_global"]
        self.global_exc = PermutedConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=k1,
            stride=1,
            padding=d1 * (k1 - 1) // 2,
            bias=False,
            padding_mode="circular",
            groups=1,
            dilation=d1,
            H = kwargs["dim"],
            W = kwargs["dim"],
        )
        self.global_exc.conv.weight = nn.Parameter(self.global_exc.conv.weight**0)
        self.global_exc.conv.weight = nn.Parameter(self.global_exc.conv.weight / self.global_exc.conv.weight.sum())



        k1 = kwargs["kernel_size_inh_global"]
        d1 = kwargs["kernel_dilation_inh_global"]
        self.global_inh = PermutedConv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=k1,
            stride=1,
            padding=d1 * (k1 - 1) // 2,
            bias=False,
            padding_mode="circular",
            groups=1,
            dilation=d1,
            H = kwargs["dim"],
            W = kwargs["dim"],
        )
        self.global_inh.conv.weight = nn.Parameter(self.global_inh.conv.weight**0)
        self.global_inh.conv.weight = nn.Parameter(self.global_inh.conv.weight / self.global_inh.conv.weight.sum())


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
        # z = self.exc_global_scale * self.global_exc(s)
        z = self.exc_local_scale * self.lat_exc(s)
        # z += self.inh_global_scale * self.global_inh(s)
        z += self.inh_local_scale * self.lat_inh(s)
        z += 2*s.mean()
        return z


class CorticalInput(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        self.input_square_dim = kwargs["input_square_dim"]
        self.internal_input_channels = 1
        self.input_mask_prob_fine = kwargs["input_mask_prob_fine"]
        self.input_mask_prob_coarse = kwargs["input_mask_prob_coarse"]
        self.input_scale = kwargs["input_scale"]
        self.output_square_dim = kwargs["dim"]
        self.square_embed = nn.Linear(input_dim, self.internal_input_channels * self.input_square_dim**2, bias=False)
        self.deconv = nn.ConvTranspose2d(
            in_channels=self.internal_input_channels,
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
        u = u.reshape(-1, self.internal_input_channels, self.input_square_dim, self.input_square_dim)
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
    "dim": 256,

    "decay": 0.99,
    "firing_threshold": 0.99,
    "reset_point": -0.1,
    "drop_prob": 0.5,
    "lower_threshold": -0.090,

    "exc_local_scale": 6.0,
    "exc_global_scale": 2.0,    
    "inh_local_scale": -1.0,
    "inh_global_scale": -0.1,

    "kernel_size_exc_local": 7,
    "kernel_dilation_exc_local": 1,
    "kernel_size_exc_global": 3,
    "kernel_dilation_exc_global": 1,
    "kernel_size_inh_local": 5,
    "kernel_dilation_inh_local": 1,
    "kernel_size_inh_global": 3,
    "kernel_dilation_inh_global": 1,

    "input_square_dim": 32,
    "input_mask_prob_fine": 0.5,
    "input_mask_prob_coarse": 0.25,
    "internal_input_channels": 1,
    "input_split": 0.1,
    "input_scale": 0.5,

}


if __name__ == "__main__":
    # import time
    # import pygame

    time_step = 0.05
    scaled_size = (512, 512)
    input_dim = 24
    input_frequency = 1

    # pygame.init()
    # size = (spiking_pars["dim"], spiking_pars["dim"])
    # screen = pygame.display.set_mode(scaled_size)
    # pygame.display.set_caption("2D Reservoir animations")

    model = Cortex(
        input_dim,
        output_size=16,
        **spiking_pars,
    )
    # data = torch.load("spiking_evo_data.pklk", map_location=DEFAULT_DEVICE)

    data = [torch.randn((256, 24)) for _ in range(80)]
    
    
    X = torch.cat(data[:10], 0)
    frames = []
    from tqdm import tqdm
    with torch.no_grad():
        for i in tqdm(range(200), ncols=100):
            y = model(X[i] * (i < 300))

            # Visualization
            Vp = model.cortex_res.V.clone().cpu().squeeze()
            Vp[Vp < 0] = 0
            Sp = model.cortex_res.S.clone().cpu().float().squeeze()
            Z = torch.stack([Vp*0, Sp, Vp])
            frame = Z.permute(1, 2, 0).numpy()
            frame = frame * 255
            frame = frame.astype(np.uint8)
            img = Image.fromarray(frame)
            frames.append(img)
            
            # surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            # scaled_surface = pygame.transform.scale(surface, scaled_size)
            # _ = screen.blit(scaled_surface, (0, 0))
            # pygame.display.flip()
            # time.sleep(time_step)
        frames[0].save(
        "spiking.gif",
        save_all=True,
        append_images=frames[1:],
        duration=4,  # milliseconds per frame
        loop=0         # 0 = infinite loop
        )