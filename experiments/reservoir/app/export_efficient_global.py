import torch
import torch.nn as nn

class EfficientGlobalSpiking(nn.Module):
    """
    Efficient approximation of sparse global connections using multiple strategies:
    1. Strided convolutions to capture distant pixels
    2. Multiple scale dilated convolutions
    3. Subsampling and upsampling for global effects
    """
    def __init__(self):
        super().__init__()

        # Local excitatory connections (kernel size 7)
        self.exc_local_conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False, padding_mode="circular")
        nn.init.ones_(self.exc_local_conv.weight)
        self.exc_local_conv.weight.data = self.exc_local_conv.weight.data / self.exc_local_conv.weight.data.sum()

        # Local inhibitory connections (kernel size 5)
        self.inh_local_conv = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False, padding_mode="circular")
        nn.init.ones_(self.inh_local_conv.weight)
        self.inh_local_conv.weight.data = self.inh_local_conv.weight.data / self.inh_local_conv.weight.data.sum()

        # Global excitatory: multi-scale dilated convolutions to capture distant connections
        self.global_exc_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=8, dilation=8, bias=False, padding_mode="circular")
        self.global_exc_conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=16, dilation=16, bias=False, padding_mode="circular")
        self.global_exc_conv3 = nn.Conv2d(1, 1, kernel_size=3, padding=32, dilation=32, bias=False, padding_mode="circular")

        # Global inhibitory: similar but separate weights
        self.global_inh_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=8, dilation=8, bias=False, padding_mode="circular")
        self.global_inh_conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=16, dilation=16, bias=False, padding_mode="circular")
        self.global_inh_conv3 = nn.Conv2d(1, 1, kernel_size=3, padding=32, dilation=32, bias=False, padding_mode="circular")

        # Initialize global connection weights with sparse random patterns
        for conv in [self.global_exc_conv1, self.global_exc_conv2, self.global_exc_conv3,
                     self.global_inh_conv1, self.global_inh_conv2, self.global_inh_conv3]:
            # Sparse initialization: mostly zeros with some random connections
            nn.init.zeros_(conv.weight)
            # Randomly set a few weights to create sparse connections
            mask = torch.rand_like(conv.weight) < 0.3  # 30% connectivity
            conv.weight.data[mask] = torch.randn_like(conv.weight.data[mask]) * 0.1
            conv.weight.data = conv.weight.data / (conv.weight.data.abs().sum() + 1e-8)

    def forward(self, V0, S0, U, decay, thr, reset, input_split,
                exc_local, exc_global, inh_local, inh_global, drop_prob, lower_thr):

        # Update membrane potential with decay and external input
        V = decay * V0 + input_split * U

        # Compute lateral interactions from previous spikes
        lateral = torch.zeros_like(V)

        # Local connections
        lateral += exc_local * self.exc_local_conv(S0)
        lateral += inh_local * self.inh_local_conv(S0)

        # Global connections: multi-scale dilated convolutions
        if exc_global != 0:
            global_exc = (self.global_exc_conv1(S0) +
                         self.global_exc_conv2(S0) +
                         self.global_exc_conv3(S0)) / 3.0
            lateral += exc_global * global_exc

        if inh_global != 0:
            global_inh = (self.global_inh_conv1(S0) +
                         self.global_inh_conv2(S0) +
                         self.global_inh_conv3(S0)) / 3.0
            lateral += inh_global * global_inh

        # Apply lateral interactions where voltage is above lower threshold
        active = V > lower_thr
        V = torch.where(active, V + (1 - input_split) * U + lateral, V)

        # Clamp voltage to reasonable range
        V = torch.clamp(V, -2, 2)

        # Generate spikes
        spikes = (V > thr).float()

        # Apply dropout to spikes (deterministic spatial pattern)
        if drop_prob > 0:
            # Create a deterministic spatial dropout pattern
            h_idx = torch.arange(V.shape[2], device=V.device, dtype=torch.float32).view(-1, 1)
            w_idx = torch.arange(V.shape[3], device=V.device, dtype=torch.float32).view(1, -1)
            spatial_pattern = torch.sin(h_idx * 0.1) * torch.cos(w_idx * 0.1)
            dropout_mask = (spatial_pattern + 1) * 0.5 > drop_prob
            spikes = spikes * dropout_mask.unsqueeze(0).unsqueeze(0)

        # Reset spiked neurons
        V = torch.where(spikes > 0, reset, V)

        # Create visualization (voltage normalized + spike overlay)
        Y = torch.clamp((V + 1) * 0.5, 0, 1) + 0.5 * spikes
        Y = torch.clamp(Y, 0, 1)

        return V, spikes, Y

def export_efficient_global_onnx():
    """Export spiking model with efficient global connections via dilated convolutions."""

    model = EfficientGlobalSpiking().eval()

    H, W = 256, 256

    # State tensors
    V0 = torch.zeros(1, 1, H, W)
    S0 = torch.zeros(1, 1, H, W)
    U = torch.zeros(1, 1, H, W)

    # Parameters
    decay = torch.tensor(0.99)
    thr = torch.tensor(0.99)
    reset = torch.tensor(-0.1)
    input_split = torch.tensor(0.1)
    exc_local = torch.tensor(6.0)
    exc_global = torch.tensor(2.0)
    inh_local = torch.tensor(-2.0)
    inh_global = torch.tensor(-0.1)
    drop_prob = torch.tensor(0.5)
    lower_thr = torch.tensor(-0.090)

    example_inputs = (V0, S0, U, decay, thr, reset, input_split,
                     exc_local, exc_global, inh_local, inh_global, drop_prob, lower_thr)

    torch.onnx.export(
        model,
        example_inputs,
        "spiking_step_efficient_global.onnx",
        input_names=["V0", "S0", "U", "decay", "thr", "reset", "input_split",
                     "exc_local", "exc_global", "inh_local", "inh_global", "drop_prob", "lower_thr"],
        output_names=["V1", "S1", "Y"],
        opset_version=11,
        do_constant_folding=False,
        verbose=False
    )

    print("✓ Exported ONNX model with efficient global connections")
    print("✓ Uses multi-scale dilated convolutions (dilation: 8, 16, 32)")
    print("✓ Approximates sparse long-range connections")
    print("✓ Much more efficient than full permutation matrix")

if __name__ == "__main__":
    export_efficient_global_onnx()