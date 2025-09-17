import torch
import torch.nn as nn

class EnhancedSpiking(nn.Module):
    """
    Enhanced spiking neural network that implements the key features:
    - Dropout probability
    - Local and global excitatory/inhibitory connections
    - Multiple connection types with different kernel sizes
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

        # Global excitatory connections (kernel size 3, simulated with avg pool)
        # Global inhibitory connections (kernel size 3, simulated with avg pool)

    def forward(self, V0, S0, U, decay, thr, reset, input_split,
                exc_local, exc_global, inh_local, inh_global, drop_prob, lower_thr):

        # Update membrane potential with decay and external input
        V = decay * V0 + input_split * U

        # Compute lateral interactions from previous spikes
        lateral = torch.zeros_like(V)

        # Local excitatory connections
        lateral += exc_local * self.exc_local_conv(S0)

        # Local inhibitory connections
        lateral += inh_local * self.inh_local_conv(S0)

        # Global connections (simplified as average pooling to avoid permutation complexity)
        if exc_global != 0 or inh_global != 0:
            # Global average with small kernel to simulate global connectivity
            global_kernel_size = 3
            pad = global_kernel_size // 2
            S0_pad = torch.nn.functional.pad(S0, (pad, pad, pad, pad), mode='circular')
            global_signal = torch.nn.functional.avg_pool2d(S0_pad, global_kernel_size, stride=1, padding=0)
            lateral += exc_global * global_signal
            lateral += inh_global * global_signal

        # Apply lateral interactions where voltage is above lower threshold
        active = V > lower_thr
        V = torch.where(active, V + (1 - input_split) * U + lateral, V)

        # Clamp voltage to reasonable range
        V = torch.clamp(V, -2, 2)

        # Generate spikes
        spikes = (V > thr).float()

        # Apply dropout to spikes (random masking)
        if drop_prob > 0:
            # Use sin^2 of scaled voltage for pseudo-random dropout
            pseudo_random = torch.sin(1000 * V) ** 2
            dropout_mask = pseudo_random > drop_prob
            spikes = spikes * dropout_mask.float()

        # Reset spiked neurons
        V = torch.where(spikes > 0, reset, V)

        # Create visualization (voltage normalized + spike overlay)
        Y = torch.clamp((V + 1) * 0.5, 0, 1) + 0.5 * spikes
        Y = torch.clamp(Y, 0, 1)

        return V, spikes, Y

def export_enhanced_onnx():
    """Export enhanced spiking neural network to ONNX format."""

    model = EnhancedSpiking().eval()

    H, W = 256, 256

    # State tensors
    V0 = torch.zeros(1, 1, H, W)
    S0 = torch.zeros(1, 1, H, W)
    U = torch.zeros(1, 1, H, W)

    # Parameters (using spiking_pars defaults)
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
        "spiking_step_enhanced.onnx",
        input_names=["V0", "S0", "U", "decay", "thr", "reset", "input_split",
                     "exc_local", "exc_global", "inh_local", "inh_global", "drop_prob", "lower_thr"],
        output_names=["V1", "S1", "Y"],
        opset_version=11,
        do_constant_folding=False,
        verbose=False
    )

    print("✓ Exported enhanced ONNX model to spiking_step_enhanced.onnx")
    print("✓ Includes: dropout, local/global exc/inh connections")
    print("✓ Model accepts 13 parameters (vs 5 in simplified version)")

if __name__ == "__main__":
    export_enhanced_onnx()