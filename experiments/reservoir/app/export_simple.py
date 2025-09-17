import torch
import torch.nn as nn

class SimpleSpiking(nn.Module):
    """
    Minimal spiking neural network for ONNX export - just the core dynamics.
    """
    def __init__(self):
        super().__init__()

    def forward(self, V0, S0, U, decay, thr, reset, input_split,
                exc_local, exc_global, inh_local, inh_global):
        # Simple membrane potential update
        V = decay * V0 + input_split * U

        # Simple lateral interactions (just average pooling to avoid conv issues)
        kernel_size = 3
        pad = kernel_size // 2
        S0_pad = torch.nn.functional.pad(S0, (pad, pad, pad, pad), mode='circular')
        lateral = torch.nn.functional.avg_pool2d(S0_pad, kernel_size, stride=1, padding=0)
        lateral = exc_local * lateral

        V = V + (1 - input_split) * U + lateral
        V = torch.clamp(V, -1, 1)

        # Generate spikes
        spikes = (V > thr).float()

        # Reset
        V = torch.where(spikes > 0, reset, V)

        # Visualization
        Y = torch.clamp(V * 0.5 + 0.5, 0, 1)

        return V, spikes, Y

def export_onnx():
    model = SimpleSpiking().eval()

    H, W = 256, 256
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

    torch.onnx.export(
        model,
        (V0, S0, U, decay, thr, reset, input_split, exc_local, exc_global, inh_local, inh_global),
        "spiking_step.onnx",
        input_names=["V0", "S0", "U", "decay", "thr", "reset", "input_split", "exc_local", "exc_global", "inh_local", "inh_global"],
        output_names=["V1", "S1", "Y"],
        opset_version=11,
        do_constant_folding=False
    )

    print("✓ Exported simplified ONNX model")

if __name__ == "__main__":
    export_onnx()