import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path to import spiking module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spiking_module_faster import CorticalReservoirSimpler, spiking_pars

class SpikingStep(nn.Module):
    """
    Simplified stateless single-step spiking neural network for ONNX export.
    Reimplements the core dynamics without complex permutation operations.
    """
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim

        # Local excitatory connections
        self.lat_exc = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False, padding_mode="circular")
        nn.init.ones_(self.lat_exc.weight)
        self.lat_exc.weight.data = self.lat_exc.weight.data / self.lat_exc.weight.data.sum()

        # Local inhibitory connections
        self.lat_inh = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False, padding_mode="circular")
        nn.init.ones_(self.lat_inh.weight)
        self.lat_inh.weight.data = self.lat_inh.weight.data / self.lat_inh.weight.data.sum()

    def forward(self, V0, S0, U, decay, thr, reset, input_split,
                exc_local, exc_global, inh_local, inh_global):

        # Update membrane potential with decay and external input
        V = decay * V0 + input_split * U

        # Compute lateral interactions from previous spikes
        lateral = exc_local * self.lat_exc(S0) + inh_local * self.lat_inh(S0)

        # Apply lateral interactions where voltage is above lower threshold
        active = V > -0.09
        V = torch.where(active, V + (1 - input_split) * U + lateral, V)

        # Clamp voltage to [0, 1]
        V = torch.clamp(V, 0, 1)

        # Generate spikes
        spikes = (V > thr).float()

        # Reset spiked neurons
        V = torch.where(spikes > 0, reset, V)

        # Create visualization (voltage with spike overlay)
        Y = torch.clamp(V + 0.5 * spikes, 0, 1)

        return V, spikes, Y

def export_onnx_model():
    """Export the spiking neural network model to ONNX format for web deployment."""

    # Create the simplified model
    model = SpikingStep(dim=256).eval()

    # Set up example inputs
    H, W = spiking_pars["dim"], spiking_pars["dim"]

    # State tensors
    V0 = torch.zeros(1, 1, H, W)
    S0 = torch.zeros(1, 1, H, W)
    U = torch.zeros(1, 1, H, W)  # External stimulation

    # Parameter scalars (using default values)
    decay = torch.tensor(spiking_pars["decay"])
    thr = torch.tensor(spiking_pars["firing_threshold"])
    reset = torch.tensor(spiking_pars["reset_point"])
    input_split = torch.tensor(spiking_pars["input_split"])
    exc_local = torch.tensor(spiking_pars["exc_local_scale"])
    exc_global = torch.tensor(spiking_pars["exc_global_scale"])
    inh_local = torch.tensor(spiking_pars["inh_local_scale"])
    inh_global = torch.tensor(spiking_pars["inh_global_scale"])

    example_inputs = (V0, S0, U, decay, thr, reset, input_split,
                     exc_local, exc_global, inh_local, inh_global)

    # Export to ONNX
    torch.onnx.export(
        model,
        example_inputs,
        "spiking_step.onnx",
        input_names=["V0", "S0", "U", "decay", "thr", "reset", "input_split",
                     "exc_local", "exc_global", "inh_local", "inh_global"],
        output_names=["V1", "S1", "Y"],
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )

    print("✓ Exported ONNX model to spiking_step.onnx")
    print(f"✓ Model input shape: V0,S0,U = [1,1,{H},{W}]")
    print("✓ Model parameters: 8 scalar values")
    print("✓ Ready for web deployment!")

if __name__ == "__main__":
    export_onnx_model()