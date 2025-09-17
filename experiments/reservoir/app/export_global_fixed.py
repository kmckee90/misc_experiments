import torch
import torch.nn as nn
import numpy as np

class TrueSparseGlobalSpiking(nn.Module):
    """
    Spiking model with true sparse global connections using a fixed permutation matrix.
    This approximates the PermutedConv2d by pre-computing a sparse global connectivity pattern.
    """
    def __init__(self, H=256, W=256):
        super().__init__()
        self.H, self.W = H, W
        N = H * W

        # Local excitatory connections (kernel size 7)
        self.exc_local_conv = nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False, padding_mode="circular")
        nn.init.ones_(self.exc_local_conv.weight)
        self.exc_local_conv.weight.data = self.exc_local_conv.weight.data / self.exc_local_conv.weight.data.sum()

        # Local inhibitory connections (kernel size 5)
        self.inh_local_conv = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False, padding_mode="circular")
        nn.init.ones_(self.inh_local_conv.weight)
        self.inh_local_conv.weight.data = self.inh_local_conv.weight.data / self.inh_local_conv.weight.data.sum()

        # Sparse global connections using a fixed sparse matrix
        # Create a sparse connectivity matrix that connects each pixel to ~8 random distant pixels
        self.create_sparse_global_connectivity(N, sparsity=8)

    def create_sparse_global_connectivity(self, N, sparsity=8):
        """Create a sparse matrix for global connections."""
        # Create sparse global connectivity matrix
        np.random.seed(42)  # Fixed seed for reproducible connectivity
        global_matrix = torch.zeros(N, N)

        for i in range(N):
            # For each pixel, connect to 'sparsity' random other pixels
            # Avoid connecting to nearby pixels (within local neighborhood)
            candidates = list(range(N))
            # Remove local neighborhood (rough approximation)
            local_start = max(0, i - 50)
            local_end = min(N, i + 50)
            candidates = [c for c in candidates if c < local_start or c > local_end]

            if len(candidates) >= sparsity:
                connections = np.random.choice(candidates, sparsity, replace=False)
                for j in connections:
                    global_matrix[i, j] = 1.0 / sparsity  # Normalize

        # Register as buffer so it's included in ONNX export
        self.register_buffer('global_exc_matrix', global_matrix)
        self.register_buffer('global_inh_matrix', global_matrix)  # Same pattern for simplicity

    def apply_sparse_global(self, spikes, matrix, scale):
        """Apply sparse global connectivity."""
        B, C, H, W = spikes.shape
        spikes_flat = spikes.view(B, C, H * W)  # [B, C, N]

        # Matrix multiplication: [B, C, N] @ [N, N] -> [B, C, N]
        global_flat = torch.matmul(spikes_flat, matrix)

        return global_flat.view(B, C, H, W) * scale

    def forward(self, V0, S0, U, decay, thr, reset, input_split,
                exc_local, exc_global, inh_local, inh_global, drop_prob, lower_thr):

        # Update membrane potential with decay and external input
        V = decay * V0 + input_split * U

        # Compute lateral interactions from previous spikes
        lateral = torch.zeros_like(V)

        # Local connections
        lateral += exc_local * self.exc_local_conv(S0)
        lateral += inh_local * self.inh_local_conv(S0)

        # TRUE sparse global connections
        if exc_global != 0:
            global_exc = self.apply_sparse_global(S0, self.global_exc_matrix, exc_global)
            lateral += global_exc

        if inh_global != 0:
            global_inh = self.apply_sparse_global(S0, self.global_inh_matrix, inh_global)
            lateral += global_inh

        # Apply lateral interactions where voltage is above lower threshold
        active = V > lower_thr
        V = torch.where(active, V + (1 - input_split) * U + lateral, V)

        # Clamp voltage to reasonable range
        V = torch.clamp(V, -2, 2)

        # Generate spikes
        spikes = (V > thr).float()

        # Apply dropout to spikes
        if drop_prob > 0:
            # Deterministic dropout based on spatial position
            y_coords = torch.arange(V.shape[2], device=V.device).view(-1, 1).expand(-1, V.shape[3]).float()
            x_coords = torch.arange(V.shape[3], device=V.device).view(1, -1).expand(V.shape[2], -1).float()
            dropout_mask = ((y_coords + x_coords * 0.7) % 1.0) > drop_prob
            spikes = spikes * dropout_mask.unsqueeze(0).unsqueeze(0)

        # Reset spiked neurons
        V = torch.where(spikes > 0, reset, V)

        # Create visualization (voltage normalized + spike overlay)
        Y = torch.clamp((V + 1) * 0.5, 0, 1) + 0.5 * spikes
        Y = torch.clamp(Y, 0, 1)

        return V, spikes, Y

def export_true_global_onnx():
    """Export spiking model with true sparse global connections."""

    model = TrueSparseGlobalSpiking(H=256, W=256).eval()

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
        "spiking_step_true_global.onnx",
        input_names=["V0", "S0", "U", "decay", "thr", "reset", "input_split",
                     "exc_local", "exc_global", "inh_local", "inh_global", "drop_prob", "lower_thr"],
        output_names=["V1", "S1", "Y"],
        opset_version=11,
        do_constant_folding=False,
        verbose=False
    )

    print("✓ Exported ONNX model with TRUE sparse global connections")
    print("✓ Each pixel connects to ~8 random distant pixels")
    print("✓ Approximates PermutedConv2d behavior")
    print(f"✓ Global connectivity matrix: {256*256} x {256*256} sparse")

if __name__ == "__main__":
    export_true_global_onnx()