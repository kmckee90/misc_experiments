# Spiking Neural Network Web Demo

An interactive WebGPU-accelerated spiking neural network visualization that can be embedded on GitHub Pages via Hugo.

## Features

- **Real-time parameter control**: Adjust neural network parameters via sidebar controls
- **Mouse interaction**: Click and drag to stimulate neurons with excitation or inhibition
- **WebGPU acceleration**: Uses ONNX Runtime Web with WebGPU backend for high performance
- **Continuous rendering**: Smooth real-time visualization of neural activity
- **State preservation**: Network state is maintained when parameters change for real-time modulation

## Quick Start

### 1. Export ONNX Model

First, generate the ONNX model from your PyTorch spiking neural network:

```bash
cd app
source ../../.venv/bin/activate  # Activate your Python environment
python export_simple.py
```

This creates `spiking_step.onnx` which contains the neural network dynamics.

### 2. Build the Web App

```bash
npm install
npm run build
```

The built app will be in the `dist/` directory.

### 3. Deploy to GitHub Pages

#### Option A: Automatic Deployment (Recommended)

1. Push this repository to GitHub
2. Enable GitHub Pages in your repository settings
3. The included GitHub Actions workflow will automatically build and deploy

#### Option B: Manual Deployment

1. Copy the contents of `dist/` to your GitHub Pages repository
2. Ensure the ONNX model file is accessible at `./spiking_step.onnx`

## Architecture

### Neural Network Model

The web app expects an ONNX model with the following interface:

**Inputs:**
- `V0`: Membrane potential state [1,1,H,W]
- `S0`: Spike state [1,1,H,W]
- `U`: External stimulation field [1,1,H,W]
- `decay`, `thr`, `reset`, `input_split`: Network parameters (scalars)
- `exc_local`, `exc_global`, `inh_local`, `inh_global`: Connectivity scales (scalars)

**Outputs:**
- `V1`: Updated membrane potential [1,1,H,W]
- `S1`: Updated spike state [1,1,H,W]
- `Y`: Visualization tensor [1,1,H,W]

### Web Implementation

- **React + Vite**: Modern web app framework
- **ONNX Runtime Web**: Runs neural network inference in browser
- **WebGPU**: Hardware acceleration for neural computations
- **Tailwind CSS**: Responsive UI styling

## Controls

### Neural Parameters
- **Decay**: Membrane potential decay rate (0.8-0.999)
- **Threshold**: Spike firing threshold (0.5-0.999)
- **Reset**: Post-spike reset voltage (-1.0-0.0)
- **Input Split**: How much external input affects neurons (0.0-1.0)
- **Excitatory/Inhibitory Scales**: Strength of local/global connections

### Mouse Interaction
- **Brush Mode**: Choose excitation or inhibition
- **Brush Radius**: Size of stimulation area (1-32 pixels)
- **Brush Strength**: Intensity of stimulation (0.01-1.0)

## Customization

### Modifying the Neural Model

To use your own spiking neural network:

1. Modify `export_simple.py` to export your PyTorch model
2. Ensure the ONNX interface matches the expected inputs/outputs
3. Rebuild with `npm run build`

### Styling and UI

- Edit components in `src/components/ui/` to modify appearance
- Adjust parameters and ranges in `src/App.jsx`
- Modify `src/index.css` for global styling

## Browser Requirements

- Modern browser with WebGPU support (Chrome 113+, Edge 113+)
- Sufficient GPU memory for 256x256 neural field
- JavaScript enabled

## Troubleshooting

### WebGPU Not Available
- Ensure browser supports WebGPU
- Try enabling experimental web platform features in browser flags
- Falls back to WASM backend if WebGPU unavailable

### ONNX Model Issues
- Verify model file is accessible at correct path
- Check browser console for loading errors
- Ensure model interface matches expected inputs/outputs

### Performance Issues
- Reduce neural field size by modifying export script
- Lower browser zoom level for better GPU performance
- Close other GPU-intensive browser tabs

## Development

```bash
npm run dev        # Start development server
npm run build      # Build for production
npm run preview    # Preview production build
```

## License

This project demonstrates spiking neural network visualization and is intended for educational and research purposes.