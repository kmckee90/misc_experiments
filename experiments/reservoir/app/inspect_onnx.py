import onnx

# Load and inspect the ONNX model
model = onnx.load("spiking_step_enhanced.onnx")

print("=== ONNX Model Inputs ===")
for inp in model.graph.input:
    print(f"Name: {inp.name}, Type: {inp.type}")

print("\n=== ONNX Model Outputs ===")
for out in model.graph.output:
    print(f"Name: {out.name}, Type: {out.type}")

print("\n=== Model Graph Info ===")
print(f"Number of nodes: {len(model.graph.node)}")