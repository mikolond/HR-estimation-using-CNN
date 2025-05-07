from Models.extractor_latent3 import Extractor
import torch

model = Extractor()
example_inputs = (torch.randn(300, 3, 192, 128),)
onnx_program = torch.onnx.export(model, example_inputs, "onnx_extractor.onnx")
onnx_program.optimize()
onnx_program.save("onnx_extractor.onnx")