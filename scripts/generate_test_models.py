import torch
import torch.nn as nn
from torch.onnx import export as export_onnx

# make this script reproducible
torch.manual_seed(1337)

OPSET_VERSION = 14

def export(model: nn.Module, in_shape: list[int], filename: str) -> None:
    export_onnx(
        model, torch.empty(in_shape), filename, opset_version=OPSET_VERSION
    )


# GEMM model
n_inputs = 16
n_outputs = 8
model = nn.Sequential(nn.Linear(n_inputs, n_outputs))
export(model, [1, n_inputs], "../tests/models/gemm.onnx")

# matmul model
n_inputs = 16
n_outputs = 8
model = nn.Sequential(nn.Linear(n_inputs, n_outputs, bias=False))
export(model, [1, n_inputs], "../tests/models/matmul.onnx")


# mul model
class MulModel(nn.Module):
    def __init__(self, n_inputs: int):
        super().__init__()
        self.y = torch.rand(n_inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.y


n_inputs = 16
model = MulModel(n_inputs)
export(model, [1, n_inputs], "../tests/models/mul.onnx")


# add model
class AddModel(nn.Module):
    def __init__(self, n_inputs: int):
        super().__init__()
        self.y = torch.rand(n_inputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.y


n_inputs = 16
model = AddModel(n_inputs)
export(model, [1, n_inputs], "../tests/models/add.onnx")


# power 2 plus power 4 model
class Power2PlusPower4Model(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * x
        return x + x * x


n_inputs = 16
model = Power2PlusPower4Model()
export(model, [1, n_inputs], "../tests/models/power2-plus-power4.onnx")

# conv model
h_inputs = 9
w_inputs = 9
c_inputs = 1
h_kernel = 3
w_kernel = 3
c_outputs = 1
model = nn.Sequential(nn.Conv2d(c_inputs, c_outputs, (h_kernel, w_kernel)))
export(model, [1, c_inputs, h_inputs, w_inputs], "../tests/models/conv.onnx")