import torch
from nnue.model import NNUEModel, halfkp_indices, flatten_indices_and_offsets

# 예시 입력 (batch size = 1)
white_idx = [123, 456, 789]
black_idx = [1000, 2000, 3000]
white_input = flatten_indices_and_offsets([white_idx])
black_input = flatten_indices_and_offsets([black_idx])

model = NNUEModel()
model.load_state_dict(torch.load("nnue_model.pth"))
model.eval()

# ONNX로 export
torch.onnx.export(
    model,
    args=(white_input, black_input),
    f="capybara_model.onnx",
    input_names=["white_input", "white_offset", "black_input", "black_offset"],
    output_names=["score"],
    dynamic_axes={
        "white_input": {0: "white_len"},
        "white_offset": {0: "batch_size"},
        "black_input": {0: "black_len"},
        "black_offset": {0: "batch_size"},
        "score": {0: "batch_size"},
    },
    opset_version=11,
)
