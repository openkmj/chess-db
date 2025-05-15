import torch
import pandas as pd
from nnue.model import *


# 모델 로드
model = NNUEModel()
model.load_state_dict(torch.load("nnue_model.pth"))
model.eval()

# 테스트 데이터 로드
df = pd.read_csv("test_data.csv")

predictions = []
targets = []
bad_rows = 0

for _, row in df.iterrows():
    try:
        fen = row["position"]
        target = float(row["result"])

        white_indices, black_indices = halfkp_indices(fen)
        white_input = flatten_indices_and_offsets([white_indices])
        black_input = flatten_indices_and_offsets([black_indices])

        with torch.no_grad():
            pred = model(white_input, black_input).item()

        predictions.append(pred)
        targets.append(target)

    except Exception as e:
        bad_rows += 1
        continue

# 리스트 → 텐서
pred_tensor = torch.tensor(predictions)
target_tensor = torch.tensor(targets)

# 🎯 MSE
mse = torch.mean((pred_tensor - target_tensor) ** 2).item()
# 🎯 MAE
mae = torch.mean(torch.abs(pred_tensor - target_tensor)).item()

print(f"✅ 테스트 샘플 수: {len(targets)} (에러: {bad_rows})")
print(f"📉 MSE: {mse:.4f}")
print(f"📉 MAE: {mae:.4f}")
