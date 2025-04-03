import chess
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.optim as optim
import time

PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}


class NNUEModel(nn.Module):
    def __init__(self, input_dim=40960, hidden_dim=256):
        super().__init__()
        # 백 / 흑 각각의 SparseLinear
        self.white_embed = nn.EmbeddingBag(input_dim, hidden_dim, mode="sum")
        self.black_embed = nn.EmbeddingBag(input_dim, hidden_dim, mode="sum")

        # MLP (512 → 32 → 32 → 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, white_input, black_input):
        white_vec = self.white_embed(*white_input)  # (B, 256)
        black_vec = self.black_embed(*black_input)  # (B, 256)
        x = torch.cat([white_vec, black_vec], dim=-1)  # (B, 512)
        return self.mlp(x).squeeze(1)  # (B,)


def flatten_indices_and_offsets(batch_indices):
    flat = []
    offsets = [0]
    for indices in batch_indices:
        flat.extend(indices)
        offsets.append(offsets[-1] + len(indices))
    return torch.tensor(flat, dtype=torch.long), torch.tensor(
        offsets[:-1], dtype=torch.long
    )


def halfkp_indices(fen: str) -> tuple[list[int], list[int]]:
    board = chess.Board(fen)
    result = []

    for turn in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(turn)
        if king_sq is None:
            result.append([])
            continue

        indices = []
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None or piece.piece_type == chess.KING:
                continue
            pt_index = PIECE_TO_INDEX.get(piece.piece_type)
            if pt_index is None:
                continue
            color_offset = 0 if piece.color == turn else 1
            piece_index = pt_index * 2 + color_offset
            rel_sq = sq ^ (0 if turn == chess.WHITE else 56)
            index = king_sq * 640 + piece_index * 64 + rel_sq
            indices.append(index)
        result.append(indices)

    return tuple(result)  # (white_indices, black_indices)


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
