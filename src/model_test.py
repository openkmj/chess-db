import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import chess


# --------------------------
# 모델 클래스 정의
# --------------------------
class NNUEModel(nn.Module):
    def __init__(self, input_dim=12 * 64 * 64, embedding_dim=256):  # 🔥 49152
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=embedding_dim
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)
        self.fc1 = nn.Linear(embedding_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, indices):
        x = self.embedding(indices)  # (N, 256)
        x = x.sum(dim=0)  # (256,)
        x = F.relu(self.fc1(x))  # (32,)
        x = F.relu(self.fc2(x))  # (32,)
        out = torch.tanh(self.fc3(x))  # (1,) → [-1, 1]
        return out.squeeze(0)


# --------------------------
# 헬퍼 함수: FEN → HalfKP
# --------------------------
PIECE_TO_INDEX = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}


def fen_to_halfkp_indices(fen: str) -> list[int]:
    board = chess.Board(fen + " 0 0")
    king_sq = board.king(chess.WHITE if board.turn else chess.BLACK)
    indices = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        piece_index = PIECE_TO_INDEX[piece.symbol()]
        idx = king_sq * 12 * 64 + piece_index * 64 + square
        indices.append(idx)

    return indices


# --------------------------
# 테스트셋 로드 및 전처리
# --------------------------
def load_test_data(csv_path: str):
    df = pd.read_csv(csv_path)
    dataset = []
    for _, row in df.iterrows():
        try:
            indices = fen_to_halfkp_indices(row["position"])
            score = float(row["result"])
            dataset.append((indices, score))
        except Exception as e:
            print(f"Skipping test row: {e}")
    return dataset


test_data = load_test_data("test_data.csv")
print(f"✅ 테스트 데이터 수: {len(test_data)}개")

# --------------------------
# 모델 로드 & 평가
# --------------------------
model = NNUEModel(input_dim=12 * 64 * 64)  # 🔥 input_dim = 49152
model.load_state_dict(torch.load("nnue_model2.pth"))
model.eval()

loss_fn = nn.MSELoss()
total_test_loss = 0

with torch.no_grad():
    for indices, target in test_data:
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        pred = model(indices_tensor)
        loss = loss_fn(pred, target_tensor)
        total_test_loss += loss.item()

avg_test_loss = total_test_loss / len(test_data)
print(f"\n📊 테스트셋 평균 Loss (MSE): {avg_test_loss:.4f}")

# --------------------------
# 예측 예시 출력
# --------------------------
print("\n🎯 예측값 vs 실제값 (5개 예시)")
for i in range(5):
    indices, label = test_data[i]
    pred = model(torch.tensor(indices, dtype=torch.long)).item()
    print(f"예측: {pred:.3f}, 실제: {label:.3f}")
