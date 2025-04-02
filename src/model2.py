import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import random
import pandas as pd
import chess
import time

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


# 데이터 로딩 및 전처리
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    dataset = []
    for _, row in df.iterrows():
        try:
            indices = fen_to_halfkp_indices(row["position"])
            score = float(row["result"])  # 이미 -1 ~ 1 사이
            dataset.append((indices, score))
        except Exception as e:
            print(f"Skipping row due to error: {e}")
    return dataset


class NNUEChessDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        indices, target = self.data[idx]
        return torch.tensor(indices, dtype=torch.long), torch.tensor(
            target, dtype=torch.float32
        )


class NNUEModel(nn.Module):
    def __init__(self, input_dim=12 * 64 * 64, embedding_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

        self.fc1 = nn.Linear(embedding_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, indices):  # indices: (B, N)
        x = self.embedding(indices)  # (B, N, 256)
        x = x.sum(dim=1)  # (B, 256)
        x = F.relu(self.fc1(x))  # (B, 32)
        x = F.relu(self.fc2(x))  # (B, 32)
        out = self.fc3(x).squeeze(1)  # (B,)
        return out


def custom_collate_fn(batch):
    indices_list, targets = zip(*batch)
    padded_indices = pad_sequence(
        indices_list, batch_first=True, padding_value=0
    )  # (B, max_len)
    targets = torch.stack(targets)
    return padded_indices, targets


data = load_data("train_data.csv")
print(f"데이터 개수: {len(data)}")

dataset = NNUEChessDataset(data)
dataloader = DataLoader(
    dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn
)

model = NNUEModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

EPOCHS = 5

for epoch in range(EPOCHS):
    start_time = time.time()
    total_loss = 0
    for indices_batch, targets in dataloader:
        optimizer.zero_grad()
        preds = model(indices_batch)  # (B, N) -> (B,)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(
        f"Epoch {epoch+1}/{EPOCHS} - Time: {elapsed_time:.2f}s - Loss: {avg_loss:.4f}"
    )

torch.save(model.state_dict(), "nnue_model2.pth")
