import chess
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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
            print(
                f"sq={sq}, piece={piece}, color={piece.color}, piece_index={piece_index,pt_index,color_offset}, rel_sq={rel_sq}, index={index}"
            )

        result.append(indices)

    return tuple(result)  # (white_indices, black_indices)


def test_halfkp_indices():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    white_indices, black_indices = halfkp_indices(fen)

    print(f"FEN: {fen}")
    print(f"White indices count: {len(white_indices)}")
    print(f"Sample White indices: {white_indices[:10]}")
    print(f"Black indices count: {len(black_indices)}")
    print(f"Sample Black indices: {black_indices[:10]}")
    # EXPECTED OUTPUT:
    # FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
    # White indices count: 30
    # Sample White indices: [2944, 2689, 2818, 3075, 2821, 2694, 2951, 2568, 2569, 2570]
    # Black indices count: 30
    # Sample Black indices: [38904, 38649, 38778, 39035, 38781, 38654, 38911, 38512, 38513, 38514]


class ChessDataset(Dataset):
    def __init__(self, df):
        self.positions = df["position"].tolist()
        self.scores = df["result"].tolist()

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        fen = self.positions[idx]
        score = self.scores[idx]
        white_indices, black_indices = halfkp_indices(fen)
        return white_indices, black_indices, torch.tensor(score, dtype=torch.float32)


def flatten_indices_and_offsets(batch_indices):
    flat = []
    offsets = [0]
    for indices in batch_indices:
        flat.extend(indices)
        offsets.append(offsets[-1] + len(indices))
    return torch.tensor(flat, dtype=torch.long), torch.tensor(
        offsets[:-1], dtype=torch.long
    )


def collate_fn(batch):
    batch = [x for x in batch if len(x[0]) > 0 and len(x[1]) > 0]
    if not batch:
        return (), (), torch.tensor([])
    white_indices, black_indices, scores = zip(*batch)
    white_input = flatten_indices_and_offsets(white_indices)
    black_input = flatten_indices_and_offsets(black_indices)
    return white_input, black_input, torch.stack(scores)


test_halfkp_indices()
