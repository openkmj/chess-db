from tensorflow.keras import models
import chess
import numpy as np


def fen_to_input(fen: str):
    board = chess.Board(fen)

    # 8x8 체스판 + 추가 정보용으로 19개의 채널로 확장
    board_input = np.zeros((8, 8, 12), dtype=np.float32)

    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        x = chess.square_file(square)  # 파일 0~7
        y = chess.square_rank(square)  # 랭크 0~7

        # 기물 종류와 색상 구분
        piece_index = piece.piece_type - 1  # 폰: 0, 나이트: 1, ... 킹: 5
        channel = piece_index if piece.color == chess.WHITE else piece_index + 6

        board_input[y, x, channel] = 1

    # 추가 정보 저장 (누구의 차례인지, 캐슬링 가능 여부, 앙파상 가능 여부)
    additional_info = np.zeros((8, 8, 7), dtype=np.float32)  # 추가 7개의 채널

    # 1. 차례 정보
    if board.turn == chess.WHITE:
        additional_info[:, :, 0] = 1  # 백의 차례
    else:
        additional_info[:, :, 0] = -1  # 흑의 차례

    # 2. 캐슬링 가능 여부 (백 킹사이드, 백 퀸사이드, 흑 킹사이드, 흑 퀸사이드)
    if board.has_kingside_castling_rights(chess.WHITE):
        additional_info[:, :, 1] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        additional_info[:, :, 2] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        additional_info[:, :, 3] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        additional_info[:, :, 4] = 1

    # 3. 앙파상 가능 여부 (앙파상 가능한 위치가 있을 경우)
    if board.ep_square:
        x_ep = chess.square_file(board.ep_square)
        y_ep = chess.square_rank(board.ep_square)
        additional_info[y_ep, x_ep, 5] = 1  # 앙파상 위치 표시

    # 추가 정보와 체스판 상태를 결합하여 최종 입력 배열 생성
    full_input = np.concatenate([board_input, additional_info], axis=2)

    return full_input


# import ./model.keras

model = models.load_model("chess_model_v1.keras")
print(model.summary())

# evaluate and print the result
while True:
    fen = input("input: ")
    if fen == "exit":
        break

    input_data = fen_to_input(fen)
    input_data = np.expand_dims(input_data, axis=0)

    result = model.predict(input_data)
    print(result)
