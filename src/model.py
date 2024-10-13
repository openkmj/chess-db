import numpy as np
import chess
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models


def fen_to_input(fen: str):
    board = chess.Board(fen + " 0 1")

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


# data set "./data_set.csv"
# fen,total,result

df = pd.read_csv("./data_set.csv")
# csv has no header. So, we need to set header
df.columns = ["fen", "total", "result"]
fens = df["fen"].values
results = df["result"].values

x_data = np.array([fen_to_input(fen) for fen in fens])
y_data = np.array(results)

# test data 1000, train data rest
# x_test = x_data[:1000]
# y_test = y_data[:1000]

# x_train = x_data[1000:]
# y_train = y_data[1000:]

x_train = x_data
y_train = y_data

print(x_train.shape, y_train.shape)
# print(x_train[100])
# print(y_train[100])


def create_model():
    model = models.Sequential()

    # 입력: 8x8 크기의 체스판 상태 + 추가 정보 (19개의 채널)
    model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=(8, 8, 19)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))

    # 출력: 체스판 상태의 평가 (백의 승리 확률 또는 점수)
    model.add(layers.Dense(1, activation="tanh"))  # -1 (흑 승리) ~ 1 (백 승리)

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    return model


model = create_model()
model.fit(x_train, y_train, epochs=50, batch_size=32)

# evaluate and print the result
# loss, acc = model.evaluate(x_test, y_test)
# print("loss=", loss)
# print("acc=", acc)


model.save("../chess_model_v1.keras")
