import pandas as pd
import chess


def is_valid_fen(fen: str) -> bool:
    try:
        board = chess.Board(fen)
        return True
    except:
        return False


# CSV 불러오기
df = pd.read_csv("train_data.csv")

# 유효한 FEN만 필터링
df["is_valid"] = df["position"].apply(is_valid_fen)
df_clean = df[df["is_valid"]].drop(columns=["is_valid"])

# 저장
df_clean.to_csv("cleaned_train_data.csv", index=False)
print(f"✅ 전처리 완료: {len(df_clean)}개 FEN만 남음")
