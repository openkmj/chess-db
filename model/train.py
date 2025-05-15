import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import torch.optim as optim
import time
from tqdm import tqdm
from nnue.model import *


if __name__ == "__main__":
    df = pd.read_csv("cleaned_train_data.csv")
    dataset = ChessDataset(df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = NNUEModel()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 5

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        model.train()
        for white_input, black_input, score in tqdm(
            dataloader, desc=f"Epoch {epoch+1}"
        ):
            if score.numel() == 0:
                continue
            optimizer.zero_grad()
            output = model(white_input, black_input)
            loss = criterion(output, score)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        elapsed_time = time.time() - start_time
        print(
            f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f} sec"
        )

    torch.save(model.state_dict(), "nnue_model.pth")
    print("✅ 모델 저장 완료: nnue_model.pth")
