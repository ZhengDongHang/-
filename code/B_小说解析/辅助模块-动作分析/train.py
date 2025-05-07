import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import Adam
import numpy as np
from transformers import BertTokenizer

# —— 配置 ——
CSV_PATH = "../../U_工具库/处理后数据/动作数据集.csv"
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 保存路径配置 ======
MODEL_SAVE_PATH = "Model_Weight"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# —— 数据加载 ——

# 加载 CSV 数据
df = pd.read_csv(CSV_PATH)


# 预处理
class ActionDataset(Dataset):
    def __init__(self, texts, actions, tokenizer, max_len=50):
        self.texts = texts
        self.actions = actions
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        action = self.actions[idx]

        # 编码文本和动作
        text_encoded = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len,
                                      return_tensors='pt')
        action_encoded = self.tokenizer(action, padding='max_length', truncation=True, max_length=self.max_len,
                                        return_tensors='pt')

        return {
            "input_ids": text_encoded['input_ids'].squeeze(0),
            "attention_mask": text_encoded['attention_mask'].squeeze(0),
            "labels": action_encoded['input_ids'].squeeze(0)
        }


# 分割数据集为训练和验证集
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)


tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')

# 清洗动作文本：提取括号内内容
def clean_action(text):
    match = re.search(r"[（(](.*?)[)）]", text)
    return match.group(1).strip() if match else text.strip()

# 清洗 action 列
train_df['action'] = train_df['action'].astype(str).apply(clean_action)
val_df['action'] = val_df['action'].astype(str).apply(clean_action)

# 构建数据集和 DataLoader
train_dataset = ActionDataset(train_df['text'].values, train_df['action'].values, tokenizer)
val_dataset = ActionDataset(val_df['text'].values, val_df['action'].values, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# —— Seq2Seq 模型 ——

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask, labels=None):
        # 编码器部分
        embedded = self.embedding(input_ids)
        _, (hidden, cell) = self.encoder(embedded)

        # 解码器部分
        batch_size, seq_len = input_ids.shape # decoder_input 初始化为 embedding 的维度
        decoder_input = torch.zeros(batch_size, seq_len, self.embedding.embedding_dim).to(input_ids.device)
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))

        # 输出预测
        logits = self.fc(decoder_output)

        return logits


# —— 训练过程 ——

def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        optimizer.zero_grad()
        output = model(input_ids, attention_mask, labels=labels)

        # 计算损失
        loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_epoch(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            output = model(input_ids, attention_mask, labels=labels)

            loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
            total_loss += loss.item()

    return total_loss / len(data_loader)


# —— 主训练过程 ——

def train_model():
    model = Seq2Seq(vocab_size=len(tokenizer), hidden_size=256).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = eval_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/seq2seq_model.pth")
    print("模型已保存。")


if __name__ == "__main__":
    train_model()
