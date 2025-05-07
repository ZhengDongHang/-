import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import joblib  # 用于保存标签编码器

# ====== 参数配置 ======
CSV_PATH = "../../U_工具库/处理后数据/情绪训练数据集（简体）.csv"  # ✅ 替换为你的CSV路径
PRETRAINED_MODEL = 'bert-base-chinese'
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
MAX_LEN = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== 保存路径配置 ======
MODEL_SAVE_PATH = "Model_Weight"
ENCODER_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, "label_encoder.pkl")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ====== 读取CSV数据 ======
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["text", "emotion"])  # 移除缺失行

# 标签编码
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['emotion'])

# 划分训练/验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# ====== 自定义 Dataset 类 ======
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ====== 加载 Tokenizer 和 Dataset ======
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL)

train_dataset = EmotionDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = EmotionDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ====== 加载预训练模型 ======
model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=len(label_encoder.classes_))
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

# ====== 训练阶段 ======
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}', leave=False)
    for batch in loop:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    print(f"[Epoch {epoch+1}] 训练损失: {total_loss / len(train_loader):.4f}")

# ====== 验证阶段 ======
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"✅ 验证准确率：{correct / total:.2%}")

# ====== 保存模型和标签编码器 ======
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
joblib.dump(label_encoder, ENCODER_SAVE_PATH)

print(f"\n✅ 模型已保存到：{MODEL_SAVE_PATH}")
print(f"✅ 标签编码器已保存为：{ENCODER_SAVE_PATH}")

# ====== 输出标签对应关系 ======
print("\n标签映射：")
for i, label in enumerate(label_encoder.classes_):
    print(f"{i}: {label}")
