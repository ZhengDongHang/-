import os
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

# ====== 配置 ======
MODEL_DIR    = "Model_Weight"
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
INPUT_CSV    = "../mid_output/1_提取后结果.csv"
OUTPUT_CSV   = "../mid_output/2_提取后结果_情绪.csv"
BATCH_SIZE   = 16
THRESHOLD    = 0.5   # 置信度阈值，低于此值标为 NaN
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 加载模型与编码器 ======
tokenizer     = BertTokenizer.from_pretrained(MODEL_DIR)
model         = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE).eval()
label_encoder = joblib.load(ENCODER_PATH)
classes       = label_encoder.classes_

# ====== 读取待评估数据 ======
df    = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
texts = df["text"].fillna("").tolist()

# ====== 批量预测情绪标签 ======
emo_labels = []
with torch.no_grad():
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i: i + BATCH_SIZE]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
        input_ids     = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        outputs = model(input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds   = np.argmax(probs, axis=1)
        max_probs = np.max(probs, axis=1)

        for pred, mp in zip(preds, max_probs):
            if mp >= THRESHOLD:
                emo_labels.append(classes[pred])
            else:
                emo_labels.append(np.nan)

# ====== 写入结果并保存 ======
df["emo_label"] = emo_labels
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"✅ 完成情绪预测，结果已保存到 {OUTPUT_CSV}")