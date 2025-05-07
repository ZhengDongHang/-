import re
import torch
import pandas as pd
from transformers import BertTokenizer


# 定义Seq2Seq模型结构（Embedding + 双层LSTM编码器/解码器 + 输出层）
class Seq2SeqModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=256, num_layers=2):
        super(Seq2SeqModel, self).__init__()
        # 词嵌入层
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        # 编码器：双层LSTM
        self.encoder = torch.nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        # 解码器：双层LSTM
        self.decoder = torch.nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True)
        # 输出层：将LSTM输出映射到词表大小
        self.fc = torch.nn.Linear(hidden_size, vocab_size)

    def encode(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        outputs, (hidden, cell) = self.encoder(embedded)
        return hidden, cell

    def decode_step(self, input_id, hidden, cell):
        # 单步解码
        # input_id: (batch_size, 1)
        embedded = self.embedding(input_id)  # (batch_size, 1, embed_dim)
        output, (hidden, cell) = self.decoder(embedded, (hidden, cell))
        output = output.squeeze(1)  # (batch_size, hidden_size)
        output = self.fc(output)  # (batch_size, vocab_size)
        return output, hidden, cell


# 准备设备（优先使用GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载中文BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
vocab_size = tokenizer.vocab_size

# 实例化模型并加载训练好的权重
model = Seq2SeqModel(vocab_size=vocab_size, embed_dim=256, hidden_size=256, num_layers=2)
model.to(device)
model_path = "Model_Weight/seq2seq_model.pth"
state_dict = torch.load(model_path, map_location=device)
# 如果保存的是state_dict：
if isinstance(state_dict, dict) and not any(isinstance(v, torch.nn.Module) for v in state_dict.values()):
    model.load_state_dict(state_dict)
else:
    # 如果直接保存的模型，则直接加载
    model = state_dict

model.eval()

# 读取输入CSV
input_csv = "../mid_output/2_提取后结果_情绪.csv"
df = pd.read_csv(input_csv, encoding='utf-8')  # 根据需要调整编码

# 准备输出列
behaviours = []

# 定义验证有效中文输出的正则
# 检查是否有中文字符
chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
# 检查是否仅由空白或中英文括号组成
bracket_pattern = re.compile(r'^[\s\(\)（）]+$')

# 设定最大输出长度（可根据训练时设置）
max_output_len = 50

# 角色“旁白”对应的标记
role_skip = "旁白"

# 开始逐行预测
with torch.no_grad():
    for idx, row in df.iterrows():
        text = str(row['text']).strip()
        role = str(row['role']).strip()

        # 如果角色为“旁白”，直接写入空字符串
        if role == role_skip:
            behaviours.append("")
            continue

        # 如果文本为空，也写空
        if text == "" or text.isspace():
            behaviours.append("")
            continue

        # 文本编码（不添加特殊token），长度限制50
        input_ids = tokenizer.encode(text, add_special_tokens=False, max_length=50, truncation=True)
        if len(input_ids) == 0:
            behaviours.append("")
            continue

        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)  # (1, seq_len)

        # 编码器前向，获取隐藏状态
        hidden, cell = model.encode(input_tensor)

        # 解码过程：使用 [CLS] 作为起始符号
        start_token = tokenizer.cls_token_id
        end_token = tokenizer.sep_token_id
        dec_input = torch.tensor([[start_token]], dtype=torch.long).to(device)

        output_ids = []
        for _ in range(max_output_len):
            output, hidden, cell = model.decode_step(dec_input, hidden, cell)
            # 取概率最高的token
            next_id = output.argmax(dim=1).item()
            # 如果遇到结束符，则停止生成
            if next_id == end_token:
                break
            output_ids.append(next_id)
            # 准备下一步输入
            dec_input = torch.tensor([[next_id]], dtype=torch.long).to(device)

        # 解码输出token为文本（跳过特殊符号）
        if len(output_ids) > 0:
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip().replace(" ", "")
        else:
            output_text = ""

        # 后处理：如果输出为空、仅空格、仅括号或不包含中文字符，则置为空
        if output_text == "" or output_text.isspace():
            behaviours.append("")
        elif bracket_pattern.match(output_text):
            behaviours.append("")
        elif not chinese_pattern.search(output_text):
            behaviours.append("")
        else:
            behaviours.append(output_text)


# 将预测结果写回DataFrame并保存
df['behaviour'] = behaviours
output_csv = "../mid_output/3_提取后结果_情绪_含动作.csv"
df.to_csv(output_csv, index=False, encoding='utf-8')
print(f"预测完成，结果已保存至 {output_csv}")
