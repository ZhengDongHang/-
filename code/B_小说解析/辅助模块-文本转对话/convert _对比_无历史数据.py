import pandas as pd
from openai import OpenAI
import os
import csv

# 配置 DeepSeek API
from ..config import API_KEY
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# 路径配置
input_path = "../mid_output/3_提取后结果_情绪_含动作.csv"
output_path = "../output/对话剧本_结构化数据_不含背景.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 获取已处理的 ID 列表（如果输出文件存在）
processed_ids = set()
if os.path.isfile(output_path):
    with open(output_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_ids.add(str(row["id"]))
else:
    # 写入表头
    with open(output_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'role', 'text', 'window_idx', 'emo_label', 'behaviour', 'dialogue'])


# 读取数据
data = pd.read_csv(input_path)
print(data)
# 新建空列表用于写结果
results = []

# 遍历每一行，逐步处理
for idx, row in data.iterrows():
    row_id = str(row["id"])
    if row_id in processed_ids:
        print(f"⏭️ 跳过已处理的 ID: {row_id}")
        continue
        
    role = str(row["role"])
    text = str(row["text"])
    emo_label = str(row.get("emo_label", ""))
    behaviour = str(row.get("behaviour", ""))

    # 获取前三句背景
    context_texts = data.iloc[max(0, idx - 3):idx]["text"].astype(str).tolist()
    background = "\n".join(context_texts)

    # 构造prompt
    if role == "旁白":
        prompt = f"""你是剧本的旁白，请结合以下背景信息，以旁白的语气和神态描述当前内容：;\n 
                    -----------------------------------------------------
                    当前内容（待转化文本）：{text}"""
    else:
        prompt = f"""你是角色“{role}”，请结合以下情绪与动作，以符合角色语气的方式表达：
                    \n 角色情绪：{emo_label};\n 角色动作：{behaviour};\n\n
                    ------------------------------------------------------------------------
                    当前内容为（待转化文本）：{text}"""

    # 调用 DeepSeek API
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个剧本创作助手，擅长将结构化的角色描述转化为自然对话文本。"},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=1.1,      # 人为空值随机性
            top_p=0.90,

        )
        dialogue = response.choices[0].message.content.strip().replace('\n', '\\n')

        print("当前角色为:", role, end='.')
        print("对话内容为:", dialogue)
        print("描述性文本为:", '(' + text + ')')
        print('*-'*30)
    except Exception as e:
        dialogue = f"(生成失败：{str(e)})"

    # 写入到文件
    with open(output_path, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(list(row) + [dialogue])


print(f"✅ 对话生成完成，保存到：{output_path}")
