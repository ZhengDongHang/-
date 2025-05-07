import os
import pandas as pd
from docx import Document
from openai import OpenAI
import json

# ——— 配置 ———
API_KEY = 'sk-XXX' # 替换为你自己的API密钥（Deepseek官网获取）
BASE_URL    = "https://api.deepseek.com/v1"
MODEL       = "deepseek-chat"
INPUT_FILES = {
    "基础版":    "对比剧本/剧本输出_纯基础版.docx",
    "不含背景":  "对比剧本/剧本输出_不含背景.docx",
    "不含情绪动作": "对比剧本/剧本输出_不含情绪动作.docx",
    "完整版":    "剧本输出.docx"
}
ORIGINAL_NOVEL = "对比剧本/原文.docx"
OUTPUT_METRICS = "评估结果.csv"
# ————————

# 初始化 DeepSeek 客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def load_docx(path):
    """将 docx 文件内容拼成一个长字符串（段落之间换行）。"""
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def load_csv_script(path):
    """将 CSV 文件中 dialogue 列拼成长文本。"""
    df = pd.read_csv(path, encoding='utf-8-sig')
    # 假设对话在 Dialogue 或 text 列
    if "Dialogue" in df.columns:
        texts = df["Dialogue"]
    elif "text" in df.columns:
        texts = df["text"]
    else:
        texts = df.iloc[:,0].astype(str)
    return "\n".join(texts.astype(str).tolist())

def evaluate_script(script_text, original_text):
    """
    调用 DeepSeek API，对一部剧本(script_text)与原文(original_text)做评估，
    输出四个指标的分数（0-10）。
    """
    system_prompt = """
你是一个剧本质量评估专家。接下来我会提供两段文本：
- 剧本：模型生成的脚本内容
- 原文：该脚本对应的小说原文

请基于以下四个维度对“剧本”做打分（0-10 分，分差要拉大体现差异）并输出合法 JSON：
1. 人物情感塑造：情感刻画是否生动、有深度
2. 动作流畅感：动作描写前后是否连贯自然
3. 剧本流畅程度：整体语言与情节连贯度
4. 原文相似度：剧本与原文在语义或风格上的贴近度

示例输出格式：
{
  "人物情感塑造": 7.5,
  "动作流畅感": 6.8,
  "剧本流畅程度": 8.2,
  "原文相似度": 5.9
}
"""
    user_prompt = f"""
剧本：
\"\"\"{script_text}\"\"\"

原文：
\"\"\"{original_text}\"\"\"
"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        response_format={'type':'json_object'},
        temperature=1.8,
        max_tokens=5000
    )
    return resp.choices[0].message.content

def main():
    # 载入原文
    original = load_docx(ORIGINAL_NOVEL)

    records = []
    for name, path in INPUT_FILES.items():
        # 根据后缀决定如何加载
        if path.endswith(".docx"):
            script = load_docx(path)
        else:
            script = load_csv_script(path)

        print(f"评估：{name} …")
        try:
            metrics = evaluate_script(script, original)
        except Exception as e:
            metrics = {
                "人物情感塑造": None,
                "动作流畅感": None,
                "剧本流畅程度": None,
                "原文相似度": None,
                "error": str(e)
            }
        # 如果返回是字符串 JSON，解析成 dict
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except json.JSONDecodeError:
                metrics = {"error": "JSON解析失败", "原始返回": metrics}

        row = {"版本": name}
        row.update(metrics)
        records.append(row)

    # 存为 CSV
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_METRICS, index=False, encoding="utf-8-sig")
    print(f"✅ 评估结果已保存至 {OUTPUT_METRICS}")

if __name__ == "__main__":
    main()
