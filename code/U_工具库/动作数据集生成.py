import os
import csv
import json
import random
from multiprocessing import Process, Queue, current_process
from openai import OpenAI

# ———— 配置 ————
API_KEY = 'sk-XXX' # 替换为你自己的API密钥（Deepseek官网获取）
BASE_URL           = "https://api.deepseek.com/v1"
OUTPUT_CSV         = "处理后数据/动作数据集.csv"
NUM_WORKERS        = 10
DOMAINS            = ["科幻", "言情", "历史", "生活", "都市", "侦探", "武打", "仙侠", "日常", "玄幻", "轻小说"]
random.shuffle(DOMAINS) # 打乱列表
SAMPLES_PER_DOMAIN = 50  # 每个领域生成多少条示例
TEXT_LENGTHS       = ['10', '20', '50']  # 分别模拟短/中/长句
# —————————————————

def init_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

def generate_text_action(domain: str, client) -> dict:
    """
    调用 DeepSeek，为指定领域生成一条 文本-动作 对应示例。
    使用 JSON Output 确保返回合法 JSON，对空值返回 None，并将 JSON 字符串解析为 dict。
    文本长度在 TEXT_LENGTHS 中随机选择。
    """
    length = random.choice(TEXT_LENGTHS)
    prompt = (
        "你是一个剧本动作生成器，只会输出严格的 JSON 对象。\n"
        "格式如下：\n"
        "{\n"
        '  "text": "示例文本",\n'
        '  "action": "（动作描述）"\n'
        "}\n"
        f"请基于领域“{domain}”生成一条大约{length}字长度的示例：\n"
        "EXAMPLE:\n"
        "{\n"
        '  "text": "他猛地抬起手臂，眼神凌厉。",\n'
        '  "action": "（挥拳击向空气）"\n'
        "}\n"
        "EXAMPLE2:\n"
        "{\n"
        '  "text": "大门猛地被推开，只见是周翔愤怒地进来，并把文件摔到了桌子上。",\n'
        '  "action": "（推门并摔文件）"\n'
        "}\n"
        "请严格按照上面格式返回，并且不要输出其他任何内容。"
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是严格的 JSON 输出助手。"},
            {"role": "user",   "content": prompt},
        ],
        response_format={'type': 'json_object'},  # 强制 JSON 输出
        temperature=1.5,                            # 增加随机性
        top_p=0.97,
        max_tokens=100
    )
    content = resp.choices[0].message.content
    if not content:
        return None
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return None
    if not isinstance(content, dict):
        return None
    return content

def worker(input_q: Queue, output_q: Queue):
    client = init_client()
    while True:
        task = input_q.get()
        if task is None:
            break
        domain, idx = task
        pair = generate_text_action(domain, client)
        if pair is None:
            print(f"[{current_process().name}] 警告：{domain} 示例 #{idx} 返回空，跳过")
            continue
        pair["domain"] = domain
        output_q.put(pair)
        print(f"[{current_process().name}] 完成 {domain} 示例 #{idx}")

def main():
    # 1. 准备任务队列
    tasks = [(d, i) for d in DOMAINS for i in range(1, SAMPLES_PER_DOMAIN + 1)]
    input_q, output_q = Queue(), Queue()

    # 2. 启动 Worker 进程
    workers = []
    for i in range(NUM_WORKERS):
        p = Process(target=worker, args=(input_q, output_q), name=f"Worker-{i+1}")
        p.start()
        workers.append(p)

    # 3. 写入 CSV 表头
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    if not os.path.exists(OUTPUT_CSV):
        os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
        with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["domain", "text", "action"])
            writer.writeheader()
    # 4. 分发任务
    for t in tasks:
        input_q.put(t)
    for _ in workers:
        input_q.put(None)

    # 5. 实时收集并写入
    # 5. 实时收集并附加写入
    for _ in tasks:
        result = output_q.get()
        if result:
            with open(OUTPUT_CSV, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["domain", "text", "action"])
                writer.writerow({
                    "domain": result["domain"],
                    "text":   result["text"],
                    "action": result["action"]
                })
            print(f"写入: domain={result['domain']} text={result['text']} action={result['action']}")

    # 6. 等待所有 Worker 结束
    for p in workers:
        p.join()

    print(f"✅ 文本-动作数据集已生成并保存到 {OUTPUT_CSV}")

    print(f"✅ 文本-动作数据集已生成，开始打乱顺序...")

    # 7. 打乱 CSV 文件内容
    with open(OUTPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = list(csv.DictReader(f))
        random.shuffle(reader)

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "text", "action"])
        writer.writeheader()
        writer.writerows(reader)

    print(f"✅ 打乱完成，最终数据已保存到 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
