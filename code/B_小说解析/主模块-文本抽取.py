import os
import json
import re
import pandas as pd
from openai import OpenAI
from multiprocessing import Process, Queue, current_process
from tqdm import tqdm  # 新增进度条
from config import API_KEY

# ———— 配置 ————
BASE_URL    = "https://api.deepseek.com/v1"
INPUT_DIR   = "../A_爬取数据/novel_data/凡人修仙传"
OUTPUT_CSV  = "1_提取后结果.csv"
FILE_NUMBERS = 2 # 读取的章节数
NUM_WORKERS = 5 # 同时的处理数
WINDOW_SIZE = 40   # 每个滑窗的行数
OVERLAP_RATE = 2/3  # 每个窗口与上一个窗口重叠2/3
# —————————————————

def init_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

def extract_turns_from_text(text: str, client) -> list[dict]:
    """
    调用 DeepSeek，从一段小说文本中抽取 [{local_id, role, text}, …]
    local_id 为该滑窗内部自增编号，从 1 开始。
    为提高鲁棒性，要求模型用 ```json ...``` 包裹输出，并严格输出合法 JSON。
    """
    prompt = (
        "你是一个剧本抽取器。\n"
        "请从下面这段小说中提取所有与“角色”相关的文本片段，\n"
        "可能是对白，也可能是场景/背景说明（旁白）。\n"
        "请只输出合法 JSON 列表，并用 ```json ...``` 包裹，格式如下：\n"
        "```json\n"
        "[\n"
        "  {\"id\": 1, \"role\": \"角色名\", \"text\": \"原文内容1\"},\n"
        "  {\"id\": 2, \"role\": \"旁白\", \"text\": \"背景介绍\"}\n"
        "]\n"
        "``` \n"
        "不要输出任何其他内容。\n\n"
        f"小说内容：\n{text}"
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个剧本抽取器。"},
            {"role": "user",   "content": prompt},
        ],
        stream=False
    )
    raw = resp.choices[0].message.content
    # 提取 ```json ... ``` 中的 JSON 部分
    m = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", raw)
    json_str = m.group(1) if m else raw.strip()
    return json.loads(json_str)

def worker(input_queue: Queue, result_queue: Queue):
    client = init_client()
    while True:
        item = input_queue.get()
        if item is None:
            break
        window_idx, window_text = item
        print(f"[{current_process().name}] 处理滑窗 #{window_idx}")

        try:
            turns = extract_turns_from_text(window_text, client)
            for t in turns:
                t["window_idx"] = window_idx
            result_queue.put(turns)
        except Exception as e:
            print(f"[{current_process().name}] 错误 in window {window_idx}: {e}")
            result_queue.put([])

def rewrite_global(all_turns: list[dict]) -> list[dict]:
    # 1. 按 window_idx & local id 排序
    sorted_turns = sorted(all_turns, key=lambda t: (t["window_idx"], t["id"]))
    # 2. 全局去重
    seen = set(); unique = []
    for t in sorted_turns:
        txt = t["text"].strip()
        if txt not in seen:
            seen.add(txt)
            unique.append({"role": t["role"], "text": txt, "window_idx": t["window_idx"]})
    # 3. 重新编号
    for idx, t in enumerate(unique, start=1):
        t["id"] = idx
    return unique

def main_multiprocess_rr():
    # 1. 读取并排序前 NUM_WORKERS 个文件
    def num_key(fname):
        m = re.match(r"^(\d+)", fname)
        return int(m.group(1)) if m else float("inf")

    files = sorted(
        [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".txt")],
        key=num_key
    )[:FILE_NUMBERS]

    # 2. 合并所有行到内存
    all_lines = []
    for fname in files:
        with open(os.path.join(INPUT_DIR, fname), encoding="utf-8") as fr:
            all_lines.extend(fr.readlines())

    # 3. 构造滑窗：每次前进 WINDOW_SIZE*(1-OVERLAP_RATE) 行
    stride = int(WINDOW_SIZE * (1 - OVERLAP_RATE))
    if stride < 1: stride = 1
    tasks = []
    for start in range(0, len(all_lines), stride):
        window_lines = all_lines[start: start + WINDOW_SIZE]
        if not window_lines:
            break
        window_text = "".join(window_lines)
        window_idx  = start // stride + 1
        tasks.append((window_idx, window_text))

    # 4. 启动子进程 & 分发任务
    input_queues = [Queue() for _ in range(NUM_WORKERS)]
    result_queue = Queue()
    workers = [
        Process(target=worker, args=(input_queues[i], result_queue), name=f"Worker-{i+1}")
        for i in range(NUM_WORKERS)
    ]
    for p in workers: p.start()

    import itertools
    rr = itertools.cycle(range(NUM_WORKERS))
    # 分发进度条
    for task in tqdm(tasks, desc="Dispatching windows"):
        input_queues[next(rr)].put(task)
    for q in input_queues:
        q.put(None)

    # 5. 收集结果进度条
    all_turns = []
    for _ in tqdm(range(len(tasks)), desc="Collecting window results"):
        all_turns.extend(result_queue.get())
    for p in workers:
        p.join()

    # 6. Rewriter：全局去重＋重新编号
    final_turns = rewrite_global(all_turns)

    # 7. 保存 CSV
    df = pd.DataFrame(final_turns)[["id", "role", "text", "window_idx"]]
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 完成，结果已保存到 {OUTPUT_CSV}")

if __name__ == "__main__":
    main_multiprocess_rr()
