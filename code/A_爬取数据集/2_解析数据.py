#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import time
import random
import re
import requests
import atexit
from multiprocessing import Pool, cpu_count
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from config import candidate_headers

# 只匹配 /book/数字/数字.html 的章节链接
CHAP_PATTERN = re.compile(r"^/book/\d+/\d+\.html$")
MAX_RETRIES  = 3
LOAD_DELAY   = 0   # Selenium 执行 JS 跳转大约需要 1 秒

# 计算脚本目录和资源路径
script_dir  = os.path.dirname(os.path.abspath(__file__))
chromedir   = os.path.join(script_dir, "chromedriver-win64")
csv_path    = os.path.join(script_dir, "novel_data", "novel_rankings.csv")
base_folder = os.path.join(script_dir, "novel_data")

# 确保输出文件夹存在
os.makedirs(base_folder, exist_ok=True)

# 准备 chromedriver 可执行文件路径
if not os.path.isdir(chromedir):
    raise FileNotFoundError(f"找不到 chromedriver 文件夹：{chromedir}")
exe_path = os.path.join(chromedir, "chromedriver.exe")
if not os.path.isfile(exe_path):
    exes = [f for f in os.listdir(chromedir) if f.lower().endswith(".exe")]
    if exes:
        exe_path = os.path.join(chromedir, exes[0])
    else:
        raise FileNotFoundError(f"chromedriver 目录下没有 .exe 文件：{chromedir}")

# 配置 Chrome 无头选项
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
service = Service(executable_path=exe_path)

# 全局变量，在子进程里初始化
session = None
driver  = None

def sanitize_filename(name: str) -> str:
    """去除文件名/文件夹名中不安全字符"""
    return "".join(c for c in name if c not in r'\/:*?"<>|').strip()

def init_worker():
    """Pool 初始化：为每个子进程创建独立的 Session 和 WebDriver"""
    global session, driver
    # 初始化 requests.Session 并绕过 JS 验证
    session = requests.Session()
    session.cookies.set("getsite", "bie5.cc", domain="www.bqgui.cc", path="/")
    session.get("https://www.bqgui.cc/userverify",
                headers=random.choice(candidate_headers),
                timeout=10)

    # 初始化 Selenium WebDriver
    driver = webdriver.Chrome(service=service, options=chrome_options)
    # 确保进程退出时关闭 driver
    atexit.register(driver.quit)

def fetch_and_save_chapters(params):
    """Worker 函数：爬取单本小说的所有章节"""
    title, url = params
    novel_dir = os.path.join(base_folder, sanitize_filename(title))
    os.makedirs(novel_dir, exist_ok=True)
    print(f"-> 处理《{title}》：{url}")

    # 1. 用 requests 抓目录
    resp = session.get(url,
                       headers=random.choice(candidate_headers),
                       timeout=10)
    if resp.status_code != 200:
        print(f"   目录页失败：{resp.status_code}")
        return

    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")

    # 2. 拿章节列表
    listmain = soup.select_one("div.listmain, div#listmain")
    if not listmain:
        print("   ⚠ 未找到章节列表")
        return
    dl = listmain.find("dl")
    if not dl:
        print("   ⚠ listmain 里无 <dl>")
        return

    # 3. 遍历所有合法 <dd>
    for dd in dl.find_all("dd"):
        a = dd.find("a")
        if not a:
            continue
        href = a.get("href", "").strip()
        if not CHAP_PATTERN.match(href):
            continue

        chap_title = a.get_text(strip=True)
        chap_url   = urljoin(url, href)
        filename   = sanitize_filename(chap_title) + ".txt"
        out_path   = os.path.join(novel_dir, filename)
        if os.path.exists(out_path):
            print(f"   已有：{filename}")
            continue

        # 4. 用 Selenium 抓章节并保存
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                driver.get(chap_url)
                # 等待 JS 跳转 / 内容加载
                time.sleep(LOAD_DELAY)

                page_source = driver.page_source
                chap_soup   = BeautifulSoup(page_source, "html.parser")
                content_div = chap_soup.find("div", id="chaptercontent")
                if not content_div:
                    print(f"   未找到内容，重试 {attempt}/{MAX_RETRIES}")
                    time.sleep(1)
                    continue

                text = content_div.get_text("\n", strip=True)
                with open(out_path, "w", encoding="utf-8") as fw:
                    fw.write(text)
                print(f"   保存：{filename}")
                break

            except Exception as e:
                print(f"   异常 {e}，重试 {attempt}/{MAX_RETRIES}")
                time.sleep(1)
        else:
            print(f"   失败超过 {MAX_RETRIES} 次，跳过章节：{chap_url}")

def main():
    # 读取 CSV，构建任务列表
    tasks = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            tasks.append((row[1], row[2]))  # (小说标题, 链接)

    # 根据 CPU 数量或固定 4 个进程
    num_proc = min(4, cpu_count())

    # 启动进程池
    with Pool(processes=num_proc, initializer=init_worker) as pool:
        pool.map(fetch_and_save_chapters, tasks)

if __name__ == "__main__":
    main()
