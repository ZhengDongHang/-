import os
import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin


def fetch_novel_rankings(url, output_csv):
    # 1. 如果输出目录不存在，就创建
    dirpath = os.path.dirname(output_csv)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    # 2. 发送请求并设置编码
    resp = requests.get(url, timeout=10)
    resp.encoding = resp.apparent_encoding  # 处理中文编码
    soup = BeautifulSoup(resp.text, 'html.parser')

    # 3. 打开 CSV 文件并写入表头
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['类目', '小说标题', '链接', '作者'])

        # 4. 遍历所有 <div class="blocks">
        for block in soup.find_all('div', class_='blocks'):
            h2 = block.find('h2')
            if not h2:
                continue
            category = h2.get_text(strip=True)

            ul = block.find('ul')
            if not ul:
                continue

            for li in ul.find_all('li'):
                a = li.find('a')
                if not a:
                    continue
                title = a.get_text(strip=True)
                link = urljoin(url, a['href'])

                # 作者名：把 li 文本去掉标题，再 lstrip '/'
                text_all = li.get_text(strip=True)
                author = text_all.replace(title, '', 1).lstrip('/').strip()

                writer.writerow([category, title, link, author])

    print(f'已将数据写入 {output_csv}')


if __name__ == '__main__':
    BASE_URL = 'https://www.bqgui.cc/top/'
    # 这里指定一个包含子目录的路径，脚本会自动创建对应目录
    OUTPUT_CSV = 'A_爬取数据/novel_data/novel_rankings.csv'
    fetch_novel_rankings(BASE_URL, OUTPUT_CSV)
