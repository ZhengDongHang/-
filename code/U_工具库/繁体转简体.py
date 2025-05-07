import pandas as pd
import opencc

# 加载数据集
df = pd.read_csv('未处理数据/情绪训练数据集（繁体）.csv')

# 初始化 OpenCC 转换器 (繁体到简体)
converter = opencc.OpenCC('t2s.json')  # t2s.json 是繁体转简体的配置文件

# 将 text 与emotion 列中的繁体字转换为简体字
df['text'] = df['text'].apply(converter.convert)
df['emotion'] = df['emotion'].apply(converter.convert)
# 保存转换后的数据到新文件
df.to_csv('处理后数据/情绪训练数据集（简体）.csv', index=False)
