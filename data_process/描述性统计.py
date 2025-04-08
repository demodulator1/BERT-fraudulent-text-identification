import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import jieba
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取CSV文件
file_path = r'd:\2025统计建模\main\data\label03-last.csv'
df = pd.read_csv(file_path)

# 确保content列存在
if 'content' not in df.columns:
    raise ValueError("CSV文件中没有'content'列")

# 使用jieba分词并查找高频词
def get_top_words(text_series, top_n=10):
    # 加载停用词表（如果有的话）
    stop_words = set()
    try:
        with open(r'd:\2025统计建模\data_process\stopwords.txt', 'r', encoding='utf-8') as f:
            for line in f:
                stop_words.add(line.strip())
    except:
        print("未找到停用词表，将不过滤停用词")
    
    # 对所有文本进行分词
    all_words = []
    for text in text_series:
        words = jieba.cut(text)
        # 过滤停用词和单个字符
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
        all_words.extend(filtered_words)
    
    # 统计词频
    word_counts = Counter(all_words)
    
    # 返回前N个高频词
    return word_counts.most_common(top_n)

# 获取高频词
top_words = get_top_words(df['content'])
print("\n文本中出现频率最高的三个词:")
for word, count in top_words:
    print(f"'{word}': 出现{count}次")
