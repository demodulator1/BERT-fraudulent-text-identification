import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取CSV文件
file_path = r'd:\2025统计建模\bert_model\combined_shuffled.csv'
df = pd.read_csv(file_path)

# 筛选label为0的行
df_label_0 = df[df['label'] == 0]

# 合并所有content_segmented文本
all_text = ' '.join(df_label_0['content_segmented'].astype(str))

# 统计词频
words = all_text.split()
word_counts = Counter(words)

# 过滤掉一些特殊标记和单个字符
filtered_word_counts = {word: count for word, count in word_counts.items() 
                       if word not in ['DIGIT', 'URL', 'PHONE', 'NAME', 'PLACE', 'CELLPHONE', 'xxx', 'xx','xxxx'] 
                       and len(word) > 1}

# 生成词云
wordcloud = WordCloud(
    font_path='simhei.ttf',  # 使用黑体字体，确保能显示中文
    width=1200,
    height=800,
    background_color='white',
    max_words=200,
    max_font_size=150,
    random_state=42
)

# 从词频生成词云
wordcloud.generate_from_frequencies(filtered_word_counts)

# 绘制词云图
plt.figure(figsize=(15, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()

# 保存词云图
save_path = r'd:\2025统计建模\bert_model\label0_wordcloud.png'
plt.savefig(save_path, dpi=300)
plt.close()

print(f"词云图已保存至: {save_path}")

# 输出前20个高频词
print("\nLabel为0的文本中出现频率最高的20个词:")
for word, count in sorted(filtered_word_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"'{word}': 出现{count}次")