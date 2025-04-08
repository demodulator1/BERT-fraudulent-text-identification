import pandas as pd
import jieba

# 读取CSV文件
df = pd.read_csv('d:\\2025统计建模\\fraud-detection-bert-main\\fraud-detection-bert-main\\label04-last.csv', encoding='gbk')

# 对content列进行分词
df['content_segmented'] = df['content'].apply(lambda x: ' '.join(jieba.cut(x)))

# 保存分词结果到新的CSV文件
df.to_csv('d:\\2025统计建模\\fraud-detection-bert-main\\fraud-detection-bert-main\\label04-last-segmented.csv', index=False, encoding='gbk')

print("分词完成，结果已保存到 label00-last-segmented.csv")