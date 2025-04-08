import pandas as pd

def remove_stop_words(text, stop_words):
    return ' '.join([word for word in text.split() if word not in stop_words])

def main():
    # 读取停用词列表
    with open('d:\\2025统计建模\\data_process\\baidu_stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = set(f.read().splitlines())

    # 读取CSV文件
    df = pd.read_csv('d:\\2025统计建模\\data_process\\label04-last-segmented.csv', encoding='gbk')

    # 去除停用词
    df['content_segmented'] = df['content_segmented'].apply(lambda x: remove_stop_words(x, stop_words))

    # 保存处理后的数据
    df.to_csv('d:\\2025统计建模\\data_process\\label04-last-segmented-cleaned.csv', index=False, encoding='utf-8')

if __name__ == '__main__':
    main()