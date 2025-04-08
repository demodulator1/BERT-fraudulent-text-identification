import pandas as pd
import os

def shuffle_and_combine_files(file_paths, output_file):
    # 初始化一个空的DataFrame
    combined_df = pd.DataFrame()

    # 逐个读取文件并合并
    for file_path in file_paths:
        print(f"正在处理文件: {file_path}")
        
        # 对label00文件特别处理，尝试多种中文编码
        if "label00-last-segmented-cleaned.csv" in file_path:
            encodings_to_try = ['gbk', 'gb18030', 'cp936', 'latin1']
            for encoding in encodings_to_try:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"使用 {encoding} 编码成功读取 label00 文件")
                    break
                except Exception as e:
                    print(f"使用 {encoding} 读取失败: {str(e)}")
            else:
                print(f"所有编码都无法读取文件，跳过此文件")
                continue
        else:
            # 其他文件尝试不同的编码读取
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                print(f"使用 utf-8 编码成功读取")
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='gbk')
                    print(f"使用 gbk 编码成功读取")
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin1')
                    print(f"使用 latin1 编码成功读取")
        
        # 确保所有文本列都是字符串类型
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # 打乱数据顺序
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)

    # 保存到新的CSV文件，使用 utf-8-sig 编码（带BOM）
    combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"已保存合并后的文件: {output_file}")

def main():
    # 文件路径列表
    file_paths = [
        'd:\\2025统计建模\\bert_model\\label00-last-segmented-cleaned.csv',
        'd:\\2025统计建模\\bert_model\\label01-last-segmented-cleaned.csv',
        'd:\\2025统计建模\\bert_model\\label02-last-segmented-cleaned.csv',
        'd:\\2025统计建模\\bert_model\\label03-last-segmented-cleaned.csv',
        'd:\\2025统计建模\\bert_model\\label04-last-segmented-cleaned.csv'
    ]

    # 输出文件路径
    output_file = 'd:\\2025统计建模\\bert_model\\combined_shuffled.csv'

    # 执行合并和打乱
    shuffle_and_combine_files(file_paths, output_file)

if __name__ == '__main__':
    main()