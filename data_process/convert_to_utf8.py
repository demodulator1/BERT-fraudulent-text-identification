import pandas as pd

def convert_to_utf8(file_paths):
    for file_path in file_paths:
        print(f"处理文件: {file_path}")
        # 尝试不同的编码读取文件，并添加低内存模式参数
        encodings = ['gbk', 'utf-8', 'latin1', 'ISO-8859-1']
        
        for encoding in encodings:
            try:
                print(f"尝试使用 {encoding} 编码读取...")
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                print(f"成功使用 {encoding} 编码读取")
                
                # 重新保存为UTF-8编码
                df.to_csv(file_path + '.utf8', index=False, encoding='utf-8')
                print(f"已保存为UTF-8编码: {file_path}.utf8")
                
                # 成功后跳出循环
                break
            except Exception as e:
                print(f"使用 {encoding} 编码读取失败: {str(e)}")
        else:
            print(f"警告: 无法读取文件 {file_path}，已跳过")

def main():
    # 文件路径列表
    file_paths = [
        'd:\\2025统计建模\\bert_model\\label00-last-segmented-cleaned.csv',
        'd:\\2025统计建模\\bert_model\\label01-last-segmented-cleaned.csv',
        'd:\\2025统计建模\\bert_model\\label02-last-segmented-cleaned.csv',
        'd:\\2025统计建模\\bert_model\\label03-last-segmented-cleaned.csv',
        'd:\\2025统计建模\\bert_model\\label04-last-segmented-cleaned.csv'
    ]

    # 执行编码转换
    convert_to_utf8(file_paths)

if __name__ == '__main__':
    main()