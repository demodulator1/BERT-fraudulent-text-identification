import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    return

set_seed(42)

def load_data(file_path):
    """加载数据集"""
    df = pd.read_csv(file_path)
    assert 'label' in df.columns, "数据集中缺少'label'列"
    assert 'content_segmented' in df.columns, "数据集中缺少'content_segmented'列"
    return df

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test, use_grid_search=True):
    """训练和评估模型"""
    if use_grid_search:
        # 定义参数网格
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto', 0.1, 1],
        }
        
        # 使用网格搜索找到最佳参数
        print("开始网格搜索最佳参数...")
        svm = SVC(random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"最佳参数: {grid_search.best_params_}")
        model = grid_search.best_estimator_
    else:
        # 使用默认参数
        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_train, y_train)
    
    # 在验证集上评估
    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_report = classification_report(y_val, val_pred)
    
    print("\n验证集结果:")
    print(f"验证集准确率: {val_accuracy:.4f}")
    print(val_report)
    
    # 在测试集上评估
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_report = classification_report(y_test, test_pred)
    
    print("\n测试集结果:")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(test_report)
    
    return model, val_accuracy, test_accuracy

def plot_results(val_accuracy, test_accuracy, save_path):
    """绘制结果图表"""
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号
    
    accuracies = [val_accuracy, test_accuracy]
    labels = ['验证集', '测试集']
    
    plt.bar(labels, accuracies)
    plt.title('TF-IDF + SVM 模型性能', fontsize=12)
    plt.ylabel('准确率', fontsize=10)
    plt.ylim(0, 1)
    
    # 在柱状图上添加具体数值
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.savefig(save_path)
    plt.close()
    print(f"结果图表已保存至: {save_path}")

def main():
    # 加载数据
    print("加载数据...")
    df = load_data('./combined_shuffled.csv')
    
    # 划分数据集 (40% 训练集, 20% 验证集, 40% 测试集)
    print("划分数据集...")
    train_val_df, test_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['label'])
    
    # 特征提取
    print("TF-IDF特征提取...")
    tfidf = TfidfVectorizer(
        max_features=5,  # 最多使用5000个特征
        min_df=5,          # 至少出现5次的词才会被考虑
        max_df=0.8,        # 出现在超过80%文档中的词会被忽略
    )
    
    # 拟合和转换训练集
    X_train = tfidf.fit_transform(train_df['content_segmented'])
    y_train = train_df['label']
    
    # 转换验证集和测试集
    X_val = tfidf.transform(val_df['content_segmented'])
    y_val = val_df['label']
    X_test = tfidf.transform(test_df['content_segmented'])
    y_test = test_df['label']
    
    # 训练和评估模型
    print("开始训练模型...")
    model, val_accuracy, test_accuracy = train_and_evaluate(
        X_train, X_val, X_test, 
        y_train, y_val, y_test,
        use_grid_search=True  # 使用网格搜索找到最佳参数
    )

if __name__ == "__main__":
    main()