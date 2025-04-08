import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# 修改导入语句，从torch.optim导入AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from genetic_optimizer import GeneticOptimizer
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，确保结果可复现
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据集
def load_data(file_path):
    df = pd.read_csv(file_path)
    # 确保数据集中有必要的列
    assert 'label' in df.columns, "数据集中缺少'label'列"
    assert 'content_segmented' in df.columns, "数据集中缺少'content_segmented'列"
    
    # 将label列转换为整数类型
    df['label'] = df['label'].astype(int)
    
    return df

# 定义数据集类
class FraudDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 使用BERT tokenizer处理文本
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 定义BERT分类模型
class BertClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout_rate=0.1, fusion_layers=4):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, output_hidden_states=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fusion_layers = fusion_layers
        
        # 特征融合层
        self.fusion_weights = nn.Parameter(torch.ones(fusion_layers) / fusion_layers)
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size)
        
        # 分类层
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask):
        # 获取BERT所有层的输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取最后几层的隐藏状态
        hidden_states = outputs.hidden_states
        
        # 只使用最后fusion_layers层进行融合
        last_hidden_states = hidden_states[-self.fusion_layers:]
        
        # 计算注意力权重 (softmax归一化)
        attn_weights = torch.softmax(self.fusion_weights, dim=0)
        
        # 动态融合不同层的特征
        fused_output = torch.zeros_like(last_hidden_states[0][:, 0])
        for i, hidden_state in enumerate(last_hidden_states):
            # 使用[CLS]标记的表示 (每个序列的第一个token)
            fused_output += attn_weights[i] * hidden_state[:, 0]
        
        # 应用层归一化
        fused_output = self.layer_norm(fused_output)
        
        # Dropout和分类
        x = self.dropout(fused_output)
        logits = self.fc(x)
        return self.softmax(logits)

# 可视化特征融合权重
def visualize_fusion_weights(model, save_path):
    # 获取融合权重
    weights = torch.softmax(model.fusion_weights, dim=0).detach().cpu().numpy()
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(weights) + 1), weights)
    plt.xlabel('BERT层 (从后向前)')
    plt.ylabel('融合权重')
    plt.title('动态特征融合权重分布')
    plt.xticks(range(1, len(weights) + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.savefig(save_path)
    plt.close()
    print(f"融合权重可视化已保存至: {save_path}")

# 训练模型
def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler=None, epochs=3):
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
        
        # 验证
        val_accuracy, val_report = evaluate_model(model, val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"训练损失: {total_loss/len(train_loader):.4f}")
        print(f"验证准确率: {val_accuracy:.4f}")
        print(val_report)
        
        # 保存最佳模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            # 修改为相对路径
            torch.save(model.state_dict(), 'best_model.pt')
            print("保存最佳模型")
        
        print("-" * 50)
    
    return best_accuracy

# 评估模型
def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    accuracy = accuracy_score(actual_labels, predictions)
    report = classification_report(actual_labels, predictions)
    
    return accuracy, report

def main():
    # 加载数据
    print("加载数据...")
    # 使用相对路径
    df = load_data('./combined_shuffled.csv')
    
    # 划分训练集、验证集和测试集
    print("划分数据集...")
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['label'])
    
    # 是否使用遗传算法优化参数
    use_genetic_optimization = True
    
    if use_genetic_optimization:
        print("使用遗传算法优化参数...")
        # 创建遗传算法优化器
        optimizer = GeneticOptimizer(
            population_size=5,  # 减小种群大小以加快优化速度
            generations=3,      # 减小代数以加快优化速度
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=1
        )
        
        # 导入当前模块以便在遗传算法中使用
        import sys
        current_module = sys.modules[__name__]
        
        # 运行优化
        best_params = optimizer.optimize(current_module, train_df, val_df, device)
        
        # 可视化优化过程
        # 修改为相对路径
        optimizer.plot_fitness_history('genetic_optimization.png')
        
        # 使用最佳参数
        learning_rate = best_params['learning_rate']
        batch_size = best_params['batch_size']
        dropout_rate = best_params['dropout_rate']
        max_len = best_params['max_len']
        fusion_layers = best_params['fusion_layers']
        
        print(f"最佳参数: 学习率={learning_rate}, 批量大小={batch_size}, Dropout率={dropout_rate}, "
              f"最大序列长度={max_len}, 融合层数={fusion_layers}")
    else:
        # 使用默认参数
        learning_rate = 2e-5
        batch_size = 16
        dropout_rate = 0.1
        max_len = 128
        fusion_layers = 4
    
    # 加载BERT tokenizer
    print("加载BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = FraudDataset(
        texts=train_df['content_segmented'].values,
        labels=train_df['label'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    val_dataset = FraudDataset(
        texts=val_df['content_segmented'].values,
        labels=val_df['label'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    test_dataset = FraudDataset(
        texts=test_df['content_segmented'].values,
        labels=test_df['label'].values,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 初始化模型
    print("初始化模型...")
    model = BertClassifier(
        'bert-base-chinese', 
        num_classes=2, 
        dropout_rate=dropout_rate,
        fusion_layers=fusion_layers
    ).to(device)
    
    # 定义优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 添加学习率调度器
    total_steps = len(train_loader) * 3  # 3个epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),  # 10%的步数用于预热
        num_training_steps=total_steps
    )
    
    # 训练模型
    print("开始训练...")
    best_accuracy = train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs=3)
    
    # 加载最佳模型
    print("加载最佳模型...")
    # 修改为相对路径
    model.load_state_dict(torch.load('best_model.pt'))
    
    # 在测试集上评估
    print("在测试集上评估...")
    test_accuracy, test_report = evaluate_model(model, test_loader)
    
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(test_report)
    
    # 可视化融合权重
    # 修改为相对路径
    visualize_fusion_weights(model, 'fusion_weights.png')

if __name__ == "__main__":
    main()