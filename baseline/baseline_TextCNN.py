import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from collections import Counter
import jieba
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class TextCNNDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        words = jieba.lcut(text)
        
        # 将文本转换为索引序列
        indexed = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        
        # 截断或填充
        if len(indexed) > self.max_len:
            indexed = indexed[:self.max_len]
        else:
            indexed += [self.vocab['<PAD>']] * (self.max_len - len(indexed))
        
        return {
            'text': torch.tensor(indexed, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                     out_channels=n_filters,
                     kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]
        
        return self.fc(cat)

def build_vocab(texts, max_vocab_size=50000):
    # 构建词汇表
    word_counts = Counter()
    for text in texts:
        words = jieba.lcut(str(text))
        word_counts.update(words)
    
    # 选择最常见的词
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    
    return vocab

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=3):
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            predictions = model(text)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证
        val_accuracy, val_report = evaluate_model(model, val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"训练损失: {total_loss/len(train_loader):.4f}")
        print(f"验证准确率: {val_accuracy:.4f}")
        print(val_report)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_textcnn_model.pt')
            print("保存最佳模型")
        
        print("-" * 50)
    
    return best_accuracy

def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(text)
            _, preds = torch.max(outputs, 1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    accuracy = accuracy_score(actual_labels, predictions)
    report = classification_report(actual_labels, predictions)
    
    return accuracy, report

def main():
    # 加载数据
    print("加载数据...")
    df = pd.read_csv('./combined_shuffled.csv')
    
    # 划分数据集
    print("划分数据集...")
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['label'])
    
    # 构建词汇表
    print("构建词汇表...")
    vocab = build_vocab(train_df['content_segmented'].values)
    
    # 模型参数
    EMBEDDING_DIM = 300
    N_FILTERS = 100
    FILTER_SIZES = [2, 3, 4]
    OUTPUT_DIM = 2
    DROPOUT = 0.5
    BATCH_SIZE = 64
    
    # 创建数据集
    print("创建数据集...")
    train_dataset = TextCNNDataset(
        texts=train_df['content_segmented'].values,
        labels=train_df['label'].values,
        vocab=vocab
    )
    
    val_dataset = TextCNNDataset(
        texts=val_df['content_segmented'].values,
        labels=val_df['label'].values,
        vocab=vocab
    )
    
    test_dataset = TextCNNDataset(
        texts=test_df['content_segmented'].values,
        labels=test_df['label'].values,
        vocab=vocab
    )
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # 初始化模型
    print("初始化模型...")
    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        n_filters=N_FILTERS,
        filter_sizes=FILTER_SIZES,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
        pad_idx=vocab['<PAD>']
    ).to(device)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    print("开始训练...")
    best_accuracy = train_model(model, train_loader, val_loader, optimizer, criterion, epochs=3)
    
    # 加载最佳模型
    print("加载最佳模型...")
    model.load_state_dict(torch.load('best_textcnn_model.pt'))
    
    # 在测试集上评估
    print("在测试集上评估...")
    test_accuracy, test_report = evaluate_model(model, test_loader)
    
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(test_report)

if __name__ == "__main__":
    main()