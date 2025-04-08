import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import uvicorn
from fastapi import FastAPI

# 加载预训练的BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 从CSV文件读取数据
df = pd.read_csv('d:\\2025统计建模\\fraud-detection-bert-main\\fraud-detection-bert-main\\label04-last.csv', encoding='gbk')

# 提取文本和标签
texts = df['text_column_name'].tolist()  # 替换为实际的文本列名
labels = df['label_column_name'].tolist()  # 替换为实际的标签列名

# 训练集和验证集划分
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# 分词处理
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

class FinancialFraudDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# 创建训练和验证数据集
train_dataset = FinancialFraudDataset(train_encodings, train_labels)
val_dataset = FinancialFraudDataset(val_encodings, val_labels)

# 加载预训练的BERT模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 训练参数设置
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# 定义计算指标函数
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 创建Trainer对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 训练模型
trainer.train()

# 保存模型
model.save_pretrained("fraud_bert_model")
tokenizer.save_pretrained("fraud_bert_model")

# FastAPI部署
app = FastAPI()

@app.post("/predict")
def predict_transaction(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        output = model(**inputs)
        prediction = torch.argmax(output.logits).item()
    return {"text": text, "fraudulent": bool(prediction)}

# 启动FastAPI服务器
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)