import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, AdamW
import matplotlib.pyplot as plt

class GeneticOptimizer:
    def __init__(self, 
                 population_size=10, 
                 generations=5, 
                 crossover_rate=0.8, 
                 mutation_rate=0.2,
                 elite_size=2):
        """
        初始化遗传算法优化器
        
        参数:
            population_size: 种群大小
            generations: 迭代代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            elite_size: 精英个体数量
        """
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.best_params = None
        self.best_fitness = 0
        self.fitness_history = []
        
    def initialize_population(self):
        """初始化种群，每个个体包含BERT模型的关键参数"""
        population = []
        
        for _ in range(self.population_size):
            # 为每个个体随机生成参数
            individual = {
                'learning_rate': 10 ** random.uniform(-5, -3),  # 学习率范围: 1e-5 到 1e-3
                'batch_size': random.choice([8, 16, 32, 64]),   # 批量大小
                'dropout_rate': random.uniform(0.1, 0.5),       # Dropout率
                'max_len': random.choice([64, 128, 256]),       # 最大序列长度
                'fusion_layers': random.randint(1, 4)           # 特征融合层数
            }
            population.append(individual)
            
        return population
    
    def fitness_function(self, individual, model_class, train_df, val_df, device):
        """评估个体适应度（模型验证准确率）"""
        # 提取参数
        learning_rate = individual['learning_rate']
        batch_size = individual['batch_size']
        dropout_rate = individual['dropout_rate']
        max_len = individual['max_len']
        fusion_layers = individual['fusion_layers']
        
        try:
            # 准备数据集
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            
            # 创建训练集
            train_dataset = model_class.FraudDataset(
                texts=train_df['content_segmented'].values,
                labels=train_df['label'].values,
                tokenizer=tokenizer,
                max_len=max_len
            )
            
            # 创建验证集
            val_dataset = model_class.FraudDataset(
                texts=val_df['content_segmented'].values,
                labels=val_df['label'].values,
                tokenizer=tokenizer,
                max_len=max_len
            )
            
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # 初始化模型
            model = model_class.BertClassifier(
                'bert-base-chinese', 
                num_classes=2, 
                dropout_rate=dropout_rate,
                fusion_layers=fusion_layers
            ).to(device)
            
            # 定义优化器和损失函数
            optimizer = AdamW(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # 快速训练（只训练1个epoch来评估参数）
            model.train()
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # 在验证集上评估
            val_accuracy, _ = model_class.evaluate_model(model, val_loader)
            
            return val_accuracy
            
        except Exception as e:
            print(f"评估个体时出错: {e}")
            return 0.0
    
    def select_parents(self, population, fitness_scores):
        """使用轮盘赌选择父代个体"""
        # 计算选择概率
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            selection_probs = [1/len(fitness_scores)] * len(fitness_scores)
        else:
            selection_probs = [f/total_fitness for f in fitness_scores]
        
        # 选择两个父代
        parents_indices = np.random.choice(
            len(population), 
            size=2, 
            p=selection_probs, 
            replace=False
        )
        
        return [population[i] for i in parents_indices]
    
    def crossover(self, parent1, parent2):
        """交叉操作生成子代"""
        if random.random() < self.crossover_rate:
            child = {}
            # 对每个参数随机选择父代
            for key in parent1.keys():
                if random.random() < 0.5:
                    child[key] = parent1[key]
                else:
                    child[key] = parent2[key]
            return child
        else:
            # 不进行交叉，随机返回一个父代
            return parent1.copy() if random.random() < 0.5 else parent2.copy()
    
    def mutate(self, individual):
        """变异操作"""
        mutated = individual.copy()
        
        for key in mutated.keys():
            # 以mutation_rate的概率对每个参数进行变异
            if random.random() < self.mutation_rate:
                if key == 'learning_rate':
                    mutated[key] = 10 ** random.uniform(-5, -3)
                elif key == 'batch_size':
                    mutated[key] = random.choice([8, 16, 32, 64])
                elif key == 'dropout_rate':
                    mutated[key] = random.uniform(0.1, 0.5)
                elif key == 'max_len':
                    mutated[key] = random.choice([64, 128, 256])
                elif key == 'fusion_layers':
                    mutated[key] = random.randint(1, 4)
        
        return mutated
    
    def optimize(self, model_class, train_df, val_df, device):
        """运行遗传算法优化过程"""
        print("开始遗传算法优化...")
        
        # 初始化种群
        population = self.initialize_population()
        
        for generation in range(self.generations):
            print(f"第 {generation+1}/{self.generations} 代")
            
            # 评估适应度
            fitness_scores = []
            for i, individual in enumerate(population):
                print(f"  评估个体 {i+1}/{self.population_size}...")
                fitness = self.fitness_function(individual, model_class, train_df, val_df, device)
                fitness_scores.append(fitness)
                print(f"  个体参数: {individual}, 适应度: {fitness:.4f}")
            
            # 记录最佳适应度
            max_fitness = max(fitness_scores)
            self.fitness_history.append(max_fitness)
            
            # 更新全局最佳参数
            best_idx = fitness_scores.index(max_fitness)
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                self.best_params = population[best_idx].copy()
                print(f"  发现新的最佳参数: {self.best_params}, 适应度: {self.best_fitness:.4f}")
            
            # 如果是最后一代，则结束
            if generation == self.generations - 1:
                break
            
            # 精英保留
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            new_population = [population[i].copy() for i in elite_indices]
            
            # 生成新一代
            while len(new_population) < self.population_size:
                # 选择父代
                parents = self.select_parents(population, fitness_scores)
                
                # 交叉
                child = self.crossover(parents[0], parents[1])
                
                # 变异
                child = self.mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        print(f"优化完成! 最佳参数: {self.best_params}, 适应度: {self.best_fitness:.4f}")
        return self.best_params
    
    def plot_fitness_history(self, save_path):
        """绘制适应度历史曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.fitness_history) + 1), self.fitness_history, marker='o')
        plt.title('遗传算法优化过程')
        plt.xlabel('代数')
        plt.ylabel('最佳适应度')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"适应度历史曲线已保存至: {save_path}")