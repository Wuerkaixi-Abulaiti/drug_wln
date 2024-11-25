import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
from datetime import datetime
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class MolecularLearning:
    def __init__(self, config=None):
        """
        初始化分子学习系统
        
        Args:
            config (dict): 配置参数字典
        """
        self.config = config or {
            'base_dir': '/root/autodl-tmp/wln',
            'batch_size': 32,
            'num_epochs': 10,
            'learning_rate': 0.001,
            'hidden_channels': 768,
            'seq_length': 20,
            'num_layers': 3,
            'dropout': 0.2,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'seed': 42
        }
        
        # 设置目录
        self.base_dir = self.config['base_dir']
        self._setup_directories()
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化组件
        self.model = None
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.scheduler = None
        self.writer = None
        self.criterion = None
        
    def _setup_directories(self):
        """设置必要的目录结构"""
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.checkpoints_dir = os.path.join(self.base_dir, 'checkpoints')
        self.results_dir = os.path.join(self.base_dir, 'evaluation_results')
        self.runs_dir = os.path.join(self.base_dir, 'runs')
        
        # 创建目录
        for dir_path in [self.data_dir, self.checkpoints_dir, 
                        self.results_dir, self.runs_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def setup_data(self, csv_file, chunk_size=100):
        """设置数据集和数据加载器"""
        from data import MolecularGraphDataset
        
        self.dataset = MolecularGraphDataset(
            root=self.data_dir,
            csv_file=csv_file,
            chunk_size=chunk_size
        )
        
        # 划分数据集
        train_dataset, val_dataset, test_dataset = self.dataset.split_dataset(
            train_ratio=self.config['train_ratio'],
            val_ratio=self.config['val_ratio'],
            test_ratio=self.config['test_ratio'],
            seed=self.config['seed']
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        print(f"\nDataset sizes:")
        print(f"Train: {len(train_dataset)}")
        print(f"Validation: {len(val_dataset)}")
        print(f"Test: {len(test_dataset)}")
        
    def setup_model(self):
        """设置模型、优化器和损失函数"""
        from models import WLN
        from train import ReconstructionLoss
        
        # 创建模型
        self.model = WLN(
            in_channels=self.dataset[0].x.size(1),
            hidden_channels=self.config['hidden_channels'],
            seq_length=self.config['seq_length'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # 设置优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # 设置学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 设置损失函数
        self.criterion = ReconstructionLoss()
        
        # 设置tensorboard writer
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = os.path.join(self.runs_dir, current_time)
        self.writer = SummaryWriter(self.log_dir)
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        valid_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", ncols=100)
        
        for batch in pbar:
            try:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                
                output = self.model(batch)
                loss = self.criterion(output, batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                valid_batches += 1
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            except Exception as e:
                print(f"\nError processing batch: {e}")
                continue
                
        return total_loss / valid_batches if valid_batches > 0 else float('inf')
    
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        valid_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validating epoch {epoch+1}", ncols=100)
            for batch in pbar:
                try:
                    batch = batch.to(self.device)
                    output = self.model(batch)
                    loss = self.criterion(output, batch)
                    
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
                    
                except Exception as e:
                    print(f"\nError processing validation batch: {e}")
                    continue
        
        return total_loss / valid_batches if valid_batches > 0 else float('inf')
    
    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        """保存模型检查点"""
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        checkpoint_dir = os.path.join(self.checkpoints_dir, current_time)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        if is_best:
            path = os.path.join(checkpoint_dir, 'best_model.pt')
        else:
            path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            
        torch.save(checkpoint, path)
    
    def train(self):
        """训练模型"""
        print("\n=== Starting Training ===")
        print(f"Total epochs: {self.config['num_epochs']}")
        print("=" * 50)
        
        best_val_loss = float('inf')
        
        try:
            for epoch in range(self.config['num_epochs']):
                train_loss = self.train_epoch(epoch)
                print(f"\nEpoch {epoch+1}/{self.config['num_epochs']} - Train Loss: {train_loss:.4f}")
                
                val_loss = self.validate(epoch)
                print(f"Validation Loss: {val_loss:.4f}")
                
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, train_loss, val_loss, is_best=True)
                    print(f">>> New best model saved (val_loss: {val_loss:.4f})")
                else:
                    self.save_checkpoint(epoch, train_loss, val_loss, is_best=False)
                
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
                    
                print("-" * 50)
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        print("\n=== Training Completed ===")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def evaluate(self):
        """评估模型"""
        # 寻找最新的模型检查点
        subdirs = sorted([d for d in os.listdir(self.checkpoints_dir) 
                         if os.path.isdir(os.path.join(self.checkpoints_dir, d))])
        if not subdirs:
            raise FileNotFoundError("No checkpoint directories found!")
        
        latest_dir = subdirs[-1]
        model_path = os.path.join(self.checkpoints_dir, latest_dir, 'best_model.pt')
        
        # 加载模型
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
        
        # 评估
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        print("\nEvaluating model on test set...")
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = batch.to(self.device)
                output = self.model(batch)
                loss = self.criterion(output, batch)
                total_loss += loss.item()
                batch_count += 1
        
        avg_test_loss = total_loss / batch_count
        print(f"\nTest Loss: {avg_test_loss:.4f}")
        return avg_test_loss
    
    def get_embeddings(self):
        """获取分子嵌入向量"""
        self.model.eval()
        embeddings = []
        
        print("\nGenerating embeddings...")
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Processing"):
                batch = batch.to(self.device)
                z = self.model.encode(batch)['pooler_output']
                embeddings.append(z.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        return embeddings
    
    def visualize_embeddings(self, embeddings, method='tsne'):
        """可视化嵌入向量"""
        print(f"\nVisualizing embeddings using {method.upper()}...")
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method == 'pca':
            reducer = PCA(n_components=2)
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced_embeddings[:, 0], 
            reduced_embeddings[:, 1],
            c=range(len(reduced_embeddings)),
            cmap='viridis',
            alpha=0.6
        )
        plt.colorbar(scatter, label='Molecule Index')
        plt.title(f'Molecule Embeddings ({method.upper()}) - Test Set')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        
        save_path = os.path.join(self.results_dir, f'embeddings_{method}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        plt.close()
    
    def analyze_embeddings(self, embeddings):
        """分析嵌入空间"""
        print("\nAnalyzing embedding space...")
        
        # 计算基本统计量
        norms = np.linalg.norm(embeddings, axis=1)
        normalized_embeddings = embeddings / norms[:, np.newaxis]
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        # PCA分析
        pca = PCA()
        pca.fit(embeddings)
        
        # 可视化相似度矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix[:100, :100],
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1
        )
        plt.title('Embedding Similarity Matrix - Test Set')
        save_path = os.path.join(self.results_dir, 'similarity_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存分析结果
        stats = {
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'mean_similarity': float(np.mean(similarity_matrix)),
            'std_similarity': float(np.std(similarity_matrix)),
            'min_similarity': float(np.min(similarity_matrix)),
            'max_similarity': float(np.max(similarity_matrix)),
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
        }
        
        return stats