import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import os
from tqdm import tqdm
from datetime import datetime

from models import WLN
from data import MolecularGraphDataset

class ReconstructionLoss(nn.Module):
    """重构损失函数"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, output, batch):
        # 计算重构损失
        reconstructed = output['reconstructed']
        original = batch.x
        
        # 确保形状匹配
        if reconstructed.shape != original.shape:
            reconstructed = reconstructed.view(original.shape)
        
        return self.mse(reconstructed, original)

class Trainer:
    def __init__(self, model, train_loader, val_loader=None,
                 optimizer=None, scheduler=None, device='cuda', log_dir='runs'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = ReconstructionLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters())
        self.scheduler = scheduler
        self.device = device
        
        # 设置日志
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = os.path.join(log_dir, current_time)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # 创建checkpoints目录
        self.checkpoint_dir = os.path.join('/root/autodl-tmp/wln/checkpoints', current_time)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"\nModel Architecture:\n{model}")
        print(f"\nTraining logs will be saved to: {self.log_dir}")
        print(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        valid_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", ncols=100)
        
        for batch in pbar:
            try:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                
                # 前向传播
                output = self.model(batch)
                loss = self.criterion(output, batch)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                valid_batches += 1
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
            except Exception as e:
                print(f"\nError processing batch: {e}")
                continue
                
        avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
        return avg_loss
        
    def validate(self, epoch):
        """验证集评估"""
        self.model.eval()
        total_loss = 0
        valid_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validating epoch {epoch+1}", ncols=100)
            for batch in pbar:
                try:
                    batch = batch.to(self.device)
                    
                    # 前向传播
                    output = self.model(batch)
                    loss = self.criterion(output, batch)
                    
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
                    
                except Exception as e:
                    print(f"\nError processing validation batch: {e}")
                    continue
        
        avg_val_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
        return avg_val_loss

    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            
        torch.save(checkpoint, path)
    
    def train(self, epochs):
        print("\n=== Starting Training ===")
        print(f"Total epochs: {epochs}")
        print("=" * 50)
        
        best_val_loss = float('inf')
        
        try:
            for epoch in range(epochs):
                # 训练
                train_loss = self.train_epoch(epoch)
                print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")
                
                # 验证
                if self.val_loader is not None:
                    val_loss = self.validate(epoch)
                    print(f"Validation Loss: {val_loss:.4f}")
                else:
                    val_loss = train_loss
                
                # 记录到tensorboard
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                if self.val_loader is not None:
                    self.writer.add_scalar('Loss/val', val_loss, epoch)
                
                # 保存检查点
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, train_loss, val_loss, is_best=True)
                    print(f">>> New best model saved (val_loss: {val_loss:.4f})")
                else:
                    self.save_checkpoint(epoch, train_loss, val_loss, is_best=False)
                
                # 学习率调整
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)
                    
                print("-" * 50)
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        print("\n=== Training Completed ===")
        print(f"Best validation loss: {best_val_loss:.4f}")

def main():
    # 设置基础目录
    BASE_DIR = '/root/autodl-tmp/wln'
    
    # 参数
    params = {
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001,
        # 'hidden_channels': 128,
        'hidden_channels': 768,
        'seq_length': 20,  # 添加序列长度参数
        # 'latent_channels': 32,
        'num_layers': 3,
        'dropout': 0.2
    }
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    dataset = MolecularGraphDataset(
        root=os.path.join(BASE_DIR, 'data'),
        csv_file=os.path.join(BASE_DIR, 'data/PDT_USPTO_MIT_last2000.csv'),
        chunk_size=100
    )
    
    # 划分数据集
    train_dataset, val_dataset, test_dataset = dataset.split_dataset(
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    # 创建模型
    model = WLN(
        in_channels=dataset[0].x.size(1),
        hidden_channels=params['hidden_channels'],
        seq_length=params['seq_length'],
        # dalatent_channels=params['latent_channels'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=os.path.join(BASE_DIR, 'runs')
    )
    
    # 开始训练
    trainer.train(epochs=params['num_epochs'])

if __name__ == '__main__':
    main()