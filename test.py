import os
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

from models import WLN
from data import MolecularGraphDataset

def find_latest_model():
    """查找最新的模型检查点"""
    checkpoints_dir = 'checkpoints'
    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Checkpoints directory '{checkpoints_dir}' not found!")
        
    subdirs = [d for d in os.listdir(checkpoints_dir) 
              if os.path.isdir(os.path.join(checkpoints_dir, d))]
    
    if not subdirs:
        raise FileNotFoundError("No checkpoint directories found!")
    
    latest_dir = sorted(subdirs)[-1]
    model_path = os.path.join(checkpoints_dir, latest_dir, 'best_model.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No best_model.pt found in {model_path}")
        
    return model_path

def load_best_model(model, model_path):
    """加载最佳模型"""
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path}")
    return model

def evaluate_model(model, test_loader, criterion, device):
    """在测试集上评估模型"""
    model.eval()
    total_loss = 0
    batch_count = 0
    
    print("\nEvaluating model on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            total_loss += loss.item()
            batch_count += 1
    
    avg_test_loss = total_loss / batch_count
    print(f"\nTest Loss: {avg_test_loss:.4f}")
    return avg_test_loss

def get_embeddings(model, loader, device):
    """获取所有分子的嵌入向量"""
    model.eval()
    embeddings = []
    smiles_list = []  # 如果需要跟踪SMILES
    
    print("\nGenerating embeddings...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing"):
            batch = batch.to(device)
            # 获取潜在表示
            z = model.encode(batch)
            embeddings.append(z.cpu().numpy())
            # 如果batch中包含SMILES信息，也可以保存
            if hasattr(batch, 'smiles'):
                smiles_list.extend(batch.smiles)
    
    embeddings = np.vstack(embeddings)
    return embeddings, smiles_list if smiles_list else None

def visualize_embeddings(embeddings, method='tsne', save_path=None):
    """使用不同方法可视化嵌入向量"""
    print(f"\nVisualizing embeddings using {method.upper()}...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    
    # 降维
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                         c=range(len(reduced_embeddings)), cmap='viridis', 
                         alpha=0.6)
    plt.colorbar(scatter, label='Molecule Index')
    plt.title(f'Molecule Embeddings ({method.upper()}) - Test Set')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close()

def analyze_embedding_space(embeddings):
    """分析嵌入空间的统计特性"""
    print("\nAnalyzing embedding space...")
    
    # 计算基本统计量
    mean = np.mean(embeddings, axis=0)
    std = np.std(embeddings, axis=0)
    
    # 计算嵌入向量之间的余弦相似度
    norms = np.linalg.norm(embeddings, axis=1)
    normalized_embeddings = embeddings / norms[:, np.newaxis]
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    # PCA分析
    pca = PCA()
    pca.fit(embeddings)
    
    # 统计信息
    stats = {
        'mean_norm': np.mean(norms),
        'std_norm': np.std(norms),
        'mean_similarity': np.mean(similarity_matrix),
        'std_similarity': np.std(similarity_matrix),
        'min_similarity': np.min(similarity_matrix),
        'max_similarity': np.max(similarity_matrix),
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'dimension_importance': std,
        'active_dimensions': np.sum(std > 0.01),
        'similarity_matrix': similarity_matrix
    }
    
    return stats

def visualize_similarity_matrix(similarity_matrix, save_path=None):
    """可视化相似度矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Embedding Similarity Matrix - Test Set')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Similarity matrix visualization saved to {save_path}")
    plt.close()

def plot_dimension_importance(stats, save_path=None):
    """绘制维度重要性"""
    plt.figure(figsize=(15, 5))
    
    # 绘制PCA解释方差比
    plt.subplot(1, 3, 1)
    plt.plot(np.cumsum(stats['explained_variance_ratio']), 'b-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    
    # 绘制维度标准差
    plt.subplot(1, 3, 2)
    plt.bar(range(len(stats['dimension_importance'])), 
            stats['dimension_importance'])
    plt.xlabel('Dimension')
    plt.ylabel('Standard Deviation')
    plt.title('Dimension Importance')
    plt.grid(True)
    
    # 绘制相似度分布
    plt.subplot(1, 3, 3)
    similarity_values = stats['similarity_matrix'][np.triu_indices_from(
        stats['similarity_matrix'], k=1)]
    plt.hist(similarity_values, bins=50, density=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Similarity Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Analysis plots saved to {save_path}")
    plt.close()

def main():
    # 设置基础目录
    BASE_DIR = '/root/autodl-tmp/wln'
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
    RESULTS_DIR = os.path.join(BASE_DIR, 'evaluation_results')
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # 创建结果目录
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # 加载数据集
        print("\nLoading dataset...")
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
        print(f"Test set size: {len(test_dataset)}")
        
        # 创建测试集数据加载器
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 找到最佳模型
        latest_dir = sorted([d for d in os.listdir(CHECKPOINTS_DIR) 
                         if os.path.isdir(os.path.join(CHECKPOINTS_DIR, d))])[-1]
        best_model_path = os.path.join(CHECKPOINTS_DIR, latest_dir, 'best_model.pt')
        print(f"\nFound best model at: {best_model_path}")
        
        # 创建模型
        model = WLN(
            in_channels=dataset[0].x.size(1),
            hidden_channels=128,
            latent_channels=32,
            num_layers=3,
            dropout=0.2
        ).to(device)
        
        # 加载模型权重
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model (best val_loss: {checkpoint.get('val_loss', checkpoint.get('loss')):.4f})")
        
        # 评估模型
        from train import ReconstructionLoss
        criterion = ReconstructionLoss()
        test_loss = evaluate_model(model, test_loader, criterion, device)
        
        # 获取嵌入向量
        embeddings, smiles_list = get_embeddings(model, test_loader, device)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        # 保存嵌入向量
        np.save(os.path.join(RESULTS_DIR, 'test_embeddings.npy'), embeddings)
        print("Test embeddings saved to 'test_embeddings.npy'")
        
        # 可视化嵌入向量
        print("\nVisualizing embeddings...")
        for method in ['tsne', 'pca']:
            visualize_embeddings(
                embeddings, 
                method=method, 
                save_path=os.path.join(RESULTS_DIR, f'test_embeddings_{method}.png')
            )
        
        # 分析嵌入空间
        stats = analyze_embedding_space(embeddings)
        
        # 打印统计信息
        print("\nEmbedding space statistics (Test Set):")
        for key, value in stats.items():
            if key != 'similarity_matrix':
                if isinstance(value, np.ndarray):
                    if len(value) > 10:
                        print(f"{key}: shape={value.shape}, mean={np.mean(value):.4f}, "
                              f"std={np.std(value):.4f}")
                    else:
                        print(f"{key}: {value}")
                else:
                    print(f"{key}: {value:.4f}")
        
        # 可视化相似度矩阵
        print("\nVisualizing similarity matrix...")
        visualize_similarity_matrix(
            stats['similarity_matrix'][:100, :100],
            save_path=os.path.join(RESULTS_DIR, 'test_similarity_matrix.png')
        )
        
        # 绘制分析图
        print("\nGenerating analysis plots...")
        plot_dimension_importance(
            stats,
            save_path=os.path.join(RESULTS_DIR, 'test_analysis_plots.png')
        )
        
        # 保存分析结果
        results = {
            'test_loss': test_loss,
            'embedding_stats': {k: v for k, v in stats.items() if k != 'similarity_matrix'},
            'model_path': best_model_path,
            'num_test_samples': len(test_dataset)
        }
        
        np.save(os.path.join(RESULTS_DIR, 'test_results.npy'), results)
        
        print(f"\nEvaluation completed! Check '{RESULTS_DIR}' directory for outputs.")
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        raise e

if __name__ == '__main__':
    main()