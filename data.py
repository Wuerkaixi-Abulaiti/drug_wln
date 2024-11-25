import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
import rdkit.Chem as Chem
from typing import List, Tuple, Optional
import os
import json
from tqdm import tqdm

# 元素列表和特征维度定义
elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 10

class MolecularGraphDataset(Dataset):
    def __init__(self, root: str, csv_file: str, chunk_size: int = 1000, transform=None):
        self.root = root
        self.raw_file = csv_file
        self.chunk_size = chunk_size
        self._processed_dir = os.path.join(root, 'processed')
        self.transform = transform
        
        os.makedirs(self._processed_dir, exist_ok=True)
        
        if not self._check_processed_files_exist():
            print("Processing raw data...")
            self._process_raw_data(csv_file)
        
        # 加载元数据
        with open(os.path.join(self._processed_dir, 'metadata.json'), 'r') as f:
            self.metadata = json.load(f)
            
        self.total_size = self.metadata['total_size']
        self.num_chunks = self.metadata['num_chunks']
        
        # 当前加载的chunk
        self.current_chunk = None
        self.current_chunk_id = None
        
        # 加载第一个chunk来确定实际大小
        self._load_chunk(0)
        self.chunk_sizes = []
        for i in range(self.num_chunks):
            chunk_file = os.path.join(self._processed_dir, f'chunk_{i}.json')
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
                self.chunk_sizes.append(len(chunk_data))

        # 设置长度
        self._length = sum(self.chunk_sizes)
        self._indices = None

    def _check_processed_files_exist(self):
        """检查处理后的文件是否存在"""
        metadata_file = os.path.join(self._processed_dir, 'metadata.json')
        return os.path.exists(metadata_file)
    
    @staticmethod
    def onek_encoding_unk(x, allowable_set):
        """One-hot encoding with unknown category"""
        if x not in allowable_set:
            x = allowable_set[-1]
        return [int(x == s) for s in allowable_set]

    @staticmethod
    def atom_features(atom) -> np.ndarray:
        """获取原子特征"""
        return np.array(
            list(MolecularGraphDataset.onek_encoding_unk(atom.GetSymbol(), elem_list)) 
            + list(MolecularGraphDataset.onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])) 
            + list(MolecularGraphDataset.onek_encoding_unk(atom.GetExplicitValence(), [1, 2, 3, 4, 5, 6]))
            + list(MolecularGraphDataset.onek_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]))
            + [atom.GetIsAromatic()],
            dtype=np.float32
        )

    @staticmethod
    def bond_features(bond) -> np.ndarray:
        """获取键特征"""
        bt = bond.GetBondType()
        return np.array([
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()
        ], dtype=np.float32)

    def len(self) -> int:
        """返回数据集的总大小"""
        return self._length

    def get(self, idx: int) -> Data:
        """获取指定索引的数据"""
        if idx < 0 or idx >= self.len():
            raise IndexError(f"Index {idx} out of range for dataset of size {self.len()}")
        
        # 找到正确的chunk
        current_pos = 0
        for chunk_id, chunk_size in enumerate(self.chunk_sizes):
            if current_pos + chunk_size > idx:
                self._load_chunk(chunk_id)
                local_idx = idx - current_pos
                break
            current_pos += chunk_size
        else:
            raise IndexError(f"Index {idx} not found in any chunk")
        
        try:
            graph_dict = self.current_chunk[local_idx]
            
            # 转换为PyG的Data对象
            data = Data(
                x=torch.FloatTensor(graph_dict['atom_features']),
                edge_index=torch.LongTensor(graph_dict['edge_index']).t().contiguous(),
                edge_attr=torch.FloatTensor(graph_dict['edge_features'])
            )
            
            if self.transform is not None:
                data = self.transform(data)
                
            return data
        except Exception as e:
            print(f"Error processing index {idx} (local_idx {local_idx} in chunk {chunk_id})")
            raise e

    def smiles_to_graph_dict(self, smiles: str) -> dict:
        """将SMILES转换为图字典格式"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
            
        # 获取原子特征
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_features_list.append(self.atom_features(atom).tolist())
            
        # 构建边索引和特征
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]
            bond_feat = self.bond_features(bond).tolist()
            edge_features += [bond_feat, bond_feat]
            
        return {
            'atom_features': atom_features_list,
            'edge_index': edge_indices,
            'edge_features': edge_features,
            'smiles': smiles
        }

    def _process_raw_data(self, csv_file: str):
        """处理原始数据并保存为chunks"""
        df = pd.read_csv(csv_file)
        smiles_list = df['SMILES'].tolist()
        
        chunk_id = 0
        current_chunk = []
        total_valid = 0
        
        for idx, smiles in enumerate(tqdm(smiles_list, desc="Processing molecules")):
            try:
                graph_dict = self.smiles_to_graph_dict(smiles)
                if graph_dict is not None:
                    current_chunk.append(graph_dict)
                    total_valid += 1
                    
                    if len(current_chunk) >= self.chunk_size:
                        self._save_chunk(current_chunk, chunk_id)
                        chunk_id += 1
                        current_chunk = []
            except Exception as e:
                print(f"Error processing SMILES at index {idx}: {e}")
                continue
                    
        # 保存最后一个chunk
        if current_chunk:
            self._save_chunk(current_chunk, chunk_id)
            chunk_id += 1
            
        # 保存元数据
        metadata = {
            'total_size': total_valid,
            'num_chunks': chunk_id,
            'chunk_size': self.chunk_size
        }
        with open(os.path.join(self._processed_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    def _save_chunk(self, chunk_data: List[dict], chunk_id: int):
        """保存数据块"""
        chunk_file = os.path.join(self._processed_dir, f'chunk_{chunk_id}.json')
        with open(chunk_file, 'w') as f:
            json.dump(chunk_data, f)
            
    def _load_chunk(self, chunk_id: int):
        """加载指定的数据块"""
        if self.current_chunk_id != chunk_id:
            chunk_file = os.path.join(self._processed_dir, f'chunk_{chunk_id}.json')
            with open(chunk_file, 'r') as f:
                self.current_chunk = json.load(f)
            self.current_chunk_id = chunk_id

    def split_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        """划分数据集"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
        
        # 设置随机种子
        torch.manual_seed(seed)
        
        # 生成随机索引
        indices = torch.randperm(self.len()).tolist()
        
        # 计算划分点
        train_size = int(train_ratio * self.len())
        val_size = int(val_ratio * self.len())
        
        # 划分数据集
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # 创建子数据集
        train_dataset = torch.utils.data.Subset(self, train_indices)
        val_dataset = torch.utils.data.Subset(self, val_indices)
        test_dataset = torch.utils.data.Subset(self, test_indices)
        
        return train_dataset, val_dataset, test_dataset

def get_feature_dimensions():
    """返回特征维度信息"""
    return {
        'atom_fdim': atom_fdim,
        'bond_fdim': bond_fdim,
        'max_nb': max_nb
    }

if __name__ == "__main__":
    # 示例用法
    dataset = MolecularGraphDataset(
        root='/root/autodl-tmp/wln/data',
        csv_file='/root/autodl-tmp/wln/data/PDT_USPTO_MIT_last2000.csv',
        chunk_size=100
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Feature dimensions: {get_feature_dimensions()}")
    print(f"Number of chunks: {dataset.num_chunks}")
    
    # 测试数据加载
    sample = dataset[0]
    print(f"Sample node features shape: {sample.x.shape}")
    print(f"Sample edge index shape: {sample.edge_index.shape}")
    print(f"Sample edge features shape: {sample.edge_attr.shape}")