# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import add_self_loops

class WLNConv(MessagePassing):
    """WLN 卷积层"""
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.linear = nn.Linear(in_channels + 6, out_channels)  # +6 for bond features
        self.batch_norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        if x is None or edge_index is None:
            raise ValueError("Input features or edge_index is None")
            
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        if edge_attr is not None:
            self_loop_attr = torch.zeros((x.size(0), edge_attr.size(1)), 
                                      device=edge_attr.device)
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.batch_norm(out)
        return out
    
    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            return F.relu(self.linear(torch.cat([x_j, edge_attr], dim=1)))
        return F.relu(self.linear(x_j))

# class WLN(nn.Module):
#     """无监督 WLN 模型 - 学习分子的潜在表示"""
#     def __init__(self, 
#                  in_channels, 
#                  hidden_channels, 
#                  latent_channels=32,
#                  num_layers=3, 
#                  dropout=0.2):
#         super().__init__()
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.latent_channels = latent_channels
#         self.in_channels = in_channels  # 保存输入维度
        
#         # 编码器部分
#         self.input_mlp = nn.Sequential(
#             nn.Linear(in_channels, hidden_channels),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_channels),
#             nn.Dropout(dropout)
#         )
        
#         # WLN层
#         self.convs = nn.ModuleList()
#         for _ in range(num_layers):
#             self.convs.append(WLNConv(hidden_channels, hidden_channels))
            
#         # 潜在空间映射
#         self.latent_mlp = nn.Sequential(
#             nn.Linear(hidden_channels * 3, hidden_channels),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_channels),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_channels, latent_channels)
#         )
        
#         # 解码器部分
#         self.node_decoder = nn.Sequential(
#             nn.Linear(latent_channels, hidden_channels),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_channels),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_channels, in_channels)
#         )
    
#     def encode(self, batch):
#         x, edge_index, edge_attr, batch_idx = (
#             batch.x, 
#             batch.edge_index, 
#             batch.edge_attr,
#             batch.batch
#         )
        
#         # 特征转换
#         x = self.input_mlp(x)
        
#         # 存储所有层的输出
#         all_layer_outputs = [x]
        
#         # 消息传递
#         for conv in self.convs:
#             x = conv(x, edge_index, edge_attr)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#             all_layer_outputs.append(x)
            
#         # 残差连接
#         x = torch.stack(all_layer_outputs).sum(dim=0)
        
#         # 多种池化
#         x_mean = global_mean_pool(x, batch_idx)
#         x_add = global_add_pool(x, batch_idx)
#         x_max = global_max_pool(x, batch_idx)
        
#         # 组合池化结果
#         x_combined = torch.cat([x_mean, x_add, x_max], dim=1)
        
#         # 映射到潜在空间
#         latent = self.latent_mlp(x_combined)
#         return latent
    
#     def forward(self, batch):
#         # 编码
#         z = self.encode(batch)
        
#         # 每个分子的潜在表示重复对应节点的次数
#         batch_size = batch.batch.max().item() + 1
#         nodes_per_graph = torch.bincount(batch.batch)
        
#         # 将潜在表示扩展到每个节点
#         z_expanded = torch.cat([z[i:i+1].repeat(count, 1) 
#                               for i, count in enumerate(nodes_per_graph)])
        
#         # 解码回节点特征
#         x_reconstructed = self.node_decoder(z_expanded)
        
#         return {
#             'latent': z,  # 潜在表示
#             'reconstructed': x_reconstructed  # 重构的节点特征
#         }

#     def get_embeddings(self, batch):
#         """仅获取潜在表示，用于下游任务"""
#         self.eval()
#         with torch.no_grad():
#             return self.encode(batch)

class WLN(nn.Module):
    """无监督 WLN 模型 - 与Transformer维度对齐的分子表示学习"""
    def __init__(self, 
                 in_channels, 
                 hidden_channels=768, 
                 seq_length=20,
                 num_layers=3, 
                 dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.seq_length = seq_length
        self.in_channels = in_channels
        
        # 编码器部分
        self.input_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(dropout)
        )
        
        # WLN层
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(WLNConv(hidden_channels, hidden_channels))
            
        # 序列表示生成
        self.sequence_proj = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(dropout)
        )
        
        # Pooler输出层
        self.pooler = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh()
        )
        
        # 解码器部分
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels)
        )
    
    def encode(self, batch):
        x, edge_index, edge_attr, batch_idx = (
            batch.x, 
            batch.edge_index, 
            batch.edge_attr,
            batch.batch
        )
        
        # 特征转换
        x = self.input_mlp(x)
        
        # 存储所有层的输出
        all_layer_outputs = [x]
        
        # 消息传递
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
            all_layer_outputs.append(x)
            
        # 残差连接
        x = torch.stack(all_layer_outputs).sum(dim=0)
        
        # 多种池化
        x_mean = global_mean_pool(x, batch_idx)
        x_add = global_add_pool(x, batch_idx)
        x_max = global_max_pool(x, batch_idx)
        
        # 组合池化结果
        x_combined = torch.cat([x_mean, x_add, x_max], dim=1)
        
        # 生成序列表示和pooler输出
        hidden_states = self.sequence_proj(x_combined)
        
        # 调整shape为[batch_size, seq_length, hidden_size]
        batch_size = hidden_states.size(0)
        last_hidden_state = hidden_states.unsqueeze(1).expand(-1, self.seq_length, -1)
        
        # 生成pooler_output
        pooler_output = self.pooler(hidden_states)
        
        return {
            'last_hidden_state': last_hidden_state,  # [batch_size, seq_length, hidden_size]
            'pooler_output': pooler_output  # [batch_size, hidden_size]
        }
    
    def forward(self, batch):
        # 编码得到transformer风格的输出
        outputs = self.encode(batch)
        
        # 获取每个分子的表示用于重构
        pooler_output = outputs['pooler_output']
        
        # 每个分子的表示重复对应节点的次数
        batch_size = batch.batch.max().item() + 1
        nodes_per_graph = torch.bincount(batch.batch)
        
        # 将表示扩展到每个节点
        z_expanded = torch.cat([pooler_output[i:i+1].repeat(count, 1) 
                              for i, count in enumerate(nodes_per_graph)])
        
        # 解码回节点特征
        x_reconstructed = self.node_decoder(z_expanded)
        
        return {
            'last_hidden_state': outputs['last_hidden_state'],
            'pooler_output': outputs['pooler_output'],
            'reconstructed': x_reconstructed
        }

    def get_embeddings(self, batch):
        """仅获取transformer风格的表示，用于下游任务"""
        self.eval()
        with torch.no_grad():
            return self.encode(batch)