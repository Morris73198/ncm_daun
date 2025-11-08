import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from collections import OrderedDict

#
# --------------------------------------------------------------------------------
# (架構定義) 
# --------------------------------------------------------------------------------
#

class GNN_Layer(MessagePassing):
    """
    GNN Layer (Message Passing) - 匹配 siyandong/neuralcomapping/utils/gnn.py
    """
    def __init__(self, in_dim, out_dim, args, aggr='add'):
        super(GNN_Layer, self).__init__(aggr=aggr)
        self.args = args
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.attn_merge = nn.Linear(2 * in_dim + args['f_dim'], in_dim)
        self.attn_proj = nn.Linear(in_dim, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(out_dim, out_dim)
        )
        
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        nn.init.xavier_uniform_(self.attn_merge.weight)
        nn.init.xavier_uniform_(self.attn_proj.weight)

    def forward(self, x, edge_index, edge_attr):
        num_nodes = x.size(0)
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        #
        # [ 錯誤 2 修正 ] 
        # ------------------------------------------------------------------------
        # torch_geometric 1.6.3 的 add_self_loops 不支援 2D 的 edge_attr。
        # 我們必須手動處理：
        
        # 1. 僅為 edge_index 添加 self-loops
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # 2. 為新的 self-loops 創建對應的「填充」屬性 (N, 6)
        #    (N = 節點數量, 6 = f_dim)
        loop_attr = torch.zeros(num_nodes, self.args['f_dim'], device=edge_attr.device)
        
        # 3. 將原始的 edge_attr 和新的 loop_attr 結合起來
        edge_attr_with_loops = torch.cat([edge_attr, loop_attr], dim=0)
        # ------------------------------------------------------------------------
        #
        
        row, col = edge_index_with_loops # [修正] 使用新的索引
        x_i = x[row]
        x_j = x[col]

        # GNN (attention)
        # [修正] 使用新的屬性張量
        alpha_in = torch.cat([x_i, x_j, edge_attr_with_loops], dim=-1)
        alpha_in = self.activation(self.attn_merge(alpha_in))
        alpha = self.attn_proj(alpha_in)
        
        alpha = torch.exp(torch.clamp(alpha, -5, 5))
        
        # 訊息傳遞
        out = self.propagate(edge_index_with_loops, x=x, alpha=alpha) # [修正] 使用新的索引
        
        # MLP (類似於原始 NCM 中的 FFN)
        out = self.mlp(out)
        return out

    def message(self, x_j, alpha):
        return x_j * alpha


class GNN(nn.Module):
    """
    GNN (Multiple Layers) - 匹配 siyandong/neuralcomapping/utils/gnn.py
    """
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        self.n_layer = args['n_layer']
        
        self.layers = nn.ModuleList()
        for i in range(self.n_layer):
            in_dim = self.args['g_h']
            out_dim = self.args['g_h']
            self.layers.append(GNN_Layer(in_dim, out_dim, args, aggr='add'))
        
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
    def forward(self, x, edge_index, edge_attr):
        x_ = x
        for i in range(self.n_layer):
            x_ = self.layers[i](x_, edge_index, edge_attr)
            if i < self.n_layer - 1:
                x_ = self.activation(x_)
        return x_


class NCM_Policy(nn.Module):
    """
    Neural CoMapping Policy - 匹配 siyandong/neuralcomapping/model.py 中的 'actor'
    """
    def __init__(self, args):
        super(NCM_Policy, self).__init__()
        self.args = args
        self.n_agents = args['n_agents']
        
        self.agent_encoder = nn.Linear(args['f_dim'], args['g_h'])
        self.frontier_encoder = nn.Linear(args['f_dim'], args['g_h'])

        self.gnn = GNN(args)

        self.affinity = nn.Linear(args['g_h'] * 2, 1)

        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        # 權重初始化 (可選，但有助於穩定)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_attr):
        x_agent = x[:self.n_agents]
        x_frontier = x[self.n_agents:]
        
        x_agent_ = self.activation(self.agent_encoder(x_agent))
        x_frontier_ = self.activation(self.frontier_encoder(x_frontier))
        x_ = torch.cat([x_agent_, x_frontier_], dim=0)

        x_ = self.gnn(x_, edge_index, edge_attr)
        
        x_agent_ = x_[:self.n_agents]
        x_frontier_ = x_[self.n_agents:]
        
        n_frontiers = x_frontier_.size(0)
        
        x_agent_ = x_agent_.unsqueeze(1).repeat(1, n_frontiers, 1)
        x_frontier_ = x_frontier_.unsqueeze(0).repeat(self.n_agents, 1, 1)
        
        affinity = torch.cat([x_agent_, x_frontier_], dim=2)
        affinity = self.affinity(affinity).squeeze(2)
        
        return affinity

#
# --------------------------------------------------------------------------------
# (特徵提取)
# --------------------------------------------------------------------------------
#
def extract_features(robots, frontiers, op_map):
    """
    Extract node features, edge features, and edge indices for GNN
    """
    n_agents = len(robots)
    n_frontiers = len(frontiers)
    
    # Node features: [x, y, op_map_val, is_agent, is_frontier, dist_to_nearest_agent]
    nodes = []
    
    # Agent nodes
    for i, (rx, ry) in enumerate(robots):
        nodes.append([rx, ry, op_map[int(ry), int(rx)], 1, 0, 0])
        
    # Frontier nodes
    for i, (fx, fy) in enumerate(frontiers):
        dists = [np.linalg.norm(np.array(r) - np.array((fx, fy))) for r in robots]
        nodes.append([fx, fy, op_map[int(fy), int(fx)], 0, 1, min(dists)])
        
    node_features = torch.FloatTensor(nodes)
    
    # Edge indices and features
    edge_index = []
    edge_attr = []
    
    # f_dim = 6
    # [dist, is_aa, is_af, 0, 0, 0]
    
    # Agent <-> Agent
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            edge_index.extend([[i, j], [j, i]])
            dist = np.linalg.norm(np.array(robots[i]) - np.array(robots[j]))
            edge_attr.extend([[dist, 1, 0, 0, 0, 0], [dist, 1, 0, 0, 0, 0]]) 

    # Agent <-> Frontier
    for i in range(n_agents):
        for j in range(n_frontiers):
            edge_index.extend([[i, n_agents + j], [n_agents + j, i]])
            dist = np.linalg.norm(np.array(robots[i]) - np.array(frontiers[j]))
            edge_attr.extend([[dist, 0, 1, 0, 0, 0], [dist, 0, 1, 0, 0, 0]])
    
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    edge_attr = torch.FloatTensor(edge_attr)
    
    if edge_index.shape[1] != edge_attr.shape[0]:
        raise RuntimeError(f"特徵提取錯誤: 邊索引數量 ({edge_index.shape[1]}) "
                         f"與邊屬性數量 ({edge_attr.shape[0]}) 不匹配!")

    return node_features, edge_attr, edge_index

#
# --------------------------------------------------------------------------------
# (權重載入)
# --------------------------------------------------------------------------------
#
def load_pretrained_ncm(model_path=None):
    """
    [已修正] 載入預訓練的NCM模型
    """
    args = {
        'n_agents': 2,
        'f_dim': 6,       
        'n_head': 1,
        'n_layer': 3,
        'dropout': 0.1,
        'f_gh': 16,
        'g_dim': 16,
        'g_global_pool': 'max',
        'g_h': 32,
        'g_h_fc': 32
    }
    
    model = NCM_Policy(args)
    
    if model_path is not None:
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            source_state_dict = None
            
            # [ 錯誤 1 修正 ]
            # 檢查 'network' 或 'model_state_dict' 
            if isinstance(checkpoint, dict) and 'network' in checkpoint:
                source_state_dict = checkpoint['network']
                print(f"ℹ️ 找到權重於 'network' 鍵中...")
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                source_state_dict = checkpoint['model_state_dict']
                print(f"ℹ️ 找到權重於 'model_state_dict' 鍵中...")
            elif isinstance(checkpoint, dict):
                 # 如果是字典但沒有那兩個鍵, 可能是原始 state_dict
                source_state_dict = checkpoint
                print(f"ℹ️ (後備) 嘗試直接載入字典...")
            else:
                 raise ValueError("權重檔既不是字典也不是 state_dict。")
            
            # [ 錯誤 1 修正 ] 
            # 建立一個新的 state_dict，並剝離 "actor." 字首
            actor_state_dict = OrderedDict()
            prefix = "actor."
            
            keys_matched = 0
            for k, v in source_state_dict.items():
                if k.startswith(prefix):
                    name = k[len(prefix):] # 移除 "actor."
                    actor_state_dict[name] = v
                    keys_matched += 1
            
            if keys_matched == 0:
                 print(f"!! 警告: 找不到 'actor.' 字首，嘗試直接載入。 (可能是規則版?)")
                 # 如果沒有 'actor.' 字首 (例如您載入了一個非 NCM 的模型)，嘗試直接載入
                 model.load_state_dict(source_state_dict, strict=False)
                 print(f"✅ (後備) 成功載入 state_dict (無 'actor.' 字首): {model_path}")
            else:
                # 載入剝離後的權重
                print(f"ℹ️ 成功剝離 {keys_matched} 個 'actor.' 權重...")
                model.load_state_dict(actor_state_dict, strict=False) # strict=False 忽略不匹配的鍵
                print(f"✅ 成功從字典載入NCM模型權重 (剝離 'actor.' 字首): {model_path}")


        except Exception as e:
            print(f"❌ 載入NCM模型權重失敗: {e}")
            print(f"!! 警告: 正在使用未經訓練的隨機權重 !!")
    
    model.eval()
    return model