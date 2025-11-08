"""
NeuralCoMapping Adapter
支持規則版(Hungarian)和神經網路版(mGNN)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


class SimplifiedGraphMatcher:
    """規則版bipartite matching"""
    
    def compute_affinity_matrix(self, robots, frontiers, op_map):
        """計算affinity matrix"""
        num_robots = len(robots)
        num_frontiers = len(frontiers)
        
        if num_frontiers == 0:
            return np.zeros((num_robots, 0))
        
        affinity = np.zeros((num_robots, num_frontiers))
        
        for i, robot_pos in enumerate(robots):
            for j, frontier_pos in enumerate(frontiers):
                dist = np.linalg.norm(np.array(robot_pos) - np.array(frontier_pos))
                gain = self._compute_exploration_gain(frontier_pos, op_map)
                affinity[i, j] = gain / (dist + 1.0)
        
        return affinity
    
    def _compute_exploration_gain(self, frontier_pos, op_map, radius=20):
        """計算探索收益"""
        x, y = int(frontier_pos[0]), int(frontier_pos[1])
        h, w = op_map.shape
        
        unknown_count = 0
        total_count = 0
        
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    total_count += 1
                    if op_map[ny, nx] == 127:
                        unknown_count += 1
        
        return unknown_count / max(total_count, 1)
    
    def solve_assignment(self, affinity_matrix):
        """求解最優分配"""
        if affinity_matrix.shape[1] == 0:
            return {}
        
        cost_matrix = -affinity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assignments = {}
        # [修正] linear_sum_assignment 找到的就已是最佳解，
        # 無論其q_value是正還是負，都應該被分配。
        for robot_idx, frontier_idx in zip(row_ind, col_ind):
            assignments[robot_idx] = frontier_idx
        
        return assignments


class NeuralGraphMatcher:
    """神經網路版matching"""
    
    def __init__(self, model):
        self.model = model
        
    def compute_affinity_matrix(self, robots, frontiers, op_map):
        """使用神經網路計算affinity"""
        from ncm_model_loader import extract_features
        
        node_features, edge_features, edge_indices = extract_features(
            robots, frontiers, op_map
        )
        
        with torch.no_grad():
            affinity = self.model(node_features, edge_indices, edge_features)
        
        return affinity.cpu().numpy()
    
    def solve_assignment(self, affinity_matrix):
        """求解最優分配"""
        if affinity_matrix.shape[1] == 0:
            return {}
        
        cost_matrix = -affinity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assignments = {}
        for robot_idx, frontier_idx in zip(row_ind, col_ind):
            if affinity_matrix[robot_idx, frontier_idx] > 0:
                assignments[robot_idx] = frontier_idx
        
        return assignments


class NeuralCoMappingPlanner:
    """
    NeuralCoMapping規劃器
    """
    
    def __init__(self, use_neural=False, model_path=None):
        self.use_neural = use_neural
        
        if use_neural:
            print("🔥 載入神經網路版NeuralCoMapping...")
            from ncm_model_loader import load_pretrained_ncm
            
            # 默認路徑支持.global
            if model_path is None:
                model_path = "a.global"
            
            model = load_pretrained_ncm(model_path)  # 使用正確的函數名
            self.matcher = NeuralGraphMatcher(model)
            print("✅ 神經網路模型載入完成!")
        else:
            print("📊 使用規則版NeuralCoMapping (Hungarian algorithm)")
            self.matcher = SimplifiedGraphMatcher()
    
    def select_frontiers(self, robots, frontiers, op_map):
        """
        為機器人選擇frontier
        
        Args:
            robots: [(x1,y1), (x2,y2)]
            frontiers: numpy array (N, 2)
            op_map: occupancy map
            
        Returns:
            {robot_idx: (fx, fy)}
        """
        if len(frontiers) == 0:
            return {}
        
        frontier_list = [tuple(f) for f in frontiers]
        
        # 計算affinity
        affinity_matrix = self.matcher.compute_affinity_matrix(
            robots, frontier_list, op_map
        )
        
        # 求解分配
        assignments_idx = self.matcher.solve_assignment(affinity_matrix)
        
        # 轉為positions
        assignments = {}
        for robot_idx, frontier_idx in assignments_idx.items():
            assignments[robot_idx] = frontier_list[frontier_idx]
        
        return assignments