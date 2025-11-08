"""
Multi-Robot Environment with NeuralCoMapping Integration
將NeuralCoMapping的global planner替換原本的DQN frontier selection
"""

import numpy as np
from neural_comapping_adapter import NeuralCoMappingPlanner
from two_robot_dueling_dqn_attention.config import ROBOT_CONFIG


class RobotWithNeuralCoMapping:
    """
    將NeuralCoMapping整合到現有Robot class的wrapper
    只替換frontier selection部分,保留A*和其他模組
    """
    
    def __init__(self, original_robot, neural_planner):
        """
        Args:
            original_robot: 你原本的Robot實例
            neural_planner: NeuralCoMappingPlanner實例
        """
        self.robot = original_robot
        self.planner = neural_planner
        
        # 記錄配對的另一個機器人(用於雙機器人系統)
        self.other_robot_wrapper = None
    
    def set_other_robot(self, other_wrapper):
        """設定另一個機器人的wrapper"""
        self.other_robot_wrapper = other_wrapper
    
    def step_with_neural_planner(self):
        """
        使用NeuralCoMapping選擇frontier,然後用A*規劃路徑
        替代原本的DQN action selection
        """
        # 如果正在執行路徑,繼續執行
        if (self.robot.is_moving_to_target and 
            self.robot.current_path is not None and 
            self.robot.current_path_index < self.robot.current_path.shape[1]):
            return self._execute_movement_step()
        
        # 路徑執行完畢或沒有路徑,重新規劃
        # 1. 獲取frontiers
        frontiers = self.robot.get_frontiers()
        
        if len(frontiers) == 0:
            return None, True  # 沒有frontier,探索完成
        
        # 2. 準備兩個機器人的位置
        robots = [
            tuple(self.robot.robot_position),
            tuple(self.robot.other_robot_position)
        ]
        
        # 3. 使用NeuralCoMapping選擇frontier
        assignments = self.planner.select_frontiers(
            robots, 
            frontiers, 
            self.robot.op_map
        )
        
        # 4. 獲取此機器人的目標frontier
        robot_idx = 0 if self.robot.is_primary else 1
        
        if robot_idx not in assignments:
            # 如果沒有分配給此機器人,選距離最近的
            dists = np.linalg.norm(frontiers - self.robot.robot_position, axis=1)
            target_frontier = frontiers[np.argmin(dists)]
        else:
            target_frontier = np.array(assignments[robot_idx])

        
        # 5. 使用A*規劃到目標frontier的路徑
        self.robot.current_target_frontier = target_frontier
        
        # 使用原有的A*路徑規劃 (使用astar_path而非astar)
        if hasattr(self.robot, 'astar_path'):
            # 如果有astar_path方法,使用它
            path = self.robot.astar_path(
                self.robot.op_map,
                self.robot.robot_position.astype(np.int32),
                target_frontier.astype(np.int32)
            )
        else:
            # 否則使用astar
            path = self.robot.astar(
                self.robot.op_map,
                self.robot.robot_position,
                target_frontier
            )
        
        if path is None:
            self.robot.is_moving_to_target = False
            return None, False
        
        # 檢查path長度
        path_length = len(path) if isinstance(path, list) else (path.shape[1] if path.ndim == 2 else path.shape[0])
        
        if path_length <= 1:
            # 路徑太短(只有起點),認為已經到達frontier
            # 強制執行感測器掃描以更新地圖並移除這個frontier
            self.robot.op_map = self.robot.inverse_sensor(
                self.robot.robot_position,
                self.robot.sensor_range,
                self.robot.op_map,
                self.robot.global_map
            )
            
            # !!重要!! 同步地圖到另一個機器人
            if hasattr(self.robot, 'shared_env') and self.robot.shared_env is not None:
                self.robot.shared_env.op_map = self.robot.op_map
            elif hasattr(self.robot, 'other_robot') and self.robot.other_robot is not None:
                self.robot.other_robot.op_map = self.robot.op_map
            
            self.robot.is_moving_to_target = False
            
            # 重新獲取frontiers看是否有新的
            new_frontiers = self.robot.get_frontiers()
            if len(new_frontiers) == 0:
                return None, True
            
            return None, False
        
        # 6. 設置路徑並開始移動
        if isinstance(path, np.ndarray):
            if path.ndim == 1:
                self.robot.current_path = path.reshape(2, -1)
            elif path.shape[0] == 2:
                self.robot.current_path = path
            else:
                self.robot.current_path = path.T
        else:
            self.robot.current_path = np.array(path).T
            
        self.robot.current_path_index = 0
        self.robot.is_moving_to_target = True
        
        return self._execute_movement_step()
    
    def _execute_movement_step(self):
        """
        執行一步移動 (使用原有的移動邏輯)
        !! 已修正為逐步移動 !!
        """
        if not self.robot.is_moving_to_target or self.robot.current_path is None:
            self.robot.is_moving_to_target = False
            return None, False
        
        if self.robot.current_path_index >= self.robot.current_path.shape[1]:
            # 路徑上的所有點都已訪問
            self.robot.is_moving_to_target = False
            
            # 檢查是否真的到達最終目標
            dist_to_target = np.linalg.norm(self.robot.robot_position - self.robot.current_target_frontier)
            
            # 從 config.py 獲取閾值 (如果不存在則默認為 10)
            target_reach_threshold = ROBOT_CONFIG.get('target_reach_threshold', 10)

            if dist_to_target < target_reach_threshold:
                 # 真的到了，強制掃描一次以清除這個frontier
                self.robot.op_map = self.robot.inverse_sensor(
                    self.robot.robot_position, self.robot.sensor_range,
                    self.robot.op_map, self.robot.global_map
                )
            
            return None, False # 移動結束

        # --- (從 multi_robot.py 複製並修改的 "正確" 邏輯) ---
        
        # 1. 獲取路徑上的 *下一個* 檢查點
        next_point = self.robot.current_path[:, self.robot.current_path_index]
        
        # 2. 計算朝向下一個檢查點的 *移動向量*
        move_vector = next_point - self.robot.robot_position
        dist = np.linalg.norm(move_vector)
        
        # 3. 從 config.py 獲取步長 (如果不存在則默認為 2)
        movement_step = ROBOT_CONFIG.get('movement_step', 2) 
        
        # 4. 確保最小移動 (如果離下一個檢查點太近，直接跳到下下個)
        MIN_MOVEMENT = 1.0
        if dist < MIN_MOVEMENT:
            self.robot.current_path_index += 1
            # 只是推進了索引，還沒移動，所以回傳 False (未完成)
            return self.robot.get_observation(), False 

        # 5. 限制這一步的長度 (關鍵！)
        if dist > movement_step:
            move_vector = move_vector * (movement_step / dist)
        
        # 6. 執行這 "一小步" 移動
        old_position = self.robot.robot_position.copy()
        new_position = self.robot.robot_position + move_vector
        self.robot.robot_position = np.round(new_position).astype(np.int64)
        
        # 7. 邊界檢查
        self.robot.robot_position[0] = np.clip(self.robot.robot_position[0], 0, self.robot.map_size[1]-1)
        self.robot.robot_position[1] = np.clip(self.robot.robot_position[1], 0, self.robot.map_size[0]-1)

        # 8. 碰撞檢查 (使用觀測地圖 op_map)
        #    (注意：原版 multi_robot.py 使用 global_map 檢查，這裡用 op_map 可能更合理)
        if self.robot.op_map[self.robot.robot_position[1], self.robot.robot_position[0]] == 1:
            self.robot.robot_position = old_position # 撤銷移動
            self.robot.is_moving_to_target = False # 路徑被擋，停止
            self.robot.current_path = None # 清除路徑，下次重新規劃
            return self.robot.get_observation(), False # 沒完成，但路徑失敗

        # 9. 更新地圖 (!! 修正：同時呼叫 robot_model 和 inverse_sensor !!)
        self.robot.op_map = self.robot.robot_model(
            self.robot.robot_position, self.robot.robot_size,
            self.robot.t, self.robot.op_map
        )
        self.robot.op_map = self.robot.inverse_sensor(
            self.robot.robot_position, self.robot.sensor_range,
            self.robot.op_map, self.robot.global_map
        )
        
        # 10. (地圖同步)
        if hasattr(self.robot, 'shared_env') and self.robot.shared_env is not None:
            self.robot.shared_env.op_map = self.robot.op_map
        elif hasattr(self.robot, 'other_robot') and self.robot.other_robot is not None:
            self.robot.other_robot.op_map = self.robot.op_map
        
        # 11. (記錄軌跡)
        self.robot.xPoint = np.append(self.robot.xPoint, self.robot.robot_position[0])
        self.robot.yPoint = np.append(self.robot.yPoint, self.robot.robot_position[1])
        
        # 12. 只有當離下一個檢查點足夠近時，才推進 path_index
        target_reach_threshold = ROBOT_CONFIG.get('target_reach_threshold', 10)
        if dist < target_reach_threshold or dist < movement_step:
             self.robot.current_path_index += 1

        # 13. (更新步數和繪圖)
        self.robot.steps += 1
        done = self.robot.check_done()
        observation = self.robot.get_observation()
        
        if self.robot.plot and self.robot.steps % 5 == 0:
            self.robot.plot_env()
            import matplotlib.pyplot as plt
            plt.pause(0.01)
        
        return observation, done


def create_neural_comapping_robots(index_map=0, use_neural=False, model_path=None):
    """
    創建使用NeuralCoMapping的雙機器人系統
    """
    from two_robot_dueling_dqn_attention.environment.multi_robot_no_unknown import Robot
    
    robot1, robot2 = Robot.create_shared_robots(
        index_map=index_map,
        train=False,
        plot=True
    )
    
    planner = NeuralCoMappingPlanner(
        use_neural=use_neural,
        model_path=model_path
    )
    
    robot1_wrapper = RobotWithNeuralCoMapping(robot1, planner)
    robot2_wrapper = RobotWithNeuralCoMapping(robot2, planner)
    
    robot1_wrapper.set_other_robot(robot2_wrapper)
    robot2_wrapper.set_other_robot(robot1_wrapper)
    
    return robot1_wrapper, robot2_wrapper


def run_episode_with_neural_comapping(robot1_wrapper, robot2_wrapper, max_steps=1000):
    """使用NeuralCoMapping運行一個episode"""
    robot1_wrapper.robot.reset()
    robot2_wrapper.robot.reset()
    
    done1 = done2 = False
    steps = 0
    
    print(f"  開始探索...", end='', flush=True)
    
    while steps < max_steps and not (done1 and done2):
        if not done1:
            _, done1 = robot1_wrapper.step_with_neural_planner()
        
        if not done2:
            _, done2 = robot2_wrapper.step_with_neural_planner()
        
        steps += 1
        
        if steps % 100 == 0:
            exploration_ratio = np.sum(robot1_wrapper.robot.op_map == 255) / \
                               np.sum(robot1_wrapper.robot.global_map == 255)
            print(f"\r  步數: {steps}, 探索率: {exploration_ratio:.1%}", end='', flush=True)
        
        if robot1_wrapper.robot.plot and steps % 50 == 0:
            import matplotlib.pyplot as plt
            robot1_wrapper.robot.plot_env()
            plt.pause(0.01)
    
    print()
    
    exploration_ratio = np.sum(robot1_wrapper.robot.op_map == 255) / \
                       np.sum(robot1_wrapper.robot.global_map == 255)
    
    return steps, exploration_ratio