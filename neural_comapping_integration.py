"""
Multi-Robot Environment with NeuralCoMapping Integration
將NeuralCoMapping的global planner替換原本的DQN frontier selection
"""

import numpy as np
from neural_comapping_adapter import NeuralCoMappingPlanner
from two_robot_dueling_dqn_attention.config import ROBOT_CONFIG


class RobotWithNeuralCoMapping:
    def __init__(self, original_robot, neural_planner):
        self.robot = original_robot
        self.planner = neural_planner
        self.other_robot_wrapper = None
        
        # [新增] 儲存 NCM 分配的長期目標
        self.current_global_goal = None
        
        # [新增] 全局規劃的頻率
        self.replanning_frequency = 10  # 每 10 步重新指派一次
    
    def set_other_robot(self, other_wrapper):
        """設定另一個機器人的wrapper"""
        self.other_robot_wrapper = other_wrapper
    
    def step_with_neural_planner(self):
        """
        [全新修正] 實現真正的 NCM 分層規劃邏輯
        """
        
        # 檢查是否需要執行「全局規劃」(NCM 分配)
        # 條件：計數器到期 OR 機器人沒有長期目標
        if (self.robot.steps % self.replanning_frequency == 0) or (self.current_global_goal is None):
            
            # --- 1. 全局規劃器 (NCM) ---
            frontiers = self.robot.get_frontiers()
            
            if len(frontiers) == 0:
                return None, self.robot.check_done()

            # [可選優化] 在這裡插入 DBSCAN 群集邏輯 (參考 siyandong/utils/map_manager.py)
            # clustered_frontiers = run_dbscan(frontiers)
            # ... 為了簡單起見，我們先使用原始 frontiers ...

            robots = [
                tuple(self.robot.robot_position),
                tuple(self.robot.other_robot_position)
            ]

            assignments = self.planner.select_frontiers(
                robots, 
                frontiers, # 理想情況下應為 clustered_frontiers
                self.robot.op_map
            )
            
            robot_idx = 0 if self.robot.is_primary else 1
            robot_name = "Robot1" if self.robot.is_primary else "Robot2" # Debug
            
            if robot_idx not in assignments:
                print(f"[NCM DEBUG] {robot_name}: NCM 未分配, 啟動 [後援-最近點]")
                dists = np.linalg.norm(frontiers - self.robot.robot_position, axis=1)
                new_target = frontiers[np.argmin(dists)]
            else:
                print(f"[NCM DEBUG] {robot_name}: NCM 分配新目標")
                new_target = np.array(assignments[robot_idx])
            
            # [關鍵] 更新長期目標
            self.current_global_goal = new_target
            # 確保 robot 物件也更新它，這樣 move_to_frontier 才能正確規劃
            self.robot.current_target_frontier = self.current_global_goal

        # --- 2. 局部規劃器 (每一步都執行) ---
        
        if self.current_global_goal is None:
            # 即使 NCM 失敗了，也沒找到後援點 (例如地圖是空的)
            return None, self.robot.check_done()

        # [關鍵] 無論如何，都朝著「儲存的」長期目標移動一步
        # move_to_frontier 內部會自己處理 A* 尋路
        observation, reward, task_done = self.robot.move_to_frontier(self.current_global_goal)
        
        # 如果 move_to_frontier 說 "done" (表示它抵達了)，
        # 我們就把長期目標設為 None，強制 NCM 在下一步重新規劃
        if task_done:
            self.current_global_goal = None

        # 檢查 Episode 是否結束 (探索率是否達標)
        episode_done = self.robot.check_done()
        
        # (地圖同步)
        if hasattr(self.robot, 'shared_env') and self.robot.shared_env is not None:
            self.robot.shared_env.op_map = self.robot.op_map
        elif hasattr(self.robot, 'other_robot') and self.robot.other_robot is not None:
            self.robot.other_robot.op_map = self.robot.op_map

        # (更新步數和繪圖)
        self.robot.steps += 1
        if self.robot.plot and self.robot.steps % 5 == 0:
            self.robot.plot_env()
            import matplotlib.pyplot as plt
            plt.pause(0.01)

        return observation, episode_done
    
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
    from two_robot_dueling_dqn_attention.environment.multi_robot import Robot
    
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