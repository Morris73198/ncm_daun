"""
Multi-Robot Environment with NeuralCoMapping Integration

[!!! 模仿 TEST4 的重大修改 !!!]
- 新增: create_robots_with_custom_positions (來自 test4_multistartpoint.py)
- 新增: save_plot (來自 test4_multistartpoint.py)
- 修改: create_neural_comapping_robots 函數以接受 map_file_path 和 pos
- 修改: run_episode_with_neural_comapping 以接受 output_dir 並儲存圖像
- 新增: 整合 RobotIndividualMapTracker
- [修改] run_episode_with_neural_comapping 現在收集完整的 R1, R2, Intersection, Union 覆蓋率
- [修復] run_episode_with_neural_comapping 同步 other_robot_position 以修正繪圖
- [修復] create_robots_with_custom_positions 恢復使用 index_map=0 (這才是 test4 的原始邏輯)
"""

import numpy as np
from neural_comapping_adapter import NeuralCoMappingPlanner
from two_robot_dueling_dqn_attention.config import ROBOT_CONFIG

# --- 新增的
import os
from scipy import spatial
import matplotlib
# 設置 matplotlib 後端為 Agg (無需顯示)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- 關鍵修改：使用與 test4 相同的 Robot 類 ---
from two_robot_dueling_dqn_attention.environment.multi_robot_no_unknown import Robot


class RobotWithNeuralCoMapping:
    def __init__(self, original_robot, neural_planner):
        self.robot = original_robot
        self.planner = neural_planner
        self.other_robot_wrapper = None
        
        # [新增] 儲存 NCM 分配的長期目標
        self.current_global_goal = None
        
        # [新增] 全局規劃的頻率
        self.replanning_frequency = 10  # 每 10 步重新指派一次
        
        # [新增] 最小目標間距離（避免兩機器人選點太近）
        self.min_target_separation = self.robot.sensor_range * 1.5  # 預設 75
    
    def set_other_robot(self, other_wrapper):
        """設定另一個機器人的wrapper"""
        self.other_robot_wrapper = other_wrapper
    
    def filter_frontiers_by_range(self, frontiers, max_range=None):
        """
        [新增] 過濾出範圍內的 frontiers
        """
        if max_range is None:
            max_range = self.robot.sensor_range * 2
        
        distances = np.linalg.norm(frontiers - self.robot.robot_position, axis=1)
        
        in_range_mask = distances <= max_range
        in_range_frontiers = frontiers[in_range_mask]
        out_range_frontiers = frontiers[~in_range_mask]
        
        return in_range_frontiers, out_range_frontiers
    
    def adjust_assignments_for_separation(self, assignments, frontiers_array, robots):
        """
        [新增] 調整分配結果，確保兩個機器人的目標點不會太近
        """
        if len(assignments) < 2:
            return assignments
        
        if 0 not in assignments or 1 not in assignments:
            return assignments
        
        target_0 = np.array(assignments[0])
        target_1 = np.array(assignments[1])
        
        target_distance = np.linalg.norm(target_0 - target_1)
        
        robot_name = "Robot1" if self.robot.is_primary else "Robot2"
        
        if target_distance < self.min_target_separation:
            # print(f"[NCM SEPARATION] {robot_name}: 目標距離 {target_distance:.1f} < {self.min_target_separation:.1f}，需要調整")
            
            dist_0_to_target_0 = np.linalg.norm(np.array(robots[0]) - target_0)
            dist_1_to_target_1 = np.linalg.norm(np.array(robots[1]) - target_1)
            
            robot_to_adjust = 0 if dist_0_to_target_0 > dist_1_to_target_1 else 1
            other_robot = 1 - robot_to_adjust
            other_target = target_0 if other_robot == 0 else target_1
            
            robot_pos = np.array(robots[robot_to_adjust])
            
            dists_to_robot = np.linalg.norm(frontiers_array - robot_pos, axis=1)
            dists_to_other_target = np.linalg.norm(frontiers_array - other_target, axis=1)
            
            valid_mask = dists_to_other_target >= self.min_target_separation
            
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                valid_dists = dists_to_robot[valid_indices]
                best_idx = valid_indices[np.argmin(valid_dists)]
                
                new_target = tuple(frontiers_array[best_idx])
                assignments[robot_to_adjust] = new_target
                # print(f"[NCM SEPARATION] 調整 Robot{robot_to_adjust+1} 的目標 (新距離: {dists_to_robot[best_idx]:.1f})")
            else:
                # print(f"[NCM SEPARATION] 警告：找不到符合距離要求的替代目標，保持原分配")
                pass
        # else:
            # print(f"[NCM SEPARATION] {robot_name}: 目標距離 {target_distance:.1f} ✓ (≥ {self.min_target_separation:.1f})")
        
        return assignments
    
    def step_with_neural_planner(self):
        """
        [全新修正] 實現真正的 NCM 分層規劃邏輯
        """
        
        if (self.robot.steps % self.replanning_frequency == 0) or (self.current_global_goal is None):
            
            all_frontiers = self.robot.get_frontiers()
            
            if len(all_frontiers) == 0:
                return None, self.robot.check_done()

            max_range = self.robot.sensor_range * 2
            in_range_frontiers, out_range_frontiers = self.filter_frontiers_by_range(
                all_frontiers, max_range
            )
            
            robot_name = "Robot1" if self.robot.is_primary else "Robot2"
            
            if len(in_range_frontiers) > 0:
                frontiers = in_range_frontiers
                # print(f"[NCM RANGE] {robot_name}: 使用 {len(in_range_frontiers)} 個範圍內的 frontiers (距離 ≤ {max_range})")
            else:
                frontiers = all_frontiers
                # print(f"[NCM RANGE] {robot_name}: 範圍內無 frontiers，使用全部 {len(all_frontiers)} 個")

            robots = [
                tuple(self.robot.robot_position),
                tuple(self.robot.other_robot_position)
            ]

            assignments = self.planner.select_frontiers(
                robots, 
                frontiers,
                self.robot.op_map
            )
            
            adjusted_assignments = self.adjust_assignments_for_separation(
                assignments, 
                frontiers,
                robots
            )
            
            robot_idx = 0 if self.robot.is_primary else 1
            
            if robot_idx not in adjusted_assignments:
                # print(f"[NCM DEBUG] {robot_name}: NCM 未分配, 啟動 [後援-最近點]")
                
                dists = np.linalg.norm(frontiers - self.robot.robot_position, axis=1)
                
                if self.other_robot_wrapper and self.other_robot_wrapper.current_global_goal is not None:
                    other_goal = self.other_robot_wrapper.current_global_goal
                    dists_to_other = np.linalg.norm(frontiers - other_goal, axis=1)
                    
                    valid_mask = dists_to_other >= self.min_target_separation
                    if np.any(valid_mask):
                        valid_indices = np.where(valid_mask)[0]
                        new_target = frontiers[valid_indices[np.argmin(dists[valid_indices])]]
                        # print(f"[NCM DEBUG] {robot_name}: 後援選點時避開另一機器人")
                    else:
                        new_target = frontiers[np.argmin(dists)]
                else:
                    new_target = frontiers[np.argmin(dists)]
            else:
                assigned_target = np.array(adjusted_assignments[robot_idx])
                dist_to_target = np.linalg.norm(assigned_target - self.robot.robot_position)
                # print(f"[NCM DEBUG] {robot_name}: NCM 分配新目標 (距離: {dist_to_target:.1f})")
                new_target = assigned_target
            
            self.current_global_goal = new_target
            self.robot.current_target_frontier = self.current_global_goal

        
        if self.current_global_goal is None:
            return None, self.robot.check_done()

        observation, reward, task_done = self.robot.move_to_frontier(self.current_global_goal)
        
        if task_done:
            self.current_global_goal = None

        episode_done = self.robot.check_done()
        
        if hasattr(self.robot, 'shared_env') and self.robot.shared_env is not None:
            self.robot.shared_env.op_map = self.robot.op_map
        elif hasattr(self.robot, 'other_robot') and self.robot.other_robot is not None:
            self.robot.other_robot.op_map = self.robot.op_map

        self.robot.steps += 1
        
        return observation, episode_done
    
    def _execute_movement_step(self):
        """
        (此函數為 Robot 類內部邏輯的簡化，原版可能更複雜)
        (在我們的實現中，此函數未被 RobotWithNeuralCoMapping 直接調用)
        """
        pass # 實際邏輯在 self.robot.move_to_frontier 內部處理

# --- 新增輔助函數 (來自 test4_multistartpoint.py) ---
def save_plot(robot, step, output_path):
    """儲存單個機器人的繪圖
    """
    # 確保我們在主執行緒中創建 Figure
    plt.figure(figsize=(10, 10))
    robot.plot_env()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close('all')  # 關閉所有圖形以釋放記憶體

# --- 新增輔助函數 (來自 test4_multistartpoint.py) ---
def create_robots_with_custom_positions(map_file_path, robot1_pos=None, robot2_pos=None, train=False, plot=True):
    """創建使用特定地圖檔案和自定義起始位置的機器人
    """
    
    class CustomRobot(Robot):
        @classmethod
        def create_shared_robots_with_custom_setup(cls, map_file_path, robot1_pos=None, robot2_pos=None, train=False, plot=True):
            """創建共享環境的機器人實例，使用指定的地圖檔案和起始位置"""
            print(f"使用指定地圖創建共享環境的機器人: {map_file_path}")
            if robot1_pos is not None:
                print(f"機器人1自定義起始位置: {robot1_pos}")
            if robot2_pos is not None:
                print(f"機器人2自定義起始位置: {robot2_pos}")
            
            # [修復] 必須傳入 0 (或一個有效的 index)，即使我們馬上要覆蓋它
            # 這是 test4 的原始邏輯
            robot1 = cls(0, train, plot, is_primary=True)
            
            # 這一行會用 map_file_path 覆蓋上面加載的 index_map=0
            global_map, initial_positions = robot1.map_setup(map_file_path)
            robot1.global_map = global_map
            
            robot1.t = robot1.map_points(global_map)
            robot1.free_tree = spatial.KDTree(robot1.free_points(global_map).tolist())
            
            if robot1_pos is not None and robot2_pos is not None:
                robot1_pos = np.array(robot1_pos, dtype=np.int64)
                robot2_pos = np.array(robot2_pos, dtype=np.int64)
                
                map_height, map_width = global_map.shape
                robot1_pos[0] = np.clip(robot1_pos[0], 0, map_width-1)
                robot1_pos[1] = np.clip(robot1_pos[1], 0, map_height-1)
                robot2_pos[0] = np.clip(robot2_pos[0], 0, map_width-1)
                robot2_pos[1] = np.clip(robot2_pos[1], 0, map_height-1)
                
                if global_map[robot1_pos[1], robot1_pos[0]] == 1:
                    print("警告: 機器人1的指定位置是障礙物，將移至 ближайшие 自由空間")
                    robot1_pos = robot1.nearest_free(robot1.free_tree, robot1_pos)
                    
                if global_map[robot2_pos[1], robot2_pos[0]] == 1:
                    print("警告: 機器人2的指定位置是障礙物，將移至 ближайшие 自由空間")
                    robot2_pos = robot1.nearest_free(robot1.free_tree, robot2_pos)
                
                robot1.robot_position = robot1_pos
                robot1.other_robot_position = robot2_pos

                # --- [所有修補程式 (v5) 已移除] ---
                # 我們不再需要儲存 'custom_start_pos'
                
            else:
                robot1.robot_position = initial_positions[0].astype(np.int64)
                robot1.other_robot_position = initial_positions[1].astype(np.int64)
                
                # --- [所有修補程式 (v5) 已移除] ---
                # 我們不再需要儲存 'custom_start_pos'
            
            robot1.op_map = np.ones(global_map.shape) * 127
            robot1.map_size = np.shape(global_map)
            
            # --- [所有修補程式 (v5) 已移除] ---
            # 我們不再需要「劫持」map_dir 或 initial_positions，
            # 因為我們不再呼叫 reset()。
            
            robot2 = cls(0, train, plot, is_primary=False, shared_env=robot1)
            
            robot1.other_robot = robot2
            
            if plot:
                if hasattr(robot1, 'fig'):
                    plt.close(robot1.fig)
                if hasattr(robot2, 'fig'):
                    plt.close(robot2.fig)
                
                robot1.initialize_visualization()
                robot2.initialize_visualization()
            
            return robot1, robot2
    
    return CustomRobot.create_shared_robots_with_custom_setup(
        map_file_path, 
        robot1_pos=robot1_pos, 
        robot2_pos=robot2_pos, 
        train=train, 
        plot=plot
    )


# --- [重大修改] ---
# 這是修復你的錯誤的關鍵：函數定義現在接受 'map_file_path', 'robot1_pos', 'robot2_pos'
def create_neural_comapping_robots(map_file_path, robot1_pos, robot2_pos, use_neural=False, model_path=None, plot=True):
    """
    [修改] 創建使用NeuralCoMapping的雙機器人系統
    - 使用 map_file_path 和自定義位置
    """
    
    robot1, robot2 = create_robots_with_custom_positions(
        map_file_path,
        robot1_pos=robot1_pos,
        robot2_pos=robot2_pos,
        train=False,
        plot=plot
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


# --- [重大修改] ---
def run_episode_with_neural_comapping(robot1_wrapper, robot2_wrapper, tracker, max_steps=1000, output_dir=None):
    """
    [修改] 
    - 接受 tracker 參數
    - 追蹤 local maps
    - [修改] 計算並返回完整的 coverage_data (R1, R2, Inter, Union)
    - [修復] 同步 other_robot_position
    """
    
    # --- [最終修復：模仿 test_4.py] ---
    # 不再呼叫 reset()，改為呼叫 begin()
    # begin() 會在 *正確的* 自訂起始點初始化感測器
    robot1_wrapper.robot.begin()
    robot2_wrapper.robot.begin()
    
    # --- [v5 修補程式已移除] ---
    # 我們不再需要 'custom_start_pos' 修補程式，
    # 因為 begin() 已經在正確的位置初始化了
    
    done1 = done2 = False
    steps = 0
    
    # [修改] 儲存完整的覆蓋率數據
    coverage_data = []
    
    print(f"  開始探索...", end='', flush=True)
    
    tracker.start_tracking()
    
    # --- [v5 邏輯保留] ---
    # 邏輯與 test_4.py 相同：
    # 在開始迴圈 *之前*，先 update() 和 save_plot() 
    
    # 1. 在新位置掃描感測器
    tracker.update()
    
    if output_dir:
        # 2. 繪製 step 0 的狀態 (現在感測器是正確的)
        save_plot(robot1_wrapper.robot, 0, os.path.join(output_dir, 'robot1_step_0000.png'))
        save_plot(robot2_wrapper.robot, 0, os.path.join(output_dir, 'robot2_step_0000.png'))
    
    # 3. 儲存地圖
    tracker.save_current_maps(0)
    # --- [邏輯保留結束] ---

    total_explorable = np.sum(robot1_wrapper.robot.global_map == 255)
    
    # 儲存 0 時刻的數據
    coverage_data.append({
        'step': 0,
        'robot1_coverage': 0.0,
        'robot2_coverage': 0.0,
        'intersection_coverage': 0.0,
        'union_coverage': 0.0
    })

    while steps < max_steps and not (done1 and done2):
        if not done1:
            _, done1 = robot1_wrapper.step_with_neural_planner()
        
        if not done2:
            _, done2 = robot2_wrapper.step_with_neural_planner()
        
        # --- [修復] ---
        # 手動同步兩個機器人*更新後*的位置，這樣繪圖才會正確
        # (這是 test4_multistartpoint.py 中的邏輯)
        robot1_wrapper.robot.other_robot_position = robot2_wrapper.robot.robot_position.copy()
        robot2_wrapper.robot.other_robot_position = robot1_wrapper.robot.robot_position.copy()
        # --- [修復結束] ---
        
        steps += 1
        
        tracker.update()
        
        if steps % 100 == 0:
            # 打印的是全局探索率
            global_exploration_ratio = np.sum(robot1_wrapper.robot.op_map == 255) / \
                                        np.sum(robot1_wrapper.robot.global_map == 255)
            print(f"\r  步數: {steps}, 全局探索率: {global_exploration_ratio:.1%}", end='', flush=True)
        
        # 繪圖邏輯 (每 10 步儲存一次)
        if steps % 10 == 0: 
            if output_dir:
                save_plot(robot1_wrapper.robot, steps, os.path.join(output_dir, f'robot1_step_{steps:04d}.png'))
                save_plot(robot2_wrapper.robot, steps, os.path.join(output_dir, f'robot2_step_{steps:04d}.png'))
                
                tracker.save_current_maps(steps)
            
            # [修改] 計算並記錄所有覆蓋率 (模仿 test4)
            robot1_map = tracker.robot1_individual_map
            robot2_map = tracker.robot2_individual_map
            
            if robot1_map is not None and robot2_map is not None and total_explorable > 0:
                
                # 計算 R1
                robot1_explored = np.sum(robot1_map == 255)
                robot1_coverage = robot1_explored / total_explorable
                
                # 計算 R2
                robot2_explored = np.sum(robot2_map == 255)
                robot2_coverage = robot2_explored / total_explorable
                
                # 計算交集 (Intersection)
                intersection = np.sum((robot1_map == 255) & (robot2_map == 255))
                intersection_coverage = intersection / total_explorable
                
                # 計算聯集 (Union)
                union = np.sum((robot1_map == 255) | (robot2_map == 255))
                union_coverage = union / total_explorable
                
                coverage_data.append({
                    'step': steps,
                    'robot1_coverage': robot1_coverage,
                    'robot2_coverage': robot2_coverage,
                    'intersection_coverage': intersection_coverage,
                    'union_coverage': union_coverage
                })
    
    print()
    
    if output_dir:
        save_plot(robot1_wrapper.robot, steps, os.path.join(output_dir, f'robot1_final_{steps:04d}.png'))
        save_plot(robot2_wrapper.robot, steps, os.path.join(output_dir, f'robot2_final_{steps:04d}.png'))
        
    tracker.save_current_maps(steps)
    
    tracker.stop_tracking()
    
    # 最終的全局探索率
    global_exploration_ratio = np.sum(robot1_wrapper.robot.op_map == 255) / \
                       np.sum(robot1_wrapper.robot.global_map == 255)
    
    # [修改] 返回 coverage_data
    return steps, global_exploration_ratio, coverage_data
