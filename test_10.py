"""
測試NeuralCoMapping整合

最終修改版：
- 模仿 test4_multistartpoint.py 的結構
- 使用指定的地圖檔案
- 測試10個不同的起始點
- 整合 RobotIndividualMapTracker
- 輸出 local map 圖片
- 收集並儲存 R1, R2, Intersection, Union 覆蓋率
- 測試結束後繪製 test4 風格的總結圖表
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必須使用 'Agg' (非互動式)
import matplotlib.pyplot as plt
import os
import csv 
from two_robot_dueling_dqn_attention.environment.robot_local_map_tracker import RobotIndividualMapTracker 

from neural_comapping_integration import (
    create_neural_comapping_robots,
    run_episode_with_neural_comapping
)


def test_ncm_multiple_start_points(map_file_path, start_points_list, output_dir='results_ncm_multi_startpoints'):
    """
    測試NeuralCoMapping在指定地圖和多個起始點的表現
    [修改] 收集完整的覆蓋率數據 (R1, R2, Inter, Union)
    [新增] 測試後繪製圖表
    """
    print("=" * 60)
    print(f"測試NeuralCoMapping (多起始點, 含Local Map追蹤)")
    print(f"地圖: {map_file_path}")
    print("=" * 60)
    
    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # [修改] 創建CSV檔案以記錄完整的覆蓋率數據 (模仿 test4)
    csv_path = os.path.join(output_dir, 'ncm_coverage_data.csv')
    fieldnames = ['StartPoint', 'Step', 'Robot1Coverage', 'Robot2Coverage', 'IntersectionCoverage', 'UnionCoverage']
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    results_by_start_point = []
    
    # 為每個起始點運行測試 (模仿 test4)
    for start_idx, (robot1_pos, robot2_pos) in enumerate(start_points_list):
        print(f"\n===== 測試起始點 {start_idx+1}/{len(start_points_list)} =====")
        print(f"機器人1起始位置: {robot1_pos}")
        print(f"機器人2起始位置: {robot2_pos}")
        
        current_output_dir = os.path.join(output_dir, f'start_point_{start_idx+1}')
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
            
        individual_maps_dir = os.path.join(current_output_dir, 'individual_maps')
        if not os.path.exists(individual_maps_dir):
            os.makedirs(individual_maps_dir)
        
        tracker = None 
        try:
            print("\n創建使用NeuralCoMapping的機器人...")
            robot1_wrapper, robot2_wrapper = create_neural_comapping_robots(
                map_file_path=map_file_path,
                robot1_pos=robot1_pos,
                robot2_pos=robot2_pos,
                use_neural=True,
                plot=True
            )
            
            tracker = RobotIndividualMapTracker(
                robot1_wrapper.robot, 
                robot2_wrapper.robot, 
                save_dir=individual_maps_dir
            )
            
            # [修改] run_episode 現在返回 'coverage_data'
            steps, exploration_ratio, coverage_data = run_episode_with_neural_comapping(
                robot1_wrapper, 
                robot2_wrapper, 
                tracker, 
                max_steps=1000,
                output_dir=current_output_dir
            )
            
            final_union_coverage = coverage_data[-1]['union_coverage'] if coverage_data else 0
            final_overlap = coverage_data[-1]['intersection_coverage'] if coverage_data else 0
            
            current_result = {
                'start_point': start_idx+1,
                'steps': steps,
                'exploration_ratio': exploration_ratio, # 這是全局探索率
                'union_coverage': final_union_coverage, # 這是 local map 聯集
                'overlap_ratio': final_overlap
            }
            results_by_start_point.append(current_result)
            
            print(f"  Steps: {steps}")
            print(f"  Final Union Coverage (Local): {final_union_coverage:.2%}")
            print(f"  Final Overlap (Local): {final_overlap:.2%}")

            # [修改] 將這個 episode 的所有數據寫入 CSV
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for data_point in coverage_data:
                    writer.writerow({
                        'StartPoint': start_idx+1,
                        'Step': data_point['step'],
                        'Robot1Coverage': data_point['robot1_coverage'],
                        'Robot2Coverage': data_point['robot2_coverage'],
                        'IntersectionCoverage': data_point['intersection_coverage'],
                        'UnionCoverage': data_point['union_coverage']
                    })
            
            if tracker:
                tracker.cleanup()

        except Exception as e:
            print(f"測試起始點 {start_idx+1} 時出錯: {e}")
            import traceback
            traceback.print_exc()
            
            if tracker is not None:
                tracker.cleanup()
    
    # 統計所有起始點的結果
    print("\n" + "=" * 60)
    print("總體測試結果統計")
    print("=" * 60)
    
    all_steps = [r['steps'] for r in results_by_start_point if 'steps' in r]
    all_union_cov = [r['union_coverage'] for r in results_by_start_point if 'union_coverage' in r]
    all_overlaps = [r['overlap_ratio'] for r in results_by_start_point if 'overlap_ratio' in r]
    
    if all_steps:
        print(f"平均步數: {np.mean(all_steps):.2f} ± {np.std(all_steps):.2f}")
        print(f"平均聯集覆蓋率 (Local): {np.mean(all_union_cov):.2%}")
        print(f"平均重疊率 (Local): {np.mean(all_overlaps):.2%}")
        print(f"最佳步數 (最少): {np.min(all_steps)}")
        print(f"最差步數 (最多): {np.max(all_steps)}")
    else:
        print("沒有成功的測試運行。")
    
    
    # --- [新增] 從 test4 複製的繪圖程式碼 ---
    print("\n正在從 CSV 生成匯總圖表...")
    try:
        # 從CSV檔案讀取所有數據
        all_data = {}
        with open(csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                start_point = int(row['StartPoint'])
                step = int(row['Step'])
                
                if start_point not in all_data:
                    all_data[start_point] = {}
                    all_data[start_point]['steps'] = []
                    all_data[start_point]['robot1'] = []
                    all_data[start_point]['robot2'] = []
                    all_data[start_point]['intersection'] = []
                    all_data[start_point]['union'] = []
                
                all_data[start_point]['steps'].append(step)
                all_data[start_point]['robot1'].append(float(row['Robot1Coverage']))
                all_data[start_point]['robot2'].append(float(row['Robot2Coverage']))
                all_data[start_point]['intersection'].append(float(row['IntersectionCoverage']))
                all_data[start_point]['union'].append(float(row['UnionCoverage']))
        
        # 為每個起始點創建單獨的全面覆蓋率圖表
        for start_point in sorted(all_data.keys()):
            plt.figure(figsize=(12, 8))
            
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['robot1'], 
                    'b-', linewidth=2, label='Robot 1')
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['robot2'], 
                    'r-', linewidth=2, label='Robot 2')
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['intersection'], 
                    'g-', linewidth=2, label='Intersection')
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['union'], 
                    'k-', linewidth=2, label='Union')
            
            plt.xlabel('Time (steps)', fontsize=14)
            plt.ylabel('Coverage', fontsize=14)
            plt.title(f'Time-Coverage Analysis for Start Point {start_point}', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.ylim(0, 1.05)
            
            plt.savefig(os.path.join(output_dir, f'time_coverage_startpoint_{start_point}.png'), dpi=300)
            plt.close()
        
        # 創建聯集覆蓋率比較圖表
        plt.figure(figsize=(12, 8))
        
        for start_point in sorted(all_data.keys()):
            plt.plot(all_data[start_point]['steps'], all_data[start_point]['union'], 
                    linewidth=2, label=f'Start Point {start_point}')
        
        plt.xlabel('Time (steps)', fontsize=14)
        plt.ylabel('Total Coverage (Union)', fontsize=14)
        plt.title('Total Coverage Comparison Across Different Start Points', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.ylim(0, 1.05)
        
        plt.savefig(os.path.join(output_dir, 'all_total_coverage_comparison.png'), dpi=300)
        plt.close()
        
        print(f"已生成所有起始點的覆蓋率分析圖表")
        
    except Exception as e:
        print(f"生成比較圖表時出錯: {str(e)}")
    # --- [新增] 繪圖程式碼結束 ---

    
    print(f"\n===== 完成所有起始點的測試 =====")
    print(f"結果儲存在: {output_dir}")
    print(f"覆蓋率數據儲存在: {csv_path}")
    
    return results_by_start_point


def main():
    """
    主函數 (模仿 test4_multistartpoint.py)
    """
    
    # 指定地圖檔案路徑 (從 test4 複製)
    
    # --- [最終修復] ---
    # 不使用 os.getcwd()，因為它不可靠 (取決於從哪裡執行腳本)
    # 我們改用 __file__ 來取得 test_10.py 檔案的 "絕對目錄"
    # 這樣無論您從哪個資料夾執行，路徑都是 100% 正確的
    
    # 取得 test_10.py 檔案所在的目錄
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # [重要] 確保這個路徑是正確的！
    # (從 script_dir (即 ncm_dungeon-main) 開始往下找)
    map_file_path = os.path.join(script_dir, 'data', 'DungeonMaps', 'test', 'img_6032b.png')
    # --- [修復結束] ---
    
    # 檢查地圖檔案是否存在 (從 test4 複製)
    if not os.path.exists(map_file_path):
        print(f"警告: 在 {map_file_path} 找不到指定的地圖檔案")
        print("請提供正確的地圖檔案路徑。")
        exit(1)
    
    # 定義10個起始點位置 [robot1_pos, robot2_pos] (從 test4 複製)
    start_points = [
        [[100, 100], [100, 100]],  # 起始點 1
        [[520, 120], [520, 120]],  # 起始點 2
        [[250, 250], [250, 250]],   # 起始點 3
        [[250, 130], [250, 130]],   # 起始點 4
        [[250, 100], [250, 100]],  # 起始點 5
        [[400, 120], [400, 120]],  # 起始點 6
        [[140, 410], [140, 410]],   # 起始點 7
        [[110, 590], [110, 590]],   # 起始點 8
        [[90, 300], [90, 300]],   # 起始點 9
        [[260, 200], [260, 200]],  # 起始點 10
    ]
    
    # 設置輸出目錄 (從 test4 複製)
    output_dir = 'results_ncm_multi_startpoints'
    
    # 運行測試
    test_ncm_multiple_start_points(map_file_path, start_points, output_dir)


if __name__ == "__main__":
    main()