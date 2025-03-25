# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # Load the CSV file
# file_path = "/workspace/vlm/lmms-eval-jake2/wandb_grid3.csv"  # Replace with the path to your CSV file
# output_dir = "/workspace/vlm/lmms-eval-jake2/figs/bilinear/grid3"  # Directory to save the plots
# os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# df = pd.read_csv(file_path)

# # Extract threshold values from the experiment names
# df[['stage_minus1', 'stage_0', 'stage_1']] = df['Name'].str.extract(
#     r'exp_run_\d+_thresh_([0-9.]+)_([0-9.]+)_([0-9.]+)'
# ).astype(float)

# # List of tasks to analyze
# tasks = {
#     "pope_pop/pope_f1_score": "Pope F1 Score",
#     "vqav2_val_lite/exact_match": "VQA Exact Match",
#     "mmbench_en_dev_lite/gpt_eval_score": "MMBench GPT Eval Score",
# }

# mmstar_tasks = [
#     "mmstar/science & technology",
#     "mmstar/logical reasoning",
#     "mmstar/math",
#     "mmstar/instance reasoning",
#     "mmstar/fine-grained perception",
#     "mmstar/coarse perception",
#     "mmstar/average",
# ]

# # Plot individual tasks
# for task_col, task_name in tasks.items():
#     if task_col not in df.columns:
#         print(f"Column {task_col} not found in the dataset. Skipping...")
#         continue

#     print(f"\nAnalyzing trends for {task_name}...")
#     for stage in ['stage_minus1', 'stage_0', 'stage_1']:
#         stage_means = df.groupby(stage)[task_col].mean()

#         plt.figure()
#         plt.plot(stage_means.index, stage_means.values, marker='o')
#         plt.title(f"{task_name} vs {stage}")
#         plt.xlabel(f"{stage} Threshold")
#         plt.ylabel(task_name)
#         plt.grid(True)
#         plt.savefig(os.path.join(output_dir, f"{task_name.replace(' ', '_')}_vs_{stage}.png"))
#         plt.close()

# # Combine MMStar tasks into a single plot
# for stage in ['stage_minus1', 'stage_0', 'stage_1']:
#     plt.figure()
#     for task_col in mmstar_tasks:
#         print(f"\nAnalyzing trends for {task_col}...")
#         if task_col not in df.columns:
#             print(f"Column {task_col} not found in the dataset. Skipping...")
#             continue
        
#         task_name = task_col.split("/")[-1]
#         stage_means = df.groupby(stage)[task_col].mean()

#         plt.plot(stage_means.index, stage_means.values, marker='o', label=task_name)

#     plt.title(f"MMStar Tasks vs {stage}")
#     plt.xlabel(f"{stage} Threshold")
#     plt.ylabel("Performance")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(output_dir, f"MMStar_Tasks_vs_{stage}.png"))
#     plt.close()


# print("Plots saved successfully.")

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# 데이터 로드
file_path = "/workspace/vlm/lmms-eval-jake2/wandb_grid3.csv"  # CSV 경로
output_dir = "/workspace/vlm/lmms-eval-jake2/figs/bilinear/grid3"  # 출력 경로
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(file_path)

# 스테이지 값 추출
df[['stage_minus1', 'stage_0', 'stage_1']] = df['Name'].str.extract(
    r'exp_run_\d+_thresh_([0-9.]+)_([0-9.]+)_([0-9.]+)'
).astype(float)

# 작업 정의
tasks = {
    "pope_pop/pope_f1_score": "Pope F1 Score",
    "vqav2_val_lite/exact_match": "VQA Exact Match",
    "mmbench_en_dev_lite/gpt_eval_score": "MMBench GPT Eval Score",
}

mmstar_tasks = {
    "mmstar/fine-grained perception": "Fine-Grained Perception",
    "mmstar/coarse perception": "Coarse Perception",
    "mmstar/average": "MMStar Average",
}

# 모든 작업 추가
tasks.update(mmstar_tasks)

# 3D 플롯 함수
def plot_3d_task_performance(task_col, task_name, df):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 점 크기 및 색상으로 성능 표시
    scatter = ax.scatter(
        df['stage_minus1'], df['stage_0'], df['stage_1'],
        c=df[task_col], cmap='viridis', s=100, edgecolors='k', alpha=0.8
    )

    # 축 및 제목 설정
    ax.set_title(f"3D Performance Plot: {task_name}", fontsize=14)
    ax.set_xlabel("Stage -1 Threshold", fontsize=12)
    ax.set_ylabel("Stage 0 Threshold", fontsize=12)
    ax.set_zlabel("Stage 1 Threshold", fontsize=12)

    # 컬러바 추가
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label(task_name, fontsize=12)

    # 파일 저장
    output_file = os.path.join(output_dir, f"3D_{task_name.replace(' ', '_')}.png")
    plt.savefig(output_file)
    plt.show()

# 작업별 플롯 생성
for task_col, task_name in tasks.items():
    if task_col not in df.columns:
        print(f"Column {task_col} not found in the dataset. Skipping...")
        continue

    plot_3d_task_performance(task_col, task_name, df)