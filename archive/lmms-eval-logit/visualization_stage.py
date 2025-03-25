# import json
# import matplotlib.pyplot as plt
# import os

# # 파일 경로 설정
# logit_file_path = "/workspace/vlm/lmms-eval-logit/logit_mmstar_100_30_70_bilinear_s1.json"

# # JSON 데이터 로드
# with open(logit_file_path, "r") as f:
#     logits_data = json.load(f)

# # 스테이지와 클래스(A, B, C, D)에 해당하는 로짓 값 저장
# stages = ["Stage 0", "Stage 1", "Stage 2", "Stage 3"]
# classes = ["A", "B", "C", "D"]
# stage_logits = {stage: {cls: [] for cls in classes} for stage in stages}

# # 데이터 추출
# for doc_id, logits_list in logits_data.items():
#     for stage_info in logits_list:
#         stage_name = stage_info["Stage"]
#         logits = stage_info["Logits"]
#         for cls in classes:
#             stage_logits[stage_name][cls].append(logits[cls])

# # 로짓 분포 시각화
# output_dir = "/workspace/vlm/lmms-eval-logit/figs/logit_dist"
# os.makedirs(output_dir, exist_ok=True)

# for stage in stages:
#     plt.figure(figsize=(10, 6))
#     for cls in classes:
#         plt.hist(stage_logits[stage][cls], bins=30, alpha=0.6, label=f"Class {cls}")
#     plt.title(f"Logit Distribution - {stage}")
#     plt.xlabel("Logit Value")
#     plt.ylabel("Frequency")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(output_dir, f"logit_distribution_{stage.replace(' ', '_')}.png"))
#     plt.close()

# print(f"Logit distributions saved in {output_dir}")

# import json
# import matplotlib.pyplot as plt
# import os

# # 파일 경로 설정
# logit_file_path = "/workspace/vlm/lmms-eval-logit/logit_mmstar_100_30_70_bilinear_s1.json"
# response_file_path = "/workspace/vlm/lmms-eval-logit/logs/liuhaotian__llava-v1.6-vicuna-7b/20250109_024012_samples_mmstar.jsonl"
# output_dir = "/workspace/vlm/lmms-eval-logit/figs/logit_dist/logit_distributions_hallucination"
# os.makedirs(output_dir, exist_ok=True)

# # JSON 데이터 로드
# with open(logit_file_path, "r") as f:
#     logits_data = json.load(f)

# with open(response_file_path, "r") as f:
#     response_data = [json.loads(line) for line in f]

# # Hallucination 여부를 기준으로 로짓 값 분류
# stages = ["Stage 0", "Stage 1", "Stage 2", "Stage 3"]
# classes = ["A", "B", "C", "D"]
# hallucination_logits = {stage: {cls: [] for cls in classes} for stage in stages}
# non_hallucination_logits = {stage: {cls: [] for cls in classes} for stage in stages}

# # 데이터 추출 및 분류
# for response in response_data:
#     doc_id = f"({response['doc_id']},)"
#     target_answer = response["doc"]["answer"]
#     predicted_answer = response["filtered_resps"][0]  # 예측 값
#     is_hallucination = predicted_answer != target_answer  # 할루시네이션 여부

#     if doc_id in logits_data:
#         logits_list = logits_data[doc_id]
#         for stage_info in logits_list:
#             stage_name = stage_info["Stage"]
#             logits = stage_info["Logits"]
#             for cls in classes:
#                 if is_hallucination:
#                     hallucination_logits[stage_name][cls].append(logits[cls])
#                 else:
#                     non_hallucination_logits[stage_name][cls].append(logits[cls])

# # 로짓 분포 시각화
# def plot_logits(logit_dict, title_prefix, output_dir):
#     for stage in stages:
#         plt.figure(figsize=(10, 6))
#         for cls in classes:
#             plt.hist(logit_dict[stage][cls], bins=30, alpha=0.6, label=f"Class {cls}")
#         plt.title(f"{title_prefix} Logit Distribution - {stage}")
#         plt.xlabel("Logit Value")
#         plt.ylabel("Frequency")
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(os.path.join(output_dir, f"{title_prefix.lower()}_logit_distribution_{stage.replace(' ', '_')}.png"))
#         plt.close()

# # 할루시네이션 및 비할루시네이션 분포 그리기
# plot_logits(hallucination_logits, "Hallucination", output_dir)
# plot_logits(non_hallucination_logits, "Non-Hallucination", output_dir)

# print(f"Logit distributions for hallucination and non-hallucination cases saved in {output_dir}")

import json
import matplotlib.pyplot as plt
import os

# 파일 경로 설정
logit_file_path = "/workspace/vlm/lmms-eval-logit/logit_mmstar_100_30_70_bilinear_s1.json"
response_file_path = "/workspace/vlm/lmms-eval-logit/logs/liuhaotian__llava-v1.6-vicuna-7b/20250109_024012_samples_mmstar.jsonl"
output_dir = "/workspace/vlm/lmms-eval-logit/stage_vs_logits"
os.makedirs(output_dir, exist_ok=True)

# JSON 데이터 로드
with open(logit_file_path, "r") as f:
    logits_data = json.load(f)

with open(response_file_path, "r") as f:
    response_data = [json.loads(line) for line in f]

# 할루시네이션 여부에 따라 로짓 값 분류
stages = ["Stage 0", "Stage 1", "Stage 2", "Stage 3"]
classes = ["A", "B", "C", "D"]
hallucination_logits = {stage: {cls: [] for cls in classes} for stage in stages}
non_hallucination_logits = {stage: {cls: [] for cls in classes} for stage in stages}

# 데이터 추출 및 분류
for response in response_data:
    doc_id = f"({response['doc_id']},)"
    target_answer = response["doc"]["answer"]
    predicted_answer = response["filtered_resps"][0]  # 예측 값
    is_hallucination = predicted_answer != target_answer  # 할루시네이션 여부

    if doc_id in logits_data:
        logits_list = logits_data[doc_id]
        for stage_info in logits_list:
            stage_name = stage_info["Stage"]
            logits = stage_info["Logits"]
            for cls in classes:
                if is_hallucination:
                    hallucination_logits[stage_name][cls].append(logits[cls])
                else:
                    non_hallucination_logits[stage_name][cls].append(logits[cls])

# 평균 로짓 계산
def calculate_mean_logits(logit_dict):
    mean_logits = {stage: {cls: 0 for cls in classes} for stage in stages}
    for stage in stages:
        for cls in classes:
            if len(logit_dict[stage][cls]) > 0:
                mean_logits[stage][cls] = sum(logit_dict[stage][cls]) / len(logit_dict[stage][cls])
    return mean_logits

hallucination_mean_logits = calculate_mean_logits(hallucination_logits)
non_hallucination_mean_logits = calculate_mean_logits(non_hallucination_logits)

# 시각화
def plot_stage_vs_logits(mean_logits, title_prefix, output_dir):
    for cls in classes:
        plt.figure(figsize=(10, 6))
        x = list(range(len(stages)))  # Stage index
        y = [mean_logits[stage][cls] for stage in stages]
        plt.plot(x, y, marker='o', label=f"Class {cls}")
        plt.title(f"{title_prefix}: Class {cls}")
        plt.xlabel("Stage")
        plt.ylabel("Average Logit")
        plt.xticks(x, stages)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{title_prefix.lower()}_class_{cls}_logit_trend.png"))
        plt.close()

# 할루시네이션 및 비할루시네이션 그래프 그리기
plot_stage_vs_logits(hallucination_mean_logits, "Hallucination", output_dir)
plot_stage_vs_logits(non_hallucination_mean_logits, "Non-Hallucination", output_dir)

print(f"Plots saved in {output_dir}")
