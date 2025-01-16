import json
from collections import defaultdict
from itertools import product
import pandas as pd
import numpy as np
from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator
from lmms_eval.tasks.mmbench.en_utils import mmbench_aggregate_dev_results_eval, mmbench_process_results
from lmms_eval.tasks.mmstar.utils import mmstar_process_results, mmstar_aggregate_results
import argparse

parser = argparse.ArgumentParser(description="Script to handle output path argument.")    

parser.add_argument(
    "--logit_file_path",
    type=str,
    required=True,  # Set to True if you want the argument to be mandatory
    help="Path to save the output file."
)

parser.add_argument(
    "--response_file_path",
    type=str,
    required=True,  # Set to True if you want the argument to be mandatory
    help="Path to save the output file."
)

parser.add_argument(
    "--output_path_grid_search",
    type=str,
    required=True,  # Set to True if you want the argument to be mandatory
    help="Path to save the output file."
)

parser.add_argument(
    "--mode",
    type=str,
    required=True,  # Set to True if you want the argument to be mandatory
    help="Path to save the output file."
)

parser.add_argument(
    "--min",
    type=float,
    default=0.0    
)

parser.add_argument(
    "--max",
    type=float,
    default=2.5    
)

parser.add_argument(
    "--step",
    type=float,
    default=0.1   
)

args = parser.parse_args()
args.output_path = "./"

# 파일 경로 설정

## mmbench
# response_file_path = "/workspace/vlm/lmms-eval-logit/logs/liuhaotian__llava-v1.6-vicuna-7b/20250109_032749_samples_mmbench_en_dev_lite.jsonl"
# mode = "mmbench"

## mmstar
# response_file_path = "/workspace/vlm/lmms-eval-logit/logs/liuhaotian__llava-v1.6-vicuna-7b/20250109_010426_samples_mmstar.jsonl"
# mode = "mmstar"

## pope
# response_file_path = "/workspace/vlm/lmms-eval-logit/logs/liuhaotian__llava-v1.6-vicuna-7b/20250109_075904_samples_pope_pop.jsonl"
# mode = "pope"

## pope_gqa
# response_file_path = "/workspace/vlm/lmms-eval-logit/logs/liuhaotian__llava-v1.6-vicuna-7b/20250109_095358_samples_pope_gqa_pop.jsonl"
# mode = "pope_gqa"

## pope_aokvqa
# response_file_path = "/workspace/vlm/lmms-eval-logit/logs/liuhaotian__llava-v1.6-vicuna-7b/20250109_064354_samples_pope_aokvqa_pop.jsonl"
# mode = "pope_aokvqa"

# logit_file_path = "/workspace/vlm/lmms-eval-updated_new/logging/cd_grid_mmbench_2.0_2.0_2.0.json"
# output_path_grid_search = "/workspace/vlm/lmms-eval-updated_new/logging/mmbench_2.0_2.0_2.0_cd_adaptive.json"

logit_file_path = args.logit_file_path
response_file_path = args.response_file_path
output_path_grid_search = args.output_path_grid_search
mode = args.mode

if mode == "pope":
    from lmms_eval.tasks.pope.utils import pope_process_results, pope_aggregate_f1_score
elif mode == "pope_gqa":
    from lmms_eval.tasks.pope_gqa.utils import pope_process_results, pope_aggregate_f1_score
elif mode == "pope_aokvqa":
    from lmms_eval.tasks.pope_aokvqa.utils import pope_process_results, pope_aggregate_f1_score

# 데이터 로드
with open(logit_file_path, "r") as f:
    logits_data = json.load(f)

with open(response_file_path, "r") as f:
    response_data = [json.loads(line) for line in f]

def eval_metric_mmbench(response_data, logits_data, alpha, beta, gamma):
    results = []

    for data in response_data:
        doc = data['doc']
        doc_id = f"({data['doc_id']},)"
        logits = logits_data[str(doc_id)]

        updated_pred = get_prediction_cd(logits, alpha, beta, gamma)
        result = [updated_pred]

        processed_results = mmbench_process_results(doc, result)
        results.append(processed_results['gpt_eval_score'])    
    
    final_result = mmbench_aggregate_dev_results_eval(results, args)
    #print(final_result)

    return final_result

def eval_metric_mmstar(response_data, logits_data, alpha, beta, gamma):
    results = []

    for data in response_data:
        doc = data['doc']
        doc_id = f"({data['doc_id']},)"
        logits = logits_data[str(doc_id)]

        updated_pred = get_prediction_cd(logits, alpha, beta, gamma)
        result = [updated_pred]

        processed_results = mmstar_process_results(doc, result)
        results.append(processed_results['average'])    
    
    # print(results[0])
    
    final_result = mmstar_aggregate_results(results)
    #print(final_result)

    return final_result

def eval_metric_pope(response_data, logits_data, alpha, beta, gamma):
    results = []

    for data in response_data:
        doc = data['doc']
        doc_id = f"({data['doc_id']},)"
        logits = logits_data[str(doc_id)]

        updated_pred = get_prediction_cd(logits, alpha, beta, gamma)
        result = [updated_pred]

        processed_results = pope_process_results(doc, result)
        results.append(processed_results["pope_f1_score"])    
    
    # print(results[0])
    
    final_result = pope_aggregate_f1_score(results)
    #print(final_result)

    return final_result

def get_prediction_cd(logits, alpha, beta, gamma):    
    adjusted_logits = calculate_adjusted_logits(logits, alpha, beta, gamma)
    predicted_answer = max(adjusted_logits, key=adjusted_logits.get)

    return predicted_answer

def calculate_adjusted_logits(logits, alpha, beta, gamma):
    stage_diff = []

    # 각 Stage에서 1등과 2등 로짓의 차이 계산
    for stage in range(4):
        sorted_logits = sorted(logits[stage]["Top-100 Logits"], key=lambda x: x[1], reverse=True)
        stage_diff.append(sorted_logits[0][1] - sorted_logits[1][1])  # 1등 - 2등 차이

    # 각 단계의 가중치 계산 (1등-2등 차이에 반비례)
    weights = 1 / (stage_diff[-1] + 1e-8)
    # Default
    # weights = 1
    adjusted_logits = {}

    # 각 라벨에 대해 adjusted logits 계산
    for label, _ in logits[0]["Top-100 Logits"]:
        adjusted_logits[label] = (
            logits[3]["Logits"].get(label, 0)
            + weights * alpha * (logits[3]["Logits"].get(label, 0) - logits[2]["Logits"].get(label, 0))
            + weights * beta * (logits[2]["Logits"].get(label, 0) - logits[1]["Logits"].get(label, 0))
            + weights * gamma * (logits[1]["Logits"].get(label, 0) - logits[0]["Logits"].get(label, 0))
        )

    return adjusted_logits

# def calculate_adjusted_logits(logits,alpha, beta, gamma):
#     adjusted_logits = {}

#     # 각 Stage에서 1등과 2등 로짓의 차이 계산
#     stage_diff = []
#     for stage in range(4):
#         sorted_logits = sorted(logits[stage]["Top-100 Logits"].values(), reverse=True)
#         stage_diff.append(sorted_logits[0] - sorted_logits[1])  # 1등 - 2등 차이

#     # 각 단계의 가중치 계산 (1등-2등 차이에 반비례)
#     # 차이가 클수록 가중치는 작게, 차이가 작을수록 가중치는 크게 설정
#     weights = stage_diff[-1]
#     print(weights)
#     # 각 라벨에 대해 adjusted logits 계산
#     for label in logits[0]["Top-100 Logits"].keys():
#         adjusted_logits[label] = (
#             logits[3]["Top-100 Logits"][label]
#             + weights[3] * (logits[3]["Top-100 Logits"][label] - logits[2]["Top-100 Logits"][label])
#             + weights[2] * (logits[2]["Top-100 Logits"][label] - logits[1]["Top-100 Logits"][label])
#             + weights[1] * (logits[1]["Top-100 Logits"][label] - logits[0]["Top-100 Logits"][label])
#         )
#     print(weights[1], weights[2] , weights[3])
#     return adjusted_logits

# 그리드 서치 설정
# grid_values = [i / 20.0 for i in range(50)]  # -1.0부터 1.0까지 0.05 간격
grid_values = list(np.arange(args.min, args.max + args.step, args.step))
best_params = {"alpha": 0, "beta": 0, "gamma": 0, "score": 0}
all_results = []

# 그리드 서치 실행
for alpha, beta, gamma in product(grid_values, repeat=3):   
# Single CD
# for value in grid_values:  
#     alpha = beta = gamma = value 
    if mode == "mmstar":
        score = eval_metric_mmstar(response_data, logits_data, alpha, beta, gamma)
    elif mode == "mmbench":
        score = eval_metric_mmbench(response_data, logits_data, alpha, beta, gamma)
    elif "pope" in mode:
        score = eval_metric_pope(response_data, logits_data, alpha, beta, gamma)

    print(f"CD: {alpha}-{beta}-{gamma}: score={score}")

    all_results.append({
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "score": score,        
    })

    if score > best_params["score"]:
        best_params = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "score": score,           
        }
    
    print(f"Current best params: {best_params}")    

# 결과 저장
output_data = {"best_params": best_params, "all_results": all_results}
with open(output_path_grid_search, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Grid search completed. Best params: {best_params}")

# # 결과 저장을 위한 초기화
# best_score = 0
# all_results = []

# # 평가 루프
# if mode == "mmstar":
#     score = eval_metric_mmstar(response_data, logits_data)  # alpha, beta, gamma 제거
# elif mode == "mmbench":
#     score = eval_metric_mmbench(response_data, logits_data, 0,0,0)  # alpha, beta, gamma 제거
# elif "pope" in mode:
#     score = eval_metric_pope(response_data, logits_data)  # alpha, beta, gamma 제거

# print(f"Adaptive CD: score={score}")

# all_results.append({
#     "adaptive": True,  # adaptive 방식 표시
#     "score": score,
# })

# if score > best_score:
#     best_score = score

#     best_params = {
#         "adaptive": True,  # 최적화된 adaptive 방식 표시
#         "score": score,
#     }

# print(f"Best adaptive score: {best_score}")

# # 결과 저장
# output_data = {"best_params": best_params, "all_results": all_results}
# with open(output_path_grid_search, "w") as f:
#     json.dump(output_data, f, indent=4)

# print(f"Grid search (adaptive) completed. Best adaptive score: {best_score}")
