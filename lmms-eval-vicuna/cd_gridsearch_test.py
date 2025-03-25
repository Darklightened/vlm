import json
from collections import defaultdict
from itertools import product
import pandas as pd
import numpy as np
from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator
from lmms_eval.tasks.mmbench.en_utils import mmbench_aggregate_dev_results_eval, mmbench_process_results
from lmms_eval.tasks.mmstar.utils import mmstar_process_results, mmstar_aggregate_results
import argparse
import sys

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


# def get_prediction_cd(logits, alpha, beta, gamma):
#     # Define sorted token list
#     token_list = list(map(lambda x: x[0], logits[0]["Top-100 Logits"]))
    
#     # Extract tokens and logits for each stage
#     token_0 = list(map(lambda x: x[0], logits[0]["Top-100 Logits"]))
#     token_1 = list(map(lambda x: x[0], logits[1]["Top-100 Logits"]))
#     token_2 = list(map(lambda x: x[0], logits[2]["Top-100 Logits"]))
#     token_3 = list(map(lambda x: x[0], logits[3]["Top-100 Logits"]))
    
#     assert token_0 == token_1 == token_2 == token_3, "Token order is different"
    
#     logit_0 = list(map(lambda x: x[1], logits[0]["Top-100 Logits"]))
#     logit_1 = list(map(lambda x: x[1], logits[1]["Top-100 Logits"]))
#     logit_2 = list(map(lambda x: x[1], logits[2]["Top-100 Logits"]))
#     logit_3 = list(map(lambda x: x[1], logits[3]["Top-100 Logits"]))
    
#     # Calculate adjusted logits
#     adjusted_logits = np.array(logit_3)
#     adjusted_logits += alpha * (np.array(logit_3) - np.array(logit_2))
#     adjusted_logits += beta * (np.array(logit_2) - np.array(logit_1))
#     adjusted_logits += gamma * (np.array(logit_1) - np.array(logit_0))
    
#     # Get the index of the maximum logit
#     max_idx = np.argmax(adjusted_logits)
#     max_token = token_list[max_idx]
    
#     return max_token

def calculate_adjusted_logits(logits, alpha, beta, gamma):
    adjusted_logits = {}

    # Helper function to get logit value for a specific label
    def get_logit_from_top100(label, top_100_logits):
        for entry in top_100_logits:
            if entry[0] == label:  # Check if label matches
                return entry[1]  # Return the corresponding logit value
        return 0  # Default to 0 if label not found

    first_place = logits[3]["Top-100 Logits"][0][1]
    second_place = logits[2]["Top-100 Logits"][0][1]

    difference = first_place - second_place
    weight = 1 / (difference + 1e-8)
    weight = 1

    # Calculate adjusted logits for each label in the last stage
    for label, _ in logits[3]["Top-100 Logits"][:20]:
        adjusted_logits[label] = (
            get_logit_from_top100(label, logits[3]["Top-100 Logits"][:20])
            + alpha *weight* (
                get_logit_from_top100(label, logits[3]["Top-100 Logits"][:20])
                - get_logit_from_top100(label, logits[2]["Top-100 Logits"][:20])
            )
            + beta * weight*(
                get_logit_from_top100(label, logits[2]["Top-100 Logits"][:20])
                - get_logit_from_top100(label, logits[1]["Top-100 Logits"][:20])
            )
            + gamma * weight*(
                get_logit_from_top100(label, logits[1]["Top-100 Logits"][:20])
                - get_logit_from_top100(label, logits[0]["Top-100 Logits"][:20])
            )
        )

    return adjusted_logits

# def calculate_adjusted_logits(logits, alpha, beta, gamma):
#     adjusted_logits = {}

#     # 마지막 단계에서 1등과 2등의 로짓 차이 계산
#     difference = logits[3]["Top-100 Logits"][0][1] - logits[3]["Top-100 Logits"][1][1]
#     weight = 1 / (difference + 1e-8)  # 가중치 계산

#     # 마지막 단계의 라벨을 기준으로 adjusted logits 계산
#     for label, value_3 in logits[3]["Top-100 Logits"]:
#         # 이전 단계에서 동일한 라벨의 값을 찾음
#         value_2 = next((val for lbl, val in logits[2]["Top-100 Logits"] if lbl == label), 0)
#         value_1 = next((val for lbl, val in logits[1]["Top-100 Logits"] if lbl == label), 0)
#         value_0 = next((val for lbl, val in logits[0]["Top-100 Logits"] if lbl == label), 0)

#         # adjusted logits 계산
#         adjusted_logits[label] = (
#             value_3
#             + weight * alpha * (value_3 - value_2)
#             + weight * beta * (value_2 - value_1)
#             + weight * gamma * (value_1 - value_0)
#         )

#     return adjusted_logits

# 그리드 서치 설정
# grid_values = [i / 20.0 for i in range(50)]  # -1.0부터 1.0까지 0.05 간격
grid_values = list(np.arange(args.min, args.max + args.step, args.step))
best_params = {"alpha": 0, "beta": 0, "gamma": 0, "score": 0}
all_results = []

# 그리드 서치 실행
# Define baselines for each benchmark
baselines = {
    "Base1": {"pope": 0.865, "pope_gqa": 0.888, "pope_aokvqa": 0.863, "mmbench": 75.8},
    "Base2": {"pope": 0.875, "pope_gqa": 0.896, "pope_aokvqa": 0.863, "mmbench": 78.0},
}

# Initialize counters for settings that exceed baselines
baseline_exceed_count = {"Base1": 0, "Base2": 0}

# Update the evaluation loop to check scores against baselines
for alpha, beta, gamma in product(grid_values, repeat=3):   
    if mode == "mmstar":
        score = eval_metric_mmstar(response_data, logits_data, alpha, beta, gamma)
    elif mode == "mmbench":
        score = eval_metric_mmbench(response_data, logits_data, alpha, beta, gamma)
    elif "pope" in mode:
        score = eval_metric_pope(response_data, logits_data, alpha, beta, gamma)

    # print(f"CD: {alpha}-{beta}-{gamma}: score={score}")

    all_results.append({
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "score": score,        
    })
    
    if score > baselines["Base1"][mode]:
        baseline_exceed_count["Base1"] += 1
    if score > baselines["Base2"][mode]:
        baseline_exceed_count["Base2"] += 1

    if score > best_params["score"]:
        best_params = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "score": score,           
        }
    
    print(f"score={score}, Base1: {baseline_exceed_count['Base1']}, Base2: {baseline_exceed_count['Base2']}")    

# Print how many settings exceed baselines
print(f"Number of settings exceeding Base1: {baseline_exceed_count['Base1']}")
print(f"Number of settings exceeding Base2: {baseline_exceed_count['Base2']}")

# Save results to output file
output_data = {"best_params": best_params, "all_results": all_results, "baseline_exceed_count": baseline_exceed_count}
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
