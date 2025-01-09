import json
from collections import defaultdict
from itertools import product

# 파일 경로 설정
logit_file_path = "/workspace/vlm/lmms-eval-logit/logit_mmstar_100_30_70_bilinear_s1.json"
response_file_path = "/workspace/vlm/lmms-eval-logit/logs/liuhaotian__llava-v1.6-vicuna-7b/20250109_024012_samples_mmstar.jsonl"
output_path_grid_search = "/workspace/vlm/lmms-eval-logit/mmstar_100_30_70_grid_search_results.json"

# 정확도 매칭 함수
def exact_match_extended(pred, gt):
    """Extended exact match for descriptive options."""
    answer = gt.lower().strip().replace("\n", " ")
    predict = pred.lower().strip().replace("\n", " ")
    try:
        if answer == predict[0]:
            return 1.0
        elif predict[0] == "(" and answer == predict[1]:
            return 1.0
        elif predict[0:7] == "option " and answer == predict[7]:
            return 1.0
        elif predict[0:14] == "the answer is " and answer == predict[14]:
            return 1.0
    except Exception:
        return 0.0
    return 0.0

# 로짓 계산
def calculate_adjusted_logits(logits, alpha, beta, gamma):
    adjusted_logits = {}
    for label in logits[0]["Logits"].keys():
        adjusted_logits[label] = (
            logits[3]["Logits"][label]
            + alpha * (logits[3]["Logits"][label] - logits[2]["Logits"][label])
            + beta * (logits[2]["Logits"][label] - logits[1]["Logits"][label])
            + gamma * (logits[1]["Logits"][label] - logits[0]["Logits"][label])
        )
    return adjusted_logits

# 결과 집계 함수
def mmstar_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score and category-level scores
    """
    l2_category_scores = defaultdict(list)
    for result in results:
        score = result["score"]
        l2_category = result["l2_category"]
        l2_category_scores[l2_category].append(score)

    # 각 카테고리 평균 계산
    l2_category_avg_score = {l2: sum(scores) / len(scores) for l2, scores in l2_category_scores.items()}

    # 전체 평균 계산
    avg_score = sum(l2_category_avg_score.values()) / len(l2_category_avg_score) 
    return avg_score, l2_category_avg_score

# 데이터 로드
with open(logit_file_path, "r") as f:
    logits_data = json.load(f)

with open(response_file_path, "r") as f:
    response_data = [json.loads(line) for line in f]

# 그리드 서치 설정
grid_values = [i / 2.0 for i in range(5)]  # -1.0부터 1.0까지 0.05 간격
best_params = {"alpha": 0, "beta": 0, "gamma": 0, "average_score": 0, "category_scores": {}}
all_results = []

# 그리드 서치 실행
for alpha, beta, gamma in product(grid_values, repeat=3):
    results = []
    for response in response_data:
        doc_id = f"({response['doc_id']},)"  # Format doc_id as tuple string
        target_answer = response["doc"]["answer"]
        l2_category = response["doc"].get("l2_category")
        if str(doc_id) in logits_data:
            logits = logits_data[str(doc_id)]
            adjusted_logits = calculate_adjusted_logits(logits, alpha, beta, gamma)
            predicted_answer = max(adjusted_logits, key=adjusted_logits.get)
            # last_stage_logits = logits[-1]["Logits"]
            # predicted_answer = max(last_stage_logits, key=last_stage_logits.get)
            # 정확도 비교
            # print(predicted_answer,target_answer)
            score = exact_match_extended(predicted_answer, target_answer)
            # print(predicted_answer,target_answer,score)
            results.append({"score": score, "l2_category": l2_category})

    # 평균 점수 및 카테고리 점수 계산
    avg_score, l2_category_scores = mmstar_aggregate_results(results)
    all_results.append({
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "average_score": avg_score,
        "category_scores": l2_category_scores
    })

    # 최적의 매개변수 업데이트
    if avg_score > best_params["average_score"]:
        best_params = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "average_score": avg_score,
            "category_scores": l2_category_scores
        }
        print(f"New best params: {best_params}")

# 결과 저장
output_data = {"best_params": best_params, "all_results": all_results}
with open(output_path_grid_search, "w") as f:
    json.dump(output_data, f, indent=4)

# print(f"Grid search completed. Best params: {best_params}")

# import json

# # 파일 경로 설정
# logit_file_path = "/workspace/vlm/lmms-eval-logit/logit_mmstar_100_30_70_bilinear_s1.json"

# # 데이터 로드
# with open(logit_file_path, "r") as f:
#     logits_data = json.load(f)

# # 불일치 결과 저장
# discrepancies = []

# # 불일치 확인
# for doc_id, stages in logits_data.items():
#     # 마지막 스테이지 로짓 값과 Text Output 가져오기
#     last_stage = stages[-1]
#     text_output = last_stage["Text Output"]
#     last_stage_logits = last_stage["Logits"]

#     # 로짓에서 가장 높은 값을 가진 선택지
#     predicted_from_logits = max(last_stage_logits, key=last_stage_logits.get)

#     # 불일치 확인
#     if text_output != predicted_from_logits:
#         discrepancies.append({
#             "doc_id": doc_id,
#             "text_output": text_output,
#             "predicted_from_logits": predicted_from_logits,
#             "last_stage_logits": last_stage_logits
#         })

# # 불일치 결과 출력
# print(f"Number of discrepancies: {len(discrepancies)}")
# if discrepancies:
#     for idx, discrepancy in enumerate(discrepancies[:10]):  # 최대 10개만 출력
#         print(f"\nDiscrepancy {idx + 1}:")
#         print(f"  Doc ID: {discrepancy['doc_id']}")
#         print(f"  Text Output: {discrepancy['text_output']}")
#         print(f"  Predicted from Logits: {discrepancy['predicted_from_logits']}")
#         print(f"  Last Stage Logits: {discrepancy['last_stage_logits']}")

# # 결과 저장 (옵션)
# output_path = "/workspace/vlm/lmms-eval-logit/discrepancies.json"
# with open(output_path, "w") as f:
#     json.dump(discrepancies, f, indent=4)

# print(f"Discrepancy results saved to {output_path}")