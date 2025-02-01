import json
from collections import defaultdict
from itertools import product

# 파일 경로 설정
logit_file_path = "/workspace/vlm/lmms-eval-logit/logit_mmbench_en_dev_lite_100_50_30_bilinear_s1.json"
response_file_path = "/workspace/vlm/lmms-eval-jake2/logs/liuhaotian__llava-v1.6-vicuna-7b/20250108_203213_samples_mmbench_en_dev_lite.jsonl"
output_path_grid_search = "/workspace/vlm/lmms-eval-logit/mmbench_grid_search_results.json"

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

# 결과 집계 함수
def calculate_hit_rates(results):
    """
    Calculate overall, category-based, and l2-category-based hit rates.
    Args:
        results: a list of dictionaries with keys "score" and "l2_category".
    Returns:
        overall_score: float
        category_scores: dict
        l2_category_scores: dict
    """
    overall_score = sum(result["score"] for result in results) / len(results) if results else 0.0

    # Category-based hit rates
    category_scores = defaultdict(list)
    l2_category_scores = defaultdict(list)
    for result in results:
        score = result["score"]
        l2_category = result["l2_category"]
        if l2_category:  # Ensure l2_category is not None
            l2_category_scores[l2_category].append(score)

    # Calculate average scores for each category
    l2_category_avg_scores = {cat: sum(scores) / len(scores) for cat, scores in l2_category_scores.items()}

    return overall_score, l2_category_avg_scores

# 데이터 로드
with open(logit_file_path, "r") as f:
    logits_data = json.load(f)

with open(response_file_path, "r") as f:
    response_data = [json.loads(line) for line in f]

# 그리드 서치 설정
grid_values = [i / 2.0 for i in range(5)]  # -1.0부터 1.0까지 0.05 간격
# grid_values = [0,0,0]
best_params = {"alpha": 0, "beta": 0, "gamma": 0, "average_score": 0}
all_results = []

# 그리드 서치 실행
for alpha, beta, gamma in product(grid_values, repeat=3):
    results = []
    for response in response_data:
        doc_id = f"({response['doc_id']},)"  # Format doc_id as tuple string
        target_answer = response["target"]
        l2_category = response["doc"].get("l2_category")
        if str(doc_id) in logits_data:
            logits = logits_data[str(doc_id)]
            adjusted_logits = calculate_adjusted_logits(logits, alpha, beta, gamma)
            predicted_answer = max(adjusted_logits, key=adjusted_logits.get)

            # 정확도 비교
            score = exact_match_extended(predicted_answer, target_answer)
            results.append({"score": score, "l2_category": l2_category})

    # 평균 점수 계산
    overall_score, l2_category_scores = calculate_hit_rates(results)
    all_results.append({"alpha": alpha, "beta": beta, "gamma": gamma, "average_score": overall_score})

    # 최적의 매개변수 업데이트
    if overall_score > best_params["average_score"]:
        best_params = {"alpha": alpha, "beta": beta, "gamma": gamma, "average_score": overall_score}
        print(f"New best params: {best_params}")

# 결과 저장
output_data = {"best_params": best_params, "all_results": all_results}
with open(output_path_grid_search, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Grid search completed. Best params: {best_params}")