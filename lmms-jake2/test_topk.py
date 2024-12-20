import torch

# Example data for testing
data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float32)

# K 값 설정 (상위 10%를 남기려면 K=0.1)
top_k_percent = 0.1  # Adjust this to test other percentages

# Threshold Index 계산
threshold_index = int(len(data) * (1 - top_k_percent))
print(f"Threshold Index: {threshold_index}")

# Threshold Value 계산
threshold_value = torch.topk(data, threshold_index).values[-1]
print(f"Threshold Value: {threshold_value}")

# 데이터 필터링 (조건: 값이 threshold를 초과하는 경우)
filtered_data = data[data > threshold_value]
print("Filtered Data (Values > Threshold):", filtered_data.tolist())