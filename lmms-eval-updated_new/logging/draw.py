import matplotlib.pyplot as plt
import os

# 데이터
hyp = [0.5, 1.0, 2.5, 5.0, 10.0]
f1_score = [87.4, 87.3, 87.5, 87.2, 85.9,]
baseline = 86.5  # 베이스라인 성능

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(hyp, f1_score, marker='o', linestyle='-', color='tab:blue', label='F1 Score')

# 베이스라인 성능 추가
plt.axhline(y=baseline, color='red', linestyle='--', label='Baseline (86.5)')

# 라벨 설정
plt.xlabel('Hyp', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)
plt.title('Relationship between Hyp and F1 Score', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.xticks(hyp, fontsize=10)
plt.yticks(fontsize=10)

# 그래프 저장
output_path = os.path.join('/workspace/vlm/lmms-eval-updated_new/logging/sensitivity', 'attn_pope_sensitivity.png')
plt.savefig(output_path, dpi=300)
plt.close()