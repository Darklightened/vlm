import json
import matplotlib.pyplot as plt
import pandas as pd
import os

# Replace 'data.json' with the path to your JSON file
with open('/workspace/vlm/lmms-eval-updated_new/logging/mmbench_en_dev_lite_2.0_2.0_2.0__cd_adaptive.json', 'r') as file:
    data = json.load(file)

# Extract relevant data
all_results = data['all_results']
df = pd.DataFrame(all_results)

# Add a new column for the sum of alpha, beta, and gamma
df['alpha_beta_gamma_sum'] = df['alpha'] + df['beta'] + df['gamma']

# Group by the sum of alpha, beta, and gamma and calculate the mean score
grouped = df.groupby('alpha_beta_gamma_sum')['score'].mean().reset_index()

# Create directory if it doesn't exist
output_dir = "/workspace/vlm/lmms-eval-updated_new/logging/stat_figs_ada"
os.makedirs(output_dir, exist_ok=True)

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.plot(grouped['alpha_beta_gamma_sum'], grouped['score'], marker='o')
plt.title('Score Distribution Across Alpha + Beta + Gamma Sum', fontsize=14)
plt.xlabel('Sum of Alpha, Beta, and Gamma', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.grid(True)

# Save the plot
output_path = os.path.join(output_dir, 'mmbench.png')
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Plot saved to {output_path}")