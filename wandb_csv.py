import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the CSV file
file_path = "/workspace/vlm/lmms-eval-jake2/wandb_export_2025-01-01T17_43_27.271+09_00.csv"  # Replace with the path to your CSV file
output_dir = "/workspace/vlm/lmms-eval-jake2/figs"  # Directory to save the plots
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

df = pd.read_csv(file_path)

# Extract threshold values from the experiment names
df[['stage_minus1', 'stage_0', 'stage_1']] = df['Name'].str.extract(
    r'exp_run_\d+_thresh_([0-9.]+)_([0-9.]+)_([0-9.]+)'
).astype(float)

# List of tasks to analyze
tasks = {
    "pope_pop/pope_f1_score": "Pope F1 Score",
    "mme/mme_percetion_score": "MME Perception Score",
    "vqav2_val_lite/exact_match": "VQA Exact Match"
}

# Analyze trends
for task_col, task_name in tasks.items():
    # Check if the task column exists
    if task_col not in df.columns:
        print(f"Column {task_col} not found in the dataset. Skipping...")
        continue

    print(f"\nAnalyzing trends for {task_name}...")
    
    # Group by each stage's threshold and compute the mean performance
    for stage in ['stage_minus1', 'stage_0', 'stage_1']:
        stage_means = df.groupby(stage)[task_col].mean()
        
        # Print the results
        print(f"\nPerformance trends for {task_name} w.r.t. {stage}:")
        print(stage_means)
        
        # Plotting and saving
        plt.figure()
        plt.plot(stage_means.index, stage_means.values, marker='o')
        plt.title(f"{task_name} vs {stage}")
        plt.xlabel(f"{stage} Threshold")
        plt.ylabel(task_name)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{task_name.replace(' ', '_')}_vs_{stage}.png"))
        plt.close()  # Close the plot to save memory