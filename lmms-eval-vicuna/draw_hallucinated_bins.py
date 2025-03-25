import pandas as pd
import ast
import re
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Function to extract numbers from cumulative confidence values
def extract_confidences(confidence_list):
    cleaned_list = []
    for confidence in confidence_list:
        if isinstance(confidence, (int, float)):
            cleaned_list.append(float(confidence))
        elif isinstance(confidence, str):
            # Handle "tensor(...)" strings
            match = re.search(r"tensor\\(([0-9.]+)\\)", confidence)
            if match:
                cleaned_list.append(float(match.group(1)))
            else:
                try:
                    cleaned_list.append(float(confidence))
                except ValueError:
                    pass
    return cleaned_list

def process_file(file_path):
    df = pd.read_csv(file_path)

    # Convert 'Cumulative Confidences' to lists
    df['Cumulative Confidences'] = df['Cumulative Confidences'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # Extract numerical values
    df['Cumulative Confidences'] = df['Cumulative Confidences'].apply(extract_confidences)

    # Mark hallucination based on the Score
    df['No Hallucination'] = df['Score'] == 1.0

    return df

def prepare_data(dfs):
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    bin_labels = [f"{bins[i]}~{bins[i+1]}" for i in range(len(bins) - 1)]
    combined_data = []

    for df in dfs:
        hallucinated_probs = [
            1 - conf for confs in df[~df['No Hallucination']]['Cumulative Confidences'] for conf in confs
        ]
        non_hallucinated_probs = [
            1 - conf for confs in df[df['No Hallucination']]['Cumulative Confidences'] for conf in confs
        ]

        total_hallucinated = len(hallucinated_probs)
        total_non_hallucinated = len(non_hallucinated_probs)

        if total_hallucinated > 0:
            hallucinated_counts = pd.cut(
                hallucinated_probs, bins=bins, labels=bin_labels, right=False, include_lowest=True
            ).value_counts()
            for prob_range, count in hallucinated_counts.items():
                combined_data.append({
                    "Hallucination Probability Range": prob_range,
                    "Count": count,
                    "Total": total_hallucinated,
                    "Ratio": count / total_hallucinated / len(dfs),  # Normalize by total and average across datasets
                    "Type": "Hallucinated"
                })

        if total_non_hallucinated > 0:
            non_hallucinated_counts = pd.cut(
                non_hallucinated_probs, bins=bins, labels=bin_labels, right=False, include_lowest=True
            ).value_counts()
            for prob_range, count in non_hallucinated_counts.items():
                combined_data.append({
                    "Hallucination Probability Range": prob_range,
                    "Count": count,
                    "Total": total_non_hallucinated,
                    "Ratio": count / total_non_hallucinated / len(dfs),  # Normalize by total and average across datasets
                    "Type": "Non-Hallucinated"
                })

    return pd.DataFrame(combined_data)

def plot_data(data, output_path):
    # Prepare bins and types
    bins = data['Hallucination Probability Range'].drop_duplicates().sort_values()
    types = data['Type'].unique()
    x = np.arange(len(bins))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot bars for hallucinated and non-hallucinated
    for i, data_type in enumerate(types):
        type_data = data[data['Type'] == data_type]
        type_ratios = type_data.groupby('Hallucination Probability Range')['Ratio'].sum().reindex(bins, fill_value=0)
        ax.bar(
            x + (i - 0.5) * width,  # Offset for hallucinated and non-hallucinated
            type_ratios.values,
            width=width,
            label=data_type,
            alpha=0.7
        )

    # Increase font sizes
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=24)  # Increase font size for x-axis tick labels
    ax.set_xlabel("Hallucination Probability Range", fontsize=24)  # Increase font size for x-axis label
    ax.set_ylabel("Ratio", fontsize=24)  # Increase font size for y-axis label
    ax.legend(fontsize=24)  # Increase font size for legend

    plt.xticks(fontsize=20)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=20)  # Increase font size for y-axis ticks
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Process and analyze hallucination probability data.")
    parser.add_argument("--INPUT_PATHS", nargs=3, required=True, help="Paths to the three input CSV files")
    parser.add_argument("--OUTPUT_PATH", default="combined_figure.png", help="Path to save the combined figure")
    args = parser.parse_args()

    # Load and process each input file
    dfs = [process_file(file_path) for file_path in args.INPUT_PATHS]

    # Prepare data
    combined_data = prepare_data(dfs)

    # Plot data
    plot_data(combined_data, args.OUTPUT_PATH)

if __name__ == "__main__":
    main()
