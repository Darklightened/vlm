import pandas as pd
import json
import argparse

def get_score_metric(mode, record):
    return {
        'pope': lambda: record['pope_accuracy']['score'],
        'mme': lambda: record['mme_percetion_score']['score'] if 'mme_percetion_score' in record and 'score' in record['mme_percetion_score'] else record['mme_cognition_score']['score'],
        'realworldqa': lambda: record['exact_match'],
        'mmstar': lambda: record['average']['score'],
        'scienceqa': lambda: record['exact_match'],
        'ai2d': lambda: record['exact_match'],
        'chartqa': lambda: record['relaxed_overall'],
    }.get(mode, lambda: "not implemented")()

# Helper function to find a key in a case-insensitive, underscore/dash-tolerant manner
def get_case_insensitive_key(record, key):
    normalized_key = key.lower().replace("_", "").replace("-", "")
    for k in record:
        if k.lower().replace("_", "").replace("-", "") == normalized_key:
            return record[k]
    return None

def main():
    parser = argparse.ArgumentParser(description="Match JSON and CSV data and extract relevant fields.")
    parser.add_argument("--LOG_FILE_PATH", required=True, help="Path to the JSONL log file")
    parser.add_argument("--CSV_FILE_PATH", required=True, help="Path to the CSV file")
    parser.add_argument("--MODE", required=True, help="Mode for score metric computation")
    parser.add_argument("--OUTPUT_FILE", default="matched_output.csv", help="Path to the output CSV file")
    args = parser.parse_args()

    # Load JSON data from a file
    with open(args.LOG_FILE_PATH, 'r') as f:
        json_data = [json.loads(line) for line in f]

    # Load CSV data
    csv_df = pd.read_csv(args.CSV_FILE_PATH)

    # Flatten cumulative confidence columns for easier matching
    csv_df["Cumulative Confidence"] = csv_df.apply(
        lambda row: [row[f"Cumulative Confidence {i+1}"] for i in range(10) if pd.notna(row.get(f"Cumulative Confidence {i+1}"))],
        axis=1
    )

    if 'Stage' in csv_df.columns and 'Doc ID' in csv_df.columns:
        if csv_df.columns.get_loc('Doc ID') > csv_df.columns.get_loc('Stage'):
            csv_df = csv_df.rename(columns={'Stage': 'Doc ID', 'Doc ID': 'Stage'})


    # Matching and extracting relevant fields
    matched_data = []
    for record in json_data:
        doc_id = record['doc_id']
        question = record['doc']['question']
        answer = record['target']
        ground_truth = record['target']
        prediction = record['filtered_resps']
        category = record['doc'].get('category', 'Unknown')

        # Handle l2_category and its variants
        l2_category = get_case_insensitive_key(record['doc'], 'l2_category') or 'Unknown'

        try:
            score = get_score_metric(args.MODE, record)
        except:
            continue

        # Filter CSV data for matching Doc ID
        stage_data = csv_df[csv_df["Doc ID"] == f"({doc_id},)"]

        # Separate stages for Stage 1 and Stage 2
        for _, row in stage_data.iterrows():
            stage_name = row["Stage"]
            text_output = row["Text Output"]
            cumulative_confidences = row["Cumulative Confidence"]

            # Add to matched data list
            matched_data.append({
                "Doc ID": doc_id,
                "Question": question,
                "Answer": answer,
                "Ground Truth": ground_truth,
                "Prediction": prediction,
                "Score": score,
                "Category": category,
                "L2 Category": l2_category,
                "Stage": stage_name,
                "Text Output": text_output,
                "Cumulative Confidences": cumulative_confidences
            })

    # Convert matched data to DataFrame and save to a new CSV
    matched_df = pd.DataFrame(matched_data)
    matched_df.to_csv(args.OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()
