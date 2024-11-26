def vqa_rad_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def vqa_rad_doc_to_text(doc):
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["question"].strip()
    return f"{question}\n"

def vqa_rad_process_results(doc, results):
    pred = results[0].lower().strip()
    gt_ans = doc["answer"].lower().strip()    
    score = 1.0 if gt_ans in pred else 0.0
    return {
        "vqa_rad_accuracy": {"score": score, "prediction": pred, "ground_truth": gt_ans},        
    }

def vqa_rad_aggregate_results(results):
    total_score = 0
    for result in results:
        total_score += result["score"]
    avg_score = total_score / len(results)
    return avg_score