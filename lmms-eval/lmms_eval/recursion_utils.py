import numpy as np
from scipy.spatial.distance import cosine
import torch
import torch.nn.functional as F
import csv
from pathlib import Path

def calculate_entropy_for_attn_threshold(attn_map):
    flattened_attn = attn_map.view(-1)
    flattened_attn = flattened_attn / flattened_attn.sum()
    
    log_probs = torch.log(flattened_attn + 1e-8)
    
    # Calculate Entropy
    entropy = -torch.sum(flattened_attn * log_probs)
    return entropy.item()

def entropy_based_threshold(attn_map, base_threshold=0.2, scaling_factor=2, max_entropy=6.0):
    # Calculate Entropy
    entropy = calculate_entropy_for_attn_threshold(attn_map)
    normalized_entropy = min(entropy / max_entropy, 1.0)
    
    # Use a sensitive scaling function to make normalized_entropy more impactful
    sensitive_entropy = torch.exp(torch.tensor(normalized_entropy * scaling_factor) - 1)  # Shift to center around 1
    
    print(f'Calculated Sensitive Entropy: {sensitive_entropy}')  
    
    # Modify the threshold based on the sensitive entropy
    threshold = base_threshold * (1 - 0.5 * sensitive_entropy)  # Smaller factor for smaller changes
    threshold = min(max(threshold, 0.1), 0.9)  # Ensure it stays within range
    return threshold

def confidence_based_threshold(cumulative_confidences, base_threshold=0.2):
    # Take the last value from cumulative_confidences as the final confidence
    final_confidence = cumulative_confidences[-1]

    # Use exponential scaling for sensitivity
    sensitive_confidence = torch.exp(torch.tensor(final_confidence))  # Center around 1
    
    print(f'Calculated Sensitive Confidence: {sensitive_confidence}')  
    
    # Modify the threshold based on the sensitive confidence
    threshold = base_threshold * sensitive_confidence  # Increase threshold as confidence rises
    threshold = min(max(threshold, 0.1), 0.9)  # Ensure threshold stays within range
    return threshold

def calculate_entropy_and_all_confidences(sequence, scores):
    """
    Calculate entropy, full sequence confidence, and per-token cumulative confidences.
    Args:
        sequence (torch.Tensor): Generated sequence of token IDs.
        scores (list of torch.Tensor): List of logit tensors, one for each token in the sequence.

    Returns:
        tuple: Full sequence confidence, entropy, and a list of per-token cumulative confidences.
    """
    log_prob_sum_full = 0.0  # Log-probability sum for calculating full sequence confidence
    entropy_sum = 0.0  # Sum of entropies
    cumulative_confidences = []  # List to store cumulative confidence up to each token

    for idx, token_id in enumerate(sequence):
        probs = F.softmax(scores[idx], dim=-1)  # Softmax to get probabilities for the current token
        token_prob = probs[0, token_id].item()  # Probability of the actual token

        # Update cumulative log probability for the full sequence up to this token
        log_prob_sum_full += np.log(token_prob + 1e-10)
        # Calculate and store cumulative confidence for this subsequence
        cumulative_confidences.append(np.exp(log_prob_sum_full))
        
        # Entropy calculation for the token
        entropy_sum -= token_prob * np.log(token_prob + 1e-10)

    # Full sequence confidence (cumulative up to the last token)
    P_T_given_I_Q_full = cumulative_confidences[-1] if cumulative_confidences else np.exp(log_prob_sum_full)

    # print(f"Overall confidence P(T | I, Q): {P_T_given_I_Q_full}")
    # print(f"Entropy H(T | I, Q): {entropy_sum}")
    # print("Per-token cumulative confidences:", cumulative_confidences)

    return P_T_given_I_Q_full, entropy_sum, cumulative_confidences

#### Recursion Methods ####

def layer_mean_based_recursion(attn = None, attn_threshold = 0.1 , image_mask = None):
    for row in range(attn.shape[0]):
        for col in range(attn.shape[1]):
            if attn[row, col] > attn_threshold:
                image_mask[row][col] = 1
    
    return image_mask

def layer_mean_topk_based_recursion(attn = None, top_k = 0.1, image_mask = None):
    flattened_attn = attn.view(-1) 
    flattened_attn = flattened_attn.float()
    threshold_index = int(len(flattened_attn) * (top_k)) 
    threshold_value = torch.topk(flattened_attn, threshold_index).values[-1]

    for row in range(attn.shape[0]):
        for col in range(attn.shape[1]):
            if attn[row, col] >= threshold_value:
                image_mask[row][col] = 1

    return image_mask

def confidence_topk_based_recursion(attn = None, top_k = 0.1, sequences = None, scores= None, image_mask = None): 
    _, _, cumulative_confidences = calculate_entropy_and_all_confidences(sequence = sequences , scores = scores)
                                                                         
    calculated_threshold = confidence_based_threshold(cumulative_confidences= cumulative_confidences, base_threshold=top_k)
    flattened_attn = attn.view(-1).float()
    threshold_index = int(len(flattened_attn) * (calculated_threshold))
    threshold_value = torch.topk(flattened_attn, threshold_index).values[-1]

    for row in range(attn.shape[0]):
        for col in range(attn.shape[1]):
            if attn[row, col] >= threshold_value:
                image_mask[row][col] = 1
    
    return image_mask