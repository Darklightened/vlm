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

def confidence_based_threshold(cumulative_confidences, base_threshold=0.3):
    # Take the last value from cumulative_confidences as the final confidence
    final_confidence = cumulative_confidences[-1]

    # Use reciprocal scaling
    sensitive_confidence = 1 / (1 + final_confidence)  # Avoid division by zero
    
    # Modify the threshold
    threshold = base_threshold * sensitive_confidence  # Decrease threshold as confidence rises
    threshold = min(max(threshold, 0.0), 1.0)  # Ensure threshold stays within range
    print(f'Calculated Threshold: {threshold}')
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
    mask = (attn >= attn_threshold).to(torch.float16)
    # mask = torch.ones_like(attn)
    # if attn_threshold != 1.0:
    #     attn_nonzero = attn.flatten()
    #     attn_nonzero = attn_nonzero[attn_nonzero != 0]
    #     mean = torch.mean(attn_nonzero)
    #     std = torch.std(attn_nonzero)
    #     z_scores = (attn - mean) / std
    #     lower_threshold = -attn_threshold
    #     lower_bound = lower_threshold * std + mean
    #     mask = (attn >= lower_bound).to(torch.float16)
    
    return mask

def layer_mean_topk_based_recursion(attn = None, top_k = 0.1, image_mask = None):
    flattened_attn = attn.view(-1) 
    flattened_attn = flattened_attn.float()

    # nonzero_indices = torch.nonzero(flattened_attn).squeeze()
    # flattened_attn = flattened_attn[nonzero_indices]

    threshold_index = int(len(flattened_attn) * (top_k)) 
    threshold_value = torch.topk(flattened_attn, threshold_index).values[-1]

    image_mask = (attn >= threshold_value).to(torch.float16)
    
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

import torch

def calculate_attention_entropy(attn):
    """
    Calculate the entropy of the attention map.
    Args:
        attn (torch.Tensor): Attention map of shape (num_heads, seq_len, seq_len).
    Returns:
        float: Mean entropy across all attention heads.
    """
    # Normalize attention map along the last dimension (softmax-like normalization)
    attn_probs = attn / attn.sum(dim=-1, keepdim=True)

    # Avoid log(0) by adding a small epsilon
    eps = 1e-9
    attn_probs = attn_probs + eps

    # Compute entropy: -sum(p * log(p))
    entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)

    # Average entropy across all heads
    mean_entropy = entropy.mean().item()
    return mean_entropy


def attn_entropy_topk_based_recursion(attn=None, image_mask=None, base_top_k=0.3): 
    """
    Adjust Top-K threshold dynamically based on attention map entropy and record the entropy.
    """
    attn = torch.nn.functional.interpolate(
        attn.unsqueeze(0).unsqueeze(0),
        size=(image_mask.shape[0], image_mask.shape[1]),
        mode='bilinear',
        align_corners=True
        ).squeeze()

    def calculate_attention_entropy(attn):
        """
        Calculate the entropy of the attention map.
        """
        # Normalize attention map along the last dimension (softmax-like normalization)
        attn_probs = attn / attn.sum(dim=-1, keepdim=True)

        # Avoid log(0) by adding a small epsilon
        eps = 1e-9
        attn_probs = attn_probs + eps

        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)

        # Average entropy across all heads
        return entropy.mean().item()

    # Calculate entropy of the attention map
    entropy = calculate_attention_entropy(attn)

    # Dynamically adjust the threshold based on entropy
    max_entropy = torch.log(torch.tensor(attn.shape[-1], dtype=torch.float))  # Max entropy for uniform dist
    normalized_entropy = entropy / max_entropy.item()
    calculated_threshold = (np.exp(base_top_k * normalized_entropy)-1) / (np.exp(base_top_k)-1)
    calculated_threshold = max(min(calculated_threshold, 1.0),0.1)

    # Print entropy and calculated threshold
    # print(f"Entropy: {entropy:.4f}, Normalized Entropy: {normalized_entropy:.4f}, Calculated Threshold: {calculated_threshold:.4f}")

    # Flatten attention map and calculate Top-K threshold
    flattened_attn = attn.view(-1).float()
    threshold_index = int(len(flattened_attn) * calculated_threshold)
    threshold_value = torch.topk(flattened_attn, threshold_index).values[-1]

    image_mask = (attn >= threshold_value).float()
    # # Update the image mask
    # for row in range(attn.shape[0]):
    #     for col in range(attn.shape[1]):
    #         if attn[row, col] >= threshold_value:
    #             image_mask[row][col] = 1

    return image_mask