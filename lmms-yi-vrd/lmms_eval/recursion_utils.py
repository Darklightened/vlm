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

def attn_entropy_topk_based_recursion(attn=None, image_mask=None, base_top_k=0.3): 
    """
    Adjust Top-K threshold dynamically based on attention map entropy and record the entropy.
    """

    # Calculate entropy of the attention map
    entropy = calculate_attention_entropy(attn)

    # Dynamically adjust the threshold based on entropy
    max_entropy = torch.log(torch.tensor(attn.shape[-1], dtype=torch.float))  # Max entropy for uniform dist
    normalized_entropy = entropy / max_entropy.item()
    calculated_threshold = (np.exp(base_top_k * normalized_entropy)-1) / (np.exp(base_top_k)-1)
    calculated_threshold = max(min(calculated_threshold, 1.0),0.1)

    # Print entropy and calculated threshold
    print(f"Entropy: {entropy:.4f}, Normalized Entropy: {normalized_entropy:.4f}, Calculated Threshold: {calculated_threshold:.4f}")

    # Flatten attention map and calculate Top-K threshold
    flattened_attn = attn.view(-1).float()
    threshold_index = int(len(flattened_attn) * calculated_threshold)
    threshold_value = torch.topk(flattened_attn, threshold_index).values[-1]

    # Update the image mask
    for row in range(attn.shape[0]):
        for col in range(attn.shape[1]):
            if attn[row, col] >= threshold_value:
                image_mask[row][col] = 1

    return image_mask, calculated_threshold

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

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd