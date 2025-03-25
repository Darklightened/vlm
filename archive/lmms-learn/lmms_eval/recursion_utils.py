import numpy as np
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
from pathlib import Path
import matplotlib.pyplot as plt

def plot_attention_distribution(ret_attn, save_path="attention_distribution.png"):
    """
    Plots and saves the distribution of attention values.

    Args:
        ret_attn (torch.Tensor): The attention tensor with shape (48, 48).
        save_path (str): The path where the plot will be saved.
    """
    # Flatten the tensor for visualization
    flattened_ret_attn = ret_attn.flatten().detach().to('cpu').numpy()

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(flattened_ret_attn, bins=30, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Distribution of Attention Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Save the plot to the specified path
    plt.savefig(save_path)
    plt.close()

class BinarizeWithSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input) 

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output 

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
        token_prob = probs[0, token_id]  # Probability of the actual token

        # Update cumulative log probability for the full sequence up to this token
        log_prob_sum_full += torch.log(token_prob + 1e-10)
        # Calculate and store cumulative confidence for this subsequence
        cumulative_confidences.append(torch.exp(log_prob_sum_full))
        
        # Entropy calculation for the token
        entropy_sum -= token_prob * torch.log(token_prob + 1e-10)

    ### old version ###
    # for idx, token_id in enumerate(sequence):
    #     probs = F.softmax(scores[idx], dim=-1)  # Softmax to get probabilities for the current token
    #     token_prob = probs[0, token_id].item()  # Probability of the actual token

    #     # Update cumulative log probability for the full sequence up to this token
    #     log_prob_sum_full += np.log(token_prob + 1e-10)
    #     # Calculate and store cumulative confidence for this subsequence
    #     cumulative_confidences.append(np.exp(log_prob_sum_full))
        
    #     # Entropy calculation for the token
    #     entropy_sum -= token_prob * np.log(token_prob + 1e-10)

    # Full sequence confidence (cumulative up to the last token)
    P_T_given_I_Q_full = cumulative_confidences[-1] if cumulative_confidences else torch.exp(log_prob_sum_full)

    # print(f"Overall confidence P(T | I, Q): {P_T_given_I_Q_full}")
    # print(f"Entropy H(T | I, Q): {entropy_sum}")
    # print("Per-token cumulative confidences:", cumulative_confidences)

    return P_T_given_I_Q_full, entropy_sum, cumulative_confidences

#### Recursion Methods ####

def layer_mean_based_recursion(attn = None, attn_threshold = 0.1 , image_mask = None):
    for row in range(attn.shape[0]):
        for col in range(attn.shape[1]):
            if attn[row, col] >= attn_threshold:
                image_mask[row][col] = 1
    
    return image_mask

def layer_mean_topk_based_recursion(attn = None, attn_threshold = 0.1, image_mask = None):
    image_mask = image_mask.clone()
    flattened_attn = attn.view(-1) 
    # print(flattened_attn)
    # flattened_attn = flattened_attn.float()
    # print(flattened_attn)
    threshold_index = (len(flattened_attn) * (attn_threshold)).to(torch.int32)
    threshold_value = torch.topk(flattened_attn, threshold_index).values[-1]

    # mask = (attn >= threshold_value).to(torch.float16)
    image_mask = torch.sigmoid((attn - threshold_value) * 100)
    image_mask = BinarizeWithSTE.apply(image_mask)
    # print(mask)
    # mask = mask.float()
    # print(mask)
    
    # updated_image_mask = torch.max(updated_image_mask, mask)
    # return updated_image_mask
    return image_mask
    

    # for row in range(attn.shape[0]):
    #     for col in range(attn.shape[1]):
    #         if attn[row, col] >= threshold_value:
    #             image_mask[row][col] = 1

    # return image_mask

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


#### Test Time Adaptation ####
def print_trainable_parameters(model):
    """
    Prints the trainable parameters of a PyTorch model.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
    """
    print("Trainable parameters in the model:")
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"{name}: {param.numel()} parameters")
    
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {total_params - trainable_params}")
    
def TTA_recursion(attn, attn_threshold=0.1, image_mask=None):
    """
    Update the image mask based on attention values and threshold using broadcasting.

    Args:
        attn (torch.Tensor): The attention values (2D tensor).
        attn_threshold (float): The threshold for attention values.
        image_mask (torch.Tensor): The image mask to be updated.

    Returns:
        torch.Tensor: Updated image mask.
    """
    # diff = attn - attn_threshold  # learnable_attn_threshold와 연산
    # print(attn.max())
    # print(attn.min())
    # print(attn.mean())
    # exit()
    temp = torch.ones_like(attn) * attn_threshold
    diff = attn - temp  # learnable_attn_threshold와 연산
    # print("Inside TTA Recursion attn_threshold requires_grad:", attn_threshold.requires_grad)
    # print("Inside TTA Recursion attn_threshold grad_fn before sigmoid:", attn_threshold.grad_fn)
    # print("Inside TTA Recursion diff grad_fn:", diff.grad_fn) 
    # print("Inside TTA Recursion diff grad_fn:", attn.grad_fn) 
    image_mask = image_mask.clone()
    image_mask = torch.sigmoid(100 * diff)  
    image_mask = BinarizeWithSTE.apply(image_mask)
    # print("Inside TTA Recursion image_mask grad_fn:", image_mask.grad_fn)
    return image_mask

def TTA_soft_topk_recursion(attn, k=0.1, image_mask=None, temperature=0.1):    
    
    probs = torch.softmax(attn / temperature, dim=-1)
    sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)
    
    ranks = torch.arange(1, attn.size(-1) + 1, device=attn.device).float()
    weights = torch.exp((ranks-1) / k)  
    
    topk_scores = torch.sum(sorted_probs * weights, dim=-1, keepdim=True)
    
    image_mask = image_mask.clone()
    
    image_mask = torch.sigmoid(topk_scores)
    image_mask = BinarizeWithSTE.apply(image_mask)

    return image_mask