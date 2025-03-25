import os
import numpy as np
import itertools

def generate_command(alpha, beta, gamma, experiment_name_suffix):
    """
    Generate the bash command to run the experiment.
    
    Args:
        contrastive_values (str): The contrastive_alphas for this run.
        experiment_name_suffix (str): Unique suffix for the wandb experiment name.
        device (str): Device to run the experiment on (default: cuda:2).
    
    Returns:
        str: The generated bash command.
    """
    base_command = (
        f"CUDA_VISIBLE_DEVICES=0,1,2,3 "
        f"python3 -m accelerate.commands.launch "
        f"--num_processes=4 "
        f"--main_process_port 29827 "
        f"-m lmms_eval "
        f"--model llava "
        f"--model_args pretrained='liuhaotian/llava-v1.6-mistral-7b' "
        f"--tasks vqav2_val_lite "
        f"--batch_size 1 "
        f"--log_samples "
        f"--log_samples_suffix llava_v1.6_pope "
        f"--output_path ./logs/ "
        f"--generation_type recursion "
        f"--fix_grid 2x2 "
        f"--attention_thresholding_type attn_topk "
        f"--attention_threshold '1.0' "
        f"--positional_embedding_type bilinear_interpolation "
        f"--remove_unpadding True "
        f"--attn_norm None "
        f"--stages '-2' '-1' '0' '1' "
        f"--verbosity DEBUG "
        f"--square 1 "
        f"--contrastive_alphas '{alpha}' '{beta}' '{gamma}' "
        f"--wandb_args 'project=llava1.6_recursive_eval_mistral-7b,entity=VLM_Hallucination_Woohyeon,name=new_exp_{experiment_name_suffix}' "
    )
    return base_command

def run_experiments():
    """
    Run experiments for specified alpha, beta, gamma values in reverse order.
    """
    # Define the specific alpha, beta, gamma values
    alphas = np.arange(0.1, 1.1, 0.2).tolist()
    betas = np.arange(0.1, 1.1, 0.2).tolist()
    gammas = np.arange(0.1, 1.1, 0.2).tolist()

    # Generate combinations for alpha, beta, gamma
    parameter_combinations = list(itertools.product(alphas, betas, gammas))

    experiment_idx = 0
    total_experiments = len(alphas)

    for alpha, beta, gamma in parameter_combinations:
        # Combine alpha, beta, gamma values
        parameters = [alpha, beta, gamma]

        # Create the unique name for the wandb experiment
        experiment_name_suffix = f"run_{experiment_idx}_params_contrastive_{'_'.join(map(lambda x: f'{x:.2f}', parameters))}"

        # # Convert the parameters to string format
        # parameter_string = f"[{','.join(map(str, parameters))}]"

        # Generate the command
        command = generate_command(alpha, beta, gamma, experiment_name_suffix)

        # Print and run the command
        print(f"Running experiment {experiment_idx + 1}/{total_experiments}: {command}")
        os.system(command)

        experiment_idx += 1


# Run the experiments with the defined thresholds
run_experiments()