from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenModel, LlavaQwenForCausalLM

from lmms_eval.recursion_utils import *
import numpy as np
import cv2
import torch.nn.functional as F
import csv
import os
import ast
from pathlib import Path
import math
import copy
import wandb
import json

from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
    get_heatmap
)
from llava.mm_utils import (
    norm_relu,
    norm_min_max,
)

@dataclass
class LlavaRecursionConfig():    
    stages: List[int] = (-2, -1, 0, 1)
    positional_embedding_type: str = "bilinear_interpolation"
    generation_type: str = "recursion"
    attention_thresholding_type: str = "attn_topk"
    attn_norm: str = None
    attention_threshold: List[float] = (1.0, 1.0, 1.0)
    save_output: bool = False
    output_csv_path: Optional[str] = None
    output_json_path: Optional[str] = None
    contrastive_alphas: List[float] = (0.0, 0.0, 0.0)
    square: int = 1
    _device: str = "cuda:0"
    use_noised_for_contrastive: bool = False
    cd_strategy: str = "default"

class LlavaQwenForRecursion(LlavaQwenForCausalLM):
    config_class = LlavaRecursionConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, **kwargs)
    
    def init_all(self, recursion_config, *args, **kwargs):
        # Initialize recursive generation attributes        
        self.stages = recursion_config.stages
        self.positional_embedding_type = recursion_config.positional_embedding_type
        self.generation_type = recursion_config.generation_type
        self.attention_thresholding_type = recursion_config.attention_thresholding_type
        self.attention_threshold = recursion_config.attention_threshold
        self.square = recursion_config.square
        self.cd_strategy = recursion_config.cd_strategy

        self.contrastive_decoder = ContrastiveDecoder(self.cd_strategy)

        # Device assignment
        self._device = recursion_config._device

        # Image and patch size calculations
        self.patch_size = 14
        self.model.to(dtype=torch.bfloat16)
        self.image_size = getattr(
            self.model.vision_tower.vision_tower.vision_model.embeddings, "image_size", None
        )
        if self.image_size is None:
            raise ValueError("`image_size` is not defined in vision tower embeddings.")

        self.default_grid_size = 27

        # Initialize downsampled vision towers after loading weights
        if recursion_config.stages[0] < 0:
            self.model.downsampled_vision_towers = self.init_downsampled_vision_towers(
                self.model.vision_tower,
                recursion_config.stages,
                recursion_config.positional_embedding_type,
                self._device,
            )
        print("Downsampled towers initialized.")

        # Save output configuration
        self.save_output = recursion_config.save_output
        self.output_csv_path = recursion_config.output_csv_path
        self.output_json_path = recursion_config.output_json_path
        self.contrastive_alphas = recursion_config.contrastive_alphas    
        self.use_noised_for_contrastive = recursion_config.use_noised_for_contrastive
    
    def init_downsampled_vision_towers(self, vision_tower, stages, positional_embedding_type, device):
        print(f"change positional embedding to {positional_embedding_type}")
        downsampled_vision_towers = torch.nn.ModuleDict()
        for stage in stages:
            if stage == 0:
                break
            downsampled_vision_towers[str(stage)] = copy.deepcopy(vision_tower)

        # Default configurations of model position embedding
        patch_size = 14
        image_size = vision_tower.vision_tower.vision_model.embeddings.image_size
        original_num_patches = vision_tower.vision_tower.vision_model.embeddings.num_patches

        for stage in stages:
            if stage == 0:
                break
            downsampled_image_size = image_size * pow(2, stage)
            # assert (downsampled_image_size // patch_size) == (downsampled_image_size / patch_size), f"unavailable stage: {stage}"
            
            
            num_patches = int(int(downsampled_image_size // patch_size) ** 2)
            num_positions = num_patches
            embed_dim = vision_tower.vision_tower.vision_model.embeddings.embed_dim

            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.image_size = downsampled_image_size
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.num_patches = num_patches
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.num_positions = num_positions
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent=False)
        
            # Modify positional embedding to match the resized image size
            if positional_embedding_type == "bilinear_interpolation":
                # Interpolate from the pretrained positional embedding
                print("Bilienar interpolation embedding type.")
                original_embedding = downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data
                
                new_h = int(original_num_patches**(1/2) * pow(2, stage))
                # (1, 1024, 24, 24)
                original_embedding = original_embedding.view(int(original_num_patches**(1/2)), int(original_num_patches**(1/2)), -1).permute(2, 0, 1).unsqueeze(0)
                
                resized_positional_embeddings = torch.nn.functional.interpolate(
                                                original_embedding,
                                                size=(new_h, new_h),  # r
                                                mode='bicubic', # https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py
                                                align_corners=False
                                            )  
                resized_positional_embeddings = resized_positional_embeddings.squeeze(0).permute(1, 2, 0).reshape(-1, 1152)
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(len(resized_positional_embeddings), embed_dim).to(dtype=torch.bfloat16, device=device)
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(resized_positional_embeddings)

        return downsampled_vision_towers
    
    def reset_image_mask(self, stage1_grid=2):
        self.image_mask = dict()
        for idx, stage in enumerate(self.stages):
            if stage == -2: grid_size = 6
            elif stage == -1: grid_size = 13
            elif stage == 0: grid_size = 27
            elif stage == 1: grid_size = 27 * stage1_grid
            self.image_mask[stage] = torch.ones(grid_size, grid_size, device=self.device)

            # if idx == 0:
            #     # activate every token of the first stage
            #     self.image_mask[stage] = torch.ones(grid_size, grid_size, device=self.device)        
            # else:
            #     self.image_mask[stage] = torch.zeros(grid_size, grid_size, device=self.device)
    
    # Method to log each stage's results
    def save_stage_to_csv(self, doc_id, stage, text_output, cumulative_confidences):
        with open(self.output_csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Prepare the row with doc_id, stage, and text_output, followed by each cumulative confidence as separate columns
            row = [doc_id, stage, text_output] + cumulative_confidences
            writer.writerow(row)
    
    def save_stage_to_json(self, doc_id, stage, text_output, logits, tokenizer, task, logits_list=None, text_outputs=None):
        # Ensure the output JSON path is defined
        assert self.output_json_path is not None, "Output JSON path is not provided."

        self.output_json_path = Path(self.output_json_path)

        # Check if the file exists, otherwise initialize it with an empty dictionary
        if not self.output_json_path.exists():
            with open(self.output_json_path, mode="w") as file:
                json.dump({}, file)  # Initialize the file with an empty dictionary

        # Load existing JSON data
        with open(self.output_json_path, mode="r") as file:
            data = json.load(file)

        # Convert doc_id to string
        doc_id = str(doc_id)

        if "mmbench" in task or "mmstar" in task:
            ## A, B, C, D
            tokens = ['A', 'B', 'C', 'D']
            token_ids = [319, 350, 315, 360]
        elif "pope" in task:
            ## yes, no
            tokens = ["Yes", "No"]
            token_ids = [3869, 1939]
        else:
            tokens = []
            token_ids = []
        
        # print(logits[0].shape) 

        if logits_list is not None:
            k = 100

            # Get top-k indices from the last stage
            final_logits = logits_list[-1][0].cpu().detach().numpy()
            final_topk_indices = final_logits.argsort()[-k:][::-1]  # Top-k indices, descending order

            final_topk_logits = [(tokenizer.decode([idx]).strip(), float(final_logits[idx])) for idx in final_topk_indices]

            # Prepare to store the results for all stages            

            for stage, (logits, text_output) in enumerate(zip(logits_list, text_outputs)):
                # Extract logits for the top-k indices from the current stage
                all_logits = logits[0].cpu().detach().numpy()  # Assuming logits is a tensor
                logits_for_labels = {
                    token: logits[0][token_id].item()
                    for token, token_id in zip(tokens, token_ids)
                }

                # Decode tokens for the top-k indices
                top_logits = [(tokenizer.decode([idx]).strip(), float(all_logits[idx])) for idx in final_topk_indices]

                # Prepare the entry for the current stage
                stage_entry = {
                    "Stage": stage,
                    "Text Output": text_output,  # Assuming text_output is defined earlier in the loop
                    "Logits": {k: float(v) for k, v in logits_for_labels.items()},  # Logits corresponding to top-k indices
                    "Top-100 Logits": top_logits,  # Decoded tokens and logits for top-k indices
                }

                # Update the JSON data with the new stage information
                if doc_id not in data:
                    data[doc_id] = []  # Initialize a list for stages if this doc_id is new

                data[doc_id].append(stage_entry)

            # Save the updated JSON data
            with open(self.output_json_path, mode="w") as file:
                json.dump(data, file, indent=4)  
        
        else:        
            k = 100

            logits_for_labels = {
                token: logits[0][0][token_id].item()
                for token, token_id in zip(tokens, token_ids)
            }

            # Get top-100 logits and decoded tokens
            all_logits = logits[0][0].cpu().detach().numpy()  # Assuming logits is a tensor
            top_indices = all_logits.argsort()[-k:][::-1]  # Top-100 indices, descending order
            top_logits = [(tokenizer.decode([idx]).strip(), float(all_logits[idx])) for idx in top_indices]

            # Prepare the entry for the current stage
            stage_entry = {
                "Stage": stage,
                "Text Output": text_output,
                "Logits": {k: float(v) for k, v in logits_for_labels.items()},  # Ensure all values are converted to float
                "Top-100 Logits": top_logits
            }

            # Update the JSON data with the new stage information
            if doc_id not in data:
                data[doc_id] = []  # Initialize a list for stages if this doc_id is new

            data[doc_id].append(stage_entry)

            # Save the updated JSON data
            with open(self.output_json_path, mode="w") as file:
                json.dump(data, file, indent=4)

    @torch.no_grad()
    def generate_recursive(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,            
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]: 
        modalities = kwargs.pop("modalities", None) if "modalities" in kwargs and modalities is None else modalities
        # position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)   
        pad_token_ids = kwargs.pop("pad_token_id", None)      
        downsampled_images = kwargs.pop("downsampled_images", None)
        stage1_grid = kwargs.pop("stage1_grid", None)
        max_length = kwargs.pop("max_length", None)
        doc_id = kwargs.pop("doc_id", None)
        flattened_visuals = kwargs.pop("flattened_visuals", None)
        use_cache = kwargs.pop("use_cache", None)
        # gen_kwargs = kwargs.pop("gen_kwargs", None)
        tokenizer = kwargs.pop("tokenizer", None)
        question_input = kwargs.pop("question_input", None)
        task = kwargs.pop("task", None)

        if self.use_noised_for_contrastive:
            noised_images = add_diffusion_noise(images, 500)

        self.reset_image_mask(stage1_grid) 
        #final_text = "" 
        final_token = []        

        for token_idx in range(max_length):
            self.reset_image_mask(stage1_grid)
            stage_logit_list = []
            text_output_list = []     

            for idx_stage, stage in enumerate(self.stages):

                last_stage = (stage == self.stages[-1])

                # print(kwargs)

                cont = self.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_ids,
                    max_new_tokens=1,
                    images=images,
                    use_cache=use_cache,
                    generation_type=self.generation_type,
                    downsampled_images=downsampled_images,
                    image_mask=self.image_mask,
                    return_dict_in_generate=True,
                    output_attentions=(not last_stage),
                    output_scores=True,
                    temperature=kwargs["temperature"],
                    top_p=kwargs["top_p"],
                    num_beams=kwargs["num_beams"],
                    do_sample=kwargs["do_sample"],
                    image_sizes=kwargs["image_sizes"],
                )

                # Delete the start-of-sequence (SOS) token if it exists
                # if cont["sequences"][0][0] == 1:  # Assuming 1 is the SOS token ID
                #     cont["sequences"] = cont["sequences"][0][1:].unsqueeze(0)

                text_outputs = tokenizer.batch_decode(cont["sequences"], skip_special_tokens=True)
                # print("text:", text_outputs.strip())
                # if not last_stage:
                #     print(len(cont["attentions"]))
                scores = cont.scores
                #print(f"scores: {scores}")
                sequences = cont["sequences"][0]
                stage_logit_list.append(scores[0])
                text_output_list.append(text_outputs[0])

                # # Save confidence for analysis
                # if self.save_output and token_idx==0:
                #     _, _, cumulative_confidences = calculate_entropy_and_all_confidences(
                #         sequences, scores=scores
                #     )
                #     self.save_stage_to_csv(doc_id, f"Stage {idx_stage}", text_outputs, cumulative_confidences)
                
                # # Save json logits
                # if self.save_output and token_idx==0:     
                #     logits = scores                       
                #     self.save_stage_to_json(
                #         doc_id=doc_id,
                #         stage=f"Stage {idx_stage}",
                #         text_output=text_outputs[0],  # Assuming batch size = 1
                #         logits=logits,  # Logits of shape [sequence length, vocab size]
                #         tokenizer = tokenizer,
                #         task = task
                #     )


                if last_stage:
                    final_logit = scores[0].clone()
                    # Save json logits
                    if self.save_output and token_idx==0:     
                        logits = scores                       
                        self.save_stage_to_json(
                            doc_id=doc_id,
                            stage=f"Stage {idx_stage}",
                            text_output=text_outputs[0],  # Assuming batch size = 1
                            logits=logits,  # Logits of shape [sequence length, vocab size]
                            logits_list = stage_logit_list,
                            text_outputs = text_output_list,
                            tokenizer = tokenizer,
                            task = task
                        )
                    # Contrastive decoding

                    # use noised version of image for decoding
                    if self.use_noised_for_contrastive:
                        print("use noised anti")
                        noised_cont = self.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            pad_token_id=pad_token_ids,
                            images=noised_images,
                            image_sizes=gen_kwargs["image_sizes"],
                            do_sample=True if gen_kwargs["temperature"] > 0 else False,
                            temperature=gen_kwargs["temperature"],
                            top_p=gen_kwargs["top_p"],
                            num_beams=gen_kwargs["num_beams"],
                            max_new_tokens=1,                    
                            generation_type=None,
                            return_dict_in_generate=True,
                            output_attentions=False,
                            output_scores=True                            
                        )
                        noised_scores = noised_cont.scores[0]
                        final_logit += self.contrastive_alphas[-1]*(final_logit - noised_scores)
                    else:
                        pass
                        # final_logit = self.contrastive_decoder.compute_final_logits(stage_logit_list, self.contrastive_alphas, cutoff=True)

                        # code_adaptive = False

                        # if code_adaptive:
                        #     # Use multi-stage contrastive decoding
                        #     for idx in range(len(self.stages) - 1):
                        #         p_v = nn.functional.softmax(stage_logit_list[idx + 1], dim=-1)
                        #         p_d = nn.functional.softmax(stage_logit_list[idx], dim=-1)

                        #         # Calculate KL-based adaptive weight
                        #         cd_alpha = self.contrastive_alphas[idx]
                        #         kl_d = 0.5 * ((torch.log2(torch.abs(p_v - p_d) ** cd_alpha + 1)) * (p_v + p_d)).sum(dim=-1).unsqueeze(-1)
                        #         kld_alpha = 1 - kl_d

                        #         # Calculate cutoff threshold
                        #         cutoff = kl_d * p_v.max(dim=-1, keepdim=True).values

                        #         # Calculate stage-specific contrastive logits
                        #         diffs = (1 + kld_alpha) * stage_logit_list[idx + 1] - kld_alpha * stage_logit_list[idx]
                        #         cd_logits = diffs.masked_fill(p_v < cutoff, -float("inf"))

                        #         # Update final logits with weighted stage contributions
                        #         final_logit += cd_logits
                        # else:
                        #     # use multi-stage contrastive decoding
                        #     for idx in range(len(self.stages) - 1):
                        #         final_logit += self.contrastive_alphas[idx] * (stage_logit_list[idx + 1] - stage_logit_list[idx])

                    best_token = torch.argmax(final_logit, dim=-1)
                    final_token.append(best_token.item())
                    #stage_text = tokenizer.batch_decode(best_token, skip_special_tokens=True)[0]

                    # Append decoded text to final output
                    final_text = tokenizer.batch_decode([final_token], skip_special_tokens=True)[0]
                    # print(f'final_text: {final_text.strip()}')

                    # Update input_ids and attention_mask for the next token
                    input_ids = torch.cat([input_ids, best_token.unsqueeze(0)], dim=-1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)], dim=-1
                    )
                    break

                ret_attn = get_heatmap(
                    self,
                    cont,
                    tokenizer,
                    question_input[0],
                    input_ids,
                    stage,
                    self.stages,
                    self.image_mask,
                    select_token=None,
                    image=flattened_visuals[0],
                    save_path=None,
                    # save_path="/workspace/vlm/temp/",
                )

                if self.attention_thresholding_type == "layer_mean":
                    self.image_mask[stage + 1] = layer_mean_based_recursion(
                        attn=ret_attn,
                        attn_threshold=self.attention_threshold[idx_stage],
                        image_mask=self.image_mask[stage + 1]
                    )
                elif self.attention_thresholding_type == "layer_mean_topk":
                    self.image_mask[stage + 1] = layer_mean_topk_based_recursion(
                        attn=ret_attn,
                        top_k=self.attention_threshold[idx_stage],
                        image_mask=self.image_mask[stage + 1]
                    )
                elif self.attention_thresholding_type == "confidence_topk":
                    self.image_mask[stage + 1] = confidence_topk_based_recursion(
                        attn=ret_attn,
                        top_k=self.attention_threshold[idx_stage],
                        sequences=sequences,
                        scores=scores,
                        image_mask=self.image_mask[stage + 1]
                    )
                elif self.attention_thresholding_type == "attn_topk":
                        # Call the function and get the entropy for the current stage
                        self.image_mask[stage + 1], calculated_threshold = attn_entropy_topk_based_recursion(
                            attn=ret_attn,
                            base_top_k=self.attention_threshold[idx_stage],
                            image_mask=self.image_mask[stage + 1]
                        )

                        # wandb.log({f"Stage_{idx_stage}_Threshold": calculated_threshold})
                else:
                    self.activate_image_mask(self.stages[idx_stage + 1])

                del cont
                torch.cuda.empty_cache()

            # Terminate if end-of-sequence (EOS) token is generated or max tokens reached
            if input_ids[0, -1] == tokenizer.eos_token_id:
                break
        
        return [final_text.strip()]
        