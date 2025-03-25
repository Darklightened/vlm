from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, MistralConfig

from torch.nn import CrossEntropyLoss

from transformers import MistralModel, MistralForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralConfig, LlavaMistralModel, LlavaMistralForCausalLM

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
    positional_embedding_type: str = "bilinear"
    generation_type: str = "recursion"
    attention_thresholding_type: str = "layer_mean_topk"
    attn_norm: str = "norm_min_max"
    attention_threshold: List[float] = (0.9, 0.9, 0.9)
    save_output: bool = False
    output_csv_path: Optional[str] = None
    output_json_path: Optional[str] = None
    contrastive_alphas: List[float] = (1.0, 1.0, 1.0)
    square: int = 1
    fix_grid: Optional[str] = "2x2"
    _device: str = "cuda:0"
    use_noised_for_contrastive: bool = False

class LlavaMistralForRecursion(LlavaMistralForCausalLM):
    config_class = LlavaRecursionConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
    
    def init_all(self, recursion_config, *args, **kwargs):
        # Initialize recursive generation attributes        
        self.stages = recursion_config.stages
        self.positional_embedding_type = recursion_config.positional_embedding_type
        self.generation_type = recursion_config.generation_type
        self.attention_thresholding_type = recursion_config.attention_thresholding_type
        self.attention_threshold = recursion_config.attention_threshold
        self.square = recursion_config.square

        # Attention normalization logic
        if recursion_config.attn_norm == "None":
            self.attn_norm = None
        elif recursion_config.attn_norm == "norm_relu":
            self.attn_norm = norm_relu
        elif recursion_config.attn_norm == "norm_min_max":
            self.attn_norm = norm_min_max
        else:
            eval_logger.info(f"Unsupported norm type. Using norm_relu.")
            self.attn_norm = norm_relu

        # Device assignment
        self._device = recursion_config._device

        # Image and patch size calculations
        self.patch_size = 14
        print(self.model)
        self.image_size = getattr(
            self.model.vision_tower.vision_tower.vision_model.embeddings, "image_size", None
        )
        if self.image_size is None:
            raise ValueError("`image_size` is not defined in vision tower embeddings.")

        self.smallest_grid_size = int(self.image_size * pow(2, self.stages[0]) // self.patch_size)
        self.largest_grid_size = int(self.image_size * pow(2, self.stages[-1]) // self.patch_size)

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

        for stage in stages:
            if stage == 0:
                break
            downsampled_image_size = image_size * pow(2, stage)
            assert (downsampled_image_size // patch_size) == (downsampled_image_size / patch_size), f"unavailable stage: {stage}"
            
            
            num_patches = int((downsampled_image_size // patch_size) ** 2)
            num_positions = num_patches + 1
            embed_dim = vision_tower.vision_tower.vision_model.embeddings.embed_dim

            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.image_size = downsampled_image_size
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.num_patches = num_patches
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.num_positions = num_positions
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent=False)
        
            # Modify positional embedding to match the resized image size
            if positional_embedding_type == "zero":       
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.float16, device=device)
                torch.nn.init.constant_(downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight, 0)
            elif positional_embedding_type == "interpolation":
                print("interpolate embedding type.")
                # Interpolate from the pretrained positional embedding
                original_embedding = downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data
                original_num_positions = original_embedding.size(0)
                new_embedding = torch.nn.functional.interpolate(
                    original_embedding.unsqueeze(0).transpose(1, 2), 
                    size=(num_positions,), 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2).squeeze(0)
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.float16, device=device)
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(new_embedding)
            
            elif positional_embedding_type == "reduced":
                print("Reduced embedding type.")
                # Reduce the pretrained embedding by truncating
                original_embedding = downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.float16, device=device)
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(original_embedding[:num_positions])
                    
            elif positional_embedding_type == "bilinear_interpolation":
                # Interpolate from the pretrained positional embedding
                print("Bilienar interpolation embedding type.")
                original_embedding = downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data
                cls_token = original_embedding[:1, :]
                original_embedding = original_embedding[1:, :]  # Skip CLS
                # (1, 1024, 24, 24)
                new_height = int(len(original_embedding) ** (1/2) / (2 ** -stage))
                original_embedding = original_embedding.view(int(len(original_embedding)**(1/2)), int(len(original_embedding)**(1/2)), -1).permute(2, 0, 1).unsqueeze(0)
                
                resized_positional_embeddings = torch.nn.functional.interpolate(
                                                original_embedding,
                                                size=(new_height, new_height),  # r
                                                mode='bilinear',
                                                align_corners=False
                                            )  
                resized_positional_embeddings = resized_positional_embeddings.squeeze(0).permute(1, 2, 0).reshape(-1, 1024)
                new_embedding = torch.cat([cls_token, resized_positional_embeddings], dim=0)  
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(len(new_embedding), embed_dim).to(dtype=torch.float16, device=device)
                downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(new_embedding)
                
        return downsampled_vision_towers
    
    def reset_image_mask(self):
        self.image_mask = dict()
        for idx, stage in enumerate(self.stages):
            self.image_mask[stage] = torch.ones(self.largest_grid_size, self.largest_grid_size, device=self.device)        

        self.pad_mask = dict()
    
    def set_pad_mask(self, size, origin_x, origin_y):
        self.pad_mask = dict()
        if self.square == 1:
            for stage in self.stages:
                grid = int(self.image_size * pow(2, stage) // self.patch_size)
                temp_tensor = torch.zeros(grid, grid, device=self.device)
                step = size / grid

                init_x, init_y = 0, 0
                stepped_x, stepped_y = 0, 0
                while stepped_x <= (size - origin_x) / 2:
                    stepped_x += step
                    init_x += 1
                while stepped_y <= (size - origin_y) / 2:
                    stepped_y += step
                    init_y += 1
                end_x, end_y = init_x, init_y
                while stepped_x < (size + origin_x) / 2:
                    stepped_x += step
                    end_x += 1
                while stepped_y < (size + origin_y) / 2:
                    stepped_y += step
                    end_y += 1

                temp_tensor[init_y - 1:end_y, init_x - 1:end_x] = 1
                temp_tensor = F.interpolate(temp_tensor.unsqueeze(0).unsqueeze(0), size=(self.largest_grid_size, self.largest_grid_size), mode='nearest').squeeze()
                self.pad_mask[stage] = temp_tensor
        elif self.square == 2:
            for stage in self.stages:
                grid = int(self.image_size * pow(2, stage) // self.patch_size)
                temp_tensor = torch.zeros(grid, grid, device=self.device)
                step = size / grid

                bounding_x, bounding_y = 0, 0
                stepped_x, stepped_y = 0, 0
                while stepped_x < origin_x:
                    stepped_x += step
                    bounding_x += 1
                while stepped_y < origin_y:
                    stepped_y += step
                    bounding_y += 1

                temp_tensor[:bounding_y, :bounding_x] = 1
                temp_tensor = F.interpolate(temp_tensor.unsqueeze(0).unsqueeze(0), size=(self.largest_grid_size, self.largest_grid_size), mode='nearest').squeeze()
                self.pad_mask[stage] = temp_tensor
        elif self.square == 3:
            for stage in self.stages:
                grid = int(self.image_size * pow(2, stage) // self.patch_size)
                temp_tensor = torch.zeros(grid, grid, device=self.device)
                step = size // grid
                bounding_x = 0
                bounding_y = 0
                for bounding_x, stepped_x in enumerate(range(0, size - origin_x, step)):
                    if stepped_x >= size - origin_x: break
                for bounding_y, stepped_y in enumerate(range(0, size - origin_y, step)):
                    if stepped_y >= size - origin_y: break
                temp_tensor[bounding_y:, bounding_x:] = 1
                temp_tensor = F.interpolate(temp_tensor.unsqueeze(0).unsqueeze(0), size=(self.largest_grid_size, self.largest_grid_size), mode='nearest').squeeze()
                self.pad_mask[stage] = temp_tensor
    
        # temp_tensor[:bounding_y, :bounding_x] = 1
        # temp_tensor = F.interpolate(temp_tensor.unsqueeze(0).unsqueeze(0), size=(self.largest_grid_size, self.largest_grid_size), mode='nearest').squeeze()
        # self.pad_mask = self.pad_mask * temp_tensor
    
    def combine_image_and_pad_mask(self):
        for stage in self.stages:
            self.image_mask[stage] = self.image_mask[stage] * self.pad_mask[stage]
    
    def activate_every_image_masks(self):
        self.image_mask = dict()
        for stage in self.stages:
            self.image_mask[stage] = torch.ones(self.largest_grid_size, self.largest_grid_size, device=self.device)        
    
    def activate_image_mask(self, stage):
        self.image_mask[stage] = torch.ones(self.largest_grid_size, self.largest_grid_size, device=self.device)
    
    # Method to log each stage's results
    def save_stage_to_csv(self, doc_id, stage, text_output, cumulative_confidences):
        with open(self.output_csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Prepare the row with doc_id, stage, and text_output, followed by each cumulative confidence as separate columns
            row = [doc_id, stage, text_output] + cumulative_confidences
            writer.writerow(row)
    
    def save_stage_to_json(self, doc_id, stage, text_output, logits, tokenizer, task):
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
        
        # print(logits[0].shape)        
        
        logits_for_labels = {
            token: logits[0][0][token_id].item()
            for token, token_id in zip(tokens, token_ids)
        }

        # Prepare the entry for the current stage
        stage_entry = {
            "Stage": stage,
            "Text Output": text_output,
            "Logits": logits_for_labels  
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
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)   
        pad_token_ids = kwargs.pop("pad_token_ids", None)      
        downsampled_images = kwargs.pop("downsampled_images", None)
        max_length = kwargs.pop("max_length", None)
        doc_id = kwargs.pop("doc_id", None)
        flattened_visuals = kwargs.pop("flattened_visuals", None)
        gen_kwargs = kwargs.pop("gen_kwargs", None)
        tokenizer = kwargs.pop("tokenizer", None)
        question_input = kwargs.pop("question_input", None)
        task = kwargs.pop("task", None)

        if self.use_noised_for_contrastive:
            noised_images = add_diffusion_noise(images, 500)

        self.reset_image_mask() 
        #final_text = "" 
        final_token = []        

        for token_idx in range(max_length):   
            cont = self.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_ids,
                images=images,
                image_sizes=gen_kwargs["image_sizes"],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=1,                    
                generation_type=self.generation_type,
                return_dict_in_generate=True,
                output_attentions=False,
                output_scores=True,
                downsampled_images=None,
                image_mask=self.image_mask
            )

            scores = cont.scores

            # Contrastive decoding
            final_logit = scores[0]

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
                    output_scores=True,
                    image_mask=self.image_mask                      
                )
                noised_scores = noised_cont.scores[0]
                # final_logit += self.contrastive_alphas[-1]*(final_logit - noised_scores)

                cutoff = torch.tensor(0.2, device=final_logit.device) * final_logit.max()
                
                logit_diff = final_logit - noised_scores

                filtered_logit_diff = logit_diff.masked_fill(final_logit < cutoff.unsqueeze(-1), 0)
                final_logit[final_logit < cutoff.unsqueeze(-1)] = -100

                final_logit += 1.0 * filtered_logit_diff
                
                best_token = torch.argmax(final_logit, dim=-1)
                final_token.append(best_token.item())
                #stage_text = tokenizer.batch_decode(best_token, skip_special_tokens=True)[0]

                # Append decoded text to final output
                final_text = tokenizer.batch_decode([final_token], skip_special_tokens=True)[0]
                # print(f'final_text: {final_text}')

                # Update input_ids and attention_mask for the next token
                input_ids = torch.cat([input_ids, best_token.unsqueeze(0)], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)], dim=-1
                )

            # Terminate if end-of-sequence (EOS) token is generated or max tokens reached
            if input_ids[0, -1] == tokenizer.eos_token_id:
                break
        
        return [final_text]
        