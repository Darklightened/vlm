#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
from abc import ABC, abstractmethod

import torch
from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, key_info

from .clip_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

import math


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            if config.mm_vision_tower.replace("./", "")[:4] == "vit/":
                config.mm_vision_tower = os.path.join(
                    config._name_or_path, config.mm_vision_tower.replace("./", "")
                )
            else:
                config.mm_vision_tower = config.mm_vision_tower.replace("./", "")
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            
    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_downsampled_vision_towers(self, stage=-1):
        downsampled_vision_towers = getattr(self, "downsampled_vision_towers", None)
        # assert type(downsampled_vision_towers) is nn.ModuleDict, "Must be dict"
        downsampled_vision_tower = downsampled_vision_towers[str(stage)]
        return downsampled_vision_tower

    def initialize_vision_modules(self, model_args):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            self.vision_tower = vision_tower
        else:
            vision_tower = self.vision_tower
            if not vision_tower.is_loaded:
                vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_downsampled_vision_towers(self, stage=-1):
        # stage -2:  84
        # stage -1: 168
        # stage  0: 336
        # stage  1: 672
        return self.get_model().get_downsampled_vision_towers(stage)

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    
    def encode_downsampled_images(self, images, stage):
        # stage -2:  84
        # stage -1: 168
        # stage  0: 336
        # stage  1: 672
        image_features = self.get_model().get_downsampled_vision_towers(stage=stage)(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images,
        downsampled_images=None, modalities=["image"], image_sizes=None, image_mask=None, generation_type=None
    ):
        vision_tower = self.get_vision_tower()

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)



        # #######################################################################
        # ## get stages
        # stages = []
        # if downsampled_images is not None:
        #     for stage in downsampled_images.keys():
        #         if stage < 0:
        #             stages.append(stage)
        #     if len(stages) > 0:
        #         stages.sort()
        
        # ## encode downsampled images
        # encoded_downsampled_image_features = []
        # for stage in stages:
        #     encoded_downsampled_image_features.append(
        #         self.encode_downsampled_images(downsampled_images[stage].squeeze(0), stage)
        #     )
        # #######################################################################

        new_input_embeds = []
        new_labels = [] if labels is not None else None

        cur_input_ids = input_ids[0]
        image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
        cur_new_input_embeds = []

        image_token_start = image_token_indices[0]

        cur_new_input_embeds.append(
            self.get_model().embed_tokens(cur_input_ids[:image_token_start])
        )

        # ###############################################
        # downsampled_features_list = []
        # stage0_features_list = []
        
        # ### STEP ###
        # # 1. interpolate mask to encoded size,
        # # 2. elementwise multiplication,
        # # 3. insert only non-zero features to list.
        # for downsampled_feature, stage in zip(encoded_downsampled_image_features, stages):
        #     mask = image_mask[stage]
        #     _, num_patches, _ = downsampled_feature.shape
        #     patch_per_side = int(math.sqrt(num_patches))

        #     resized_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(patch_per_side, patch_per_side), mode='nearest')

        #     downsampled_feature = downsampled_feature[0]
        #     downsampled_feature = downsampled_feature.squeeze().view(patch_per_side, patch_per_side, -1)
        #     resized_mask = resized_mask.squeeze().unsqueeze(-1)
            
        #     downsampled_feature = downsampled_feature * resized_mask
        #     # downsampled_feature = downsampled_feature.view(num_patches, -1)
        #     downsampled_feature = downsampled_feature.flatten(0, 1)
            
        #     for f in downsampled_feature:
        #         if f.min() == 0 and f.max() == 0: continue
        #         downsampled_features_list.append(f.unsqueeze(0))
        #     # downsampled_features_list.append(self.model.image_newline.unsqueeze(0))
        
        # # patches_per_side = self.get_vision_tower().num_patches_per_side
        # patches_per_side = 32
        # stage0_feature = image_features[0].view(patches_per_side, patches_per_side, -1)
        # mask = image_mask[0]
        # resized_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(patches_per_side, patches_per_side), mode='nearest')
        # resized_mask = resized_mask.squeeze().unsqueeze(-1)
        # stage0_feature = stage0_feature * resized_mask
        # # stage0_feature = stage0_feature.view(patches_per_side ** 2, -1)
        # stage0_feature = stage0_feature.flatten(0, 1)
        # for f in stage0_feature:
        #     if f.min() == 0 and f.max() == 0: continue
        #     stage0_features_list.append(f.unsqueeze(0))
        # # stage0_features_list.append(self.model.image_newline.unsqueeze(0))
        
        # concat_list = []
        # if len(downsampled_features_list) > 0:
        #     downsampled_features = torch.cat(downsampled_features_list, dim=0)
        #     concat_list.append(downsampled_features)
        # if len(stage0_features_list) > 0:
        #     stage0_features = torch.cat(stage0_features_list, dim=0)
        #     concat_list.append(stage0_features)

        # image_feature = torch.cat(concat_list, dim=0)
        # image_feature = image_feature.type(torch.bfloat16)
        # #########################################################

        # # print("image features:", image_feature.shape)
        # # cur_image_features = image_features[0]
        cur_new_input_embeds.append(image_features[0])
        # cur_new_input_embeds.append(image_feature)

        cur_input_ids = cur_input_ids[image_token_start + 1 :]

        if cur_input_ids.numel() > 0:
            cur_new_input_embeds.append(
                self.get_model().embed_tokens(cur_input_ids)
            )

        cur_new_input_embeds = [
            x.to(device=self.device) for x in cur_new_input_embeds
        ]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
        new_input_embeds.append(cur_new_input_embeds)

        # if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
        #     print("@@@")
        #     max_len = max(x.shape[0] for x in new_input_embeds)

        #     new_input_embeds_align = []
        #     for cur_new_embed in new_input_embeds:
        #         cur_new_embed = torch.cat(
        #             (
        #                 cur_new_embed,
        #                 torch.zeros(
        #                     (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
        #                     dtype=cur_new_embed.dtype,
        #                     device=cur_new_embed.device,
        #                 ),
        #             ),
        #             dim=0,
        #         )
        #         new_input_embeds_align.append(cur_new_embed)
        #     new_input_embeds = torch.stack(new_input_embeds_align, dim=0)
        # else:
        #     new_input_embeds = torch.stack(new_input_embeds, dim=0)
        
        new_input_embeds = torch.stack(new_input_embeds, dim=0)

        # if attention_mask is not None:
        #     new_attn_mask_pad_right = torch.full(
        #         (
        #             attention_mask.shape[0],
        #             new_input_embeds.shape[1] - input_ids.shape[1],
        #         ),
        #         True,
        #         dtype=attention_mask.dtype,
        #         device=attention_mask.device,
        #     )
        #     attention_mask = torch.cat(
        #         (attention_mask, new_attn_mask_pad_right), dim=1
        #     )
        #     assert attention_mask.shape == new_input_embeds.shape[:2]

        attention_mask = torch.ones((1, new_input_embeds.shape[1]), device=new_input_embeds.device)

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
