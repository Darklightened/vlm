from PIL import Image
from io import BytesIO
import base64
import math
import ast
import re
import torch
import copy
from llava_yi.model.constants import IMAGE_TOKEN_INDEX

import torch
from llava_yi.model import LlavaLlamaForCausalLM
from llava_yi.model.constants import IMAGE_TOKEN_INDEX
from transformers import AutoTokenizer, StoppingCriteria

## Additional imports for utils
import requests
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from matplotlib import cm


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [
        tokenizer(chunk).input_ids for chunk in prompt.split("<image_placeholder>")
    ]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids

# def load_pretrained_model(
#     model_path, load_8bit=False, load_4bit=False, device_map="auto", multimodal="IMAGE"
# ):


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
            return False
        else:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, self.start_len :], skip_special_tokens=True
            )
            flag = True
            for output in outputs:
                for keyword in self.keywords:
                    if keyword not in output:
                        flag = False
                        return False
            return flag


def init_downsampled_vision_towers(vision_tower, stages, positional_embedding_type, device):
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
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.bfloat16, device=device)
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
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.bfloat16, device=device)
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(new_embedding)
        
        elif positional_embedding_type == "reduced":
            print("Reduced embedding type.")
            # Reduce the pretrained embedding by truncating
            original_embedding = downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.bfloat16, device=device)
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
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(len(new_embedding), embed_dim).to(dtype=torch.bfloat16, device=device)
            downsampled_vision_towers[str(stage)].vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(new_embedding)
            
    return downsampled_vision_towers                                

def resize_and_center_crop(image, shortest_edge_length):
    # Calculate new dimensions and resize
    aspect_ratio = float(image.width) / float(image.height)
    if aspect_ratio > 1:
        new_width = int(shortest_edge_length * aspect_ratio)
        new_height = shortest_edge_length
    else:
        new_width = shortest_edge_length
        new_height = int(shortest_edge_length / aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # Calculate the position and perform the center crop
    left = (new_width - shortest_edge_length) / 2
    top = (new_height - shortest_edge_length) / 2
    right = (new_width + shortest_edge_length) / 2
    bottom = (new_height + shortest_edge_length) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))

    return cropped_image


def auto_pad_images(image, grid_params):
    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert len(grid_params) > 0, "Grid parameters should not be empty"

    # Step 1: Calculate and find the closest aspect ratio
    input_width, input_height = image.size
    input_aspect_ratio = input_width / input_height
    candidate_resolutions = [(w / h, w, h) for w in grid_params for h in grid_params]
    closest_aspect_ratio = min(candidate_resolutions, key=lambda x: abs(input_aspect_ratio - x[0]))

    candidate_resolutions = [(x[1], x[2]) for x in candidate_resolutions if abs(x[0] - closest_aspect_ratio[0]) < 1e-3]

    target_resolution = min(candidate_resolutions, key=lambda res: abs(max(input_width, input_height) / max(res) - 1))

    resize_width, resize_height = target_resolution
    if input_width > input_height:
        resize_height = int(resize_width / input_aspect_ratio)
    else:
        resize_width = int(resize_height * input_aspect_ratio)
    resized_image = image.resize((resize_width, resize_height), Image.ANTIALIAS)

    # Step 5: Pad the resized image if necessary to match the target resolution
    pad_width = target_resolution[0] - resize_width
    pad_height = target_resolution[1] - resize_height
    padded_image = Image.new("RGB", target_resolution, color=(0, 0, 0))
    padded_image.paste(resized_image, (pad_width // 2, pad_height // 2))

    return padded_image


def extract_patches(image, patch_size, overlap_ratio):
    assert isinstance(image, Image.Image), "Input should be a Pillow Image"
    assert patch_size > 0, "Patch size should be greater than 0"
    assert 0 <= overlap_ratio < 1, "Overlap ratio should be between 0 and 1"

    W, H = image.size
    patches = []

    stride = int(patch_size * (1 - overlap_ratio))

    num_patches_y = (H - patch_size) // stride + 1
    num_patches_x = (W - patch_size) // stride + 1

    y_start = (H - (num_patches_y - 1) * stride - patch_size) // 2
    x_start = (W - (num_patches_x - 1) * stride - patch_size) // 2

    for y in range(y_start, y_start + num_patches_y * stride, stride):
        for x in range(x_start, x_start + num_patches_x * stride, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    return patches


def process_highres_image_crop_split(image, data_args, processor=None):
    crop_resolution = data_args.image_crop_resolution
    split_resolution = data_args.image_split_resolution
    if processor is None:
        processor = data_args.image_processor
    image_crop = resize_and_center_crop(image, crop_resolution)
    image_patches = extract_patches(image_crop, patch_size=split_resolution, overlap_ratio=0)
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def process_highres_image(image, processor, grid_pinpoints):
    grid_params = [int(x) for x in grid_pinpoints.split(",")]
    width_height = max(image.size)
    fit_grid_params = [x for x in grid_params if x >= width_height]
    if len(fit_grid_params) == 0:
        select_size = max(grid_params)
    else:
        select_size = min(fit_grid_params)
    # FIXME: always select the 448
    select_size = max(grid_params)
    image_padded = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))

    # FIXME: this seems to be a bug that it always resizes instead of padding
    image_original_resize = image.resize((processor.size["shortest_edge"], processor.size["shortest_edge"]))
    image_padded = image_padded.resize((select_size, select_size))
    image_patches = extract_patches(image_padded, patch_size=processor.size["shortest_edge"], overlap_ratio=0)
    image_patches = [image_original_resize] + image_patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)

        # Calculate effective and wasted resolutions
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    # Convert grid_pinpoints from string to list
    if isinstance(grid_pinpoints, str) and "x" in grid_pinpoints:
        try:
            patch_size = processor.size[0]
        except Exception as e:
            patch_size = processor.size["shortest_edge"]
        assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
        # Use regex to extract the range from the input string
        matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
        range_start = tuple(map(int, matches[0]))
        range_end = tuple(map(int, matches[-1]))
        # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
        grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
        # Multiply all elements by patch_size
        grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size["height"])

    # FIXME: this seems to be a bug that it resizes instead of pad.
    # but to keep it consistent with previous, i will keep it as it is
    # TODO: uncomment below to ablate with the padding
    if isinstance(processor.size, dict):
        shortest_edge = processor.size["shortest_edge"]
    else:
        shortest_edge = min(processor.size)
    image_original_resize = image.resize((shortest_edge, shortest_edge))
    # image_padded_square = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    # image_original_resize = image_padded_square.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors="pt")["pixel_values"][0] for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    return image_processor(images, return_tensors="pt")["pixel_values"]
    # image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    # new_images = []
    # for image in images:
    #     image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
    #     new_images.append(image)
    # if image_aspect_ratio == "highres":
    #     for image in images:
    #         image = process_highres_image(image, image_processor, model_cfg.image_grid_pinpoints)
    #         new_images.append(image)
    # elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
    #     for image in images:
    #         image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
    #         new_images.append(image)
    # elif image_aspect_ratio == "crop_split":
    #     for image in images:
    #         image = process_highres_image_crop_split(image, model_cfg, image_processor)
    #         new_images.append(image)
    # elif image_aspect_ratio == "pad":
    #     for image in images:
    #         image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
    #         image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    #         new_images.append(image)
    # else:
    #     return image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    # if all(x.shape == new_images[0].shape for x in new_images):
    #     new_images = torch.stack(new_images, dim=0)
    # return new_images


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0] :] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False


# many are copied from https://github.com/mattneary/attention/blob/master/attention/attention.py
# here it nullifies the attention over the first token (<bos>)
# which in practice we find to be a good idea

def aggregate_llm_attention(attn):
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            attns_per_head[-1][1:].cpu(),
            # attns_per_head[-1].cpu(),
            # add zero for the final generated token, which never
            # gets any attention
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    
    # print(f"Aggregated attention shape: {torch.stack(avged).mean(dim=0).shape}")
    return torch.stack(avged).mean(dim=0)

def aggregate_llm_attention_single_layer(layer):
    """
    Extract average attention vector for a single layer.
    Args:
        layer: The attention matrix for a single layer. Shape [batch_size, num_heads, seq_len, seq_len].
    Returns:
        Aggregated attention vector for the given layer.
    """
    #print(f"Processing single layer attention with shape: {layer.shape}")
    layer_attns = layer.squeeze(0)  # Remove batch dimension
    #print(f"Layer attention shape after squeeze: {layer_attns.shape}")
    attns_per_head = layer_attns.mean(dim=0)  # Average across attention heads
    #print(f"Attention per head shape: {attns_per_head.shape}")
    vec = torch.concat((
        # We zero the first entry because it's called null attention
        torch.tensor([0.]),
        # Take [-1], the last row corresponding to the current token
        attns_per_head[-1][1:].cpu(),
        # Add zero for the final generated token, which never gets attention
        torch.tensor([0.]),
    ))
    #print(f"Vector shape before normalization: {vec.shape}")
    normalized_vec = vec / vec.sum()
    #print(f"Vector shape after normalization: {normalized_vec.shape}")
    return normalized_vec


def aggregate_vit_attention(attn, select_layer=-2, all_prev_layers=True):
    '''Assuming LLaVA-style `select_layer` which is -2 by default'''
    if all_prev_layers:
        avged = []
        for i, layer in enumerate(attn):
            if i > len(attn) + select_layer:
                break
            layer_attns = layer.squeeze(0)
            attns_per_head = layer_attns.mean(dim=0)
            vec = attns_per_head[1:, 1:].cpu() # the first token is <CLS>
            avged.append(vec / vec.sum(-1, keepdim=True))
        return torch.stack(avged).mean(dim=0)
    else:
        layer = attn[select_layer]
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = attns_per_head[1:, 1:].cpu()
        return vec / vec.sum(-1, keepdim=True)


def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HSV)
    hm = np.float32(heatmap) / 255
    cam = hm + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap


### Normalizing functions
def norm_relu(ret_attn):
    for i in range(len(ret_attn)):
        temp = ret_attn[i] / ret_attn[i].max()
        temp = torch.relu(temp)
        ret_attn[i] = temp
    return ret_attn

def norm_min_max(ret_attn):
    for i in range(len(ret_attn)):
        temp = ret_attn[i]
        temp += temp.min()
        temp /= temp.max()
        ret_attn[i] = temp
    return ret_attn

def descending_indices_2d(tensor):
    """
    Converts a 2D tensor into its descending order indices.
    
    Args:
        tensor (torch.Tensor): Input 2D tensor.
        
    Returns:
        torch.Tensor: 2D tensor where each element is replaced with its descending order rank index.
    """
    # 1. Flatten the tensor to 1D
    flat_tensor = tensor.flatten()

    # 2. Sort indices in descending order
    sorted_indices = torch.argsort(flat_tensor)

    # 3. Create rank indices
    ranks = torch.zeros_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(len(flat_tensor), device=tensor.device)

    # 4. Reshape back to 2D
    rank_indices_2d = ranks.view(tensor.shape).to(torch.float32)

    return rank_indices_2d
###


def get_heatmap(
    model,
    outputs,
    tokenizer,
    prompt,
    input_ids,
    current_stage,
    stages,
    image_mask,
    select_token=None,
    image=None,
    save_path=None,
    attn_norm=None):

    # Constructing the LLM attention matrix
    aggregated_prompt_attention = []
    for i, layer in enumerate(outputs["attentions"][0]):
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        cur = attns_per_head[:-1].cpu().clone()
        cur[1:, 0] = 0.0
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        aggregated_prompt_attention.append(cur)
    
    aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)    

    # Constructing the LLM attention matrix
    llm_attn_matrix = heterogenous_stack(
        [torch.tensor([1])]
        + list(aggregated_prompt_attention) 
        + list(map(aggregate_llm_attention, outputs["attentions"]))
    )    

    input_token_len = 0
    for stage in stages:
        mask_patches_per_side, _ = image_mask[stage].shape
        mask_grid = int(mask_patches_per_side * pow(2, stage - max(stages)))
        mask = torch.nn.functional.interpolate(image_mask[stage].unsqueeze(0).unsqueeze(0), size=(mask_grid, mask_grid), mode='nearest')
        patches_in_stage = mask.sum().item()
        input_token_len += patches_in_stage
    input_token_len += len(input_ids[0]) - 1 # -1 for the <image> token
    input_token_len = int(input_token_len)

    output_token_len = 1
    output_token_start = input_token_len
    output_token_end = output_token_start + output_token_len
    output_token_inds = list(range(output_token_start, output_token_end))

    # Initialize results
    ret_attn_list = [torch.zeros_like(image_mask[0]) for _ in output_token_inds]
    stage_attn_list = []
    
    input_offset = 0
    for stage in stages:
        if stage < 0:
            vision_tower = model.model.downsampled_vision_towers[str(stage)]
        else:
            vision_tower = model.model.vision_tower
        
        num_patches = vision_tower.vision_tower.vision_model.embeddings.num_patches
        grid_size = int(math.sqrt(num_patches))
        if stage == 1:
            grid_size *= 2
            
        used_token_list = torch.nn.functional.interpolate(image_mask[stage].unsqueeze(0).unsqueeze(0), size=(grid_size, grid_size), mode='nearest').squeeze()

        if stage == 1:
            block1 = used_token_list[0:24,  0:24].flatten()   # top-left
            block2 = used_token_list[0:24, 24:48].flatten()   # top-right
            block3 = used_token_list[24:48, 0:24].flatten()   # bottom-left
            block4 = used_token_list[24:48, 24:48].flatten()  # bottom-right
            used_token_list = torch.cat([block1, block2, block3, block4], dim=0).tolist()
        else:
            used_token_list = used_token_list.view(-1).tolist()
        
        if sum(used_token_list) == 0:
            # every token in this stage is masked
            continue
        
        index = (input_ids[0] == -200).nonzero(as_tuple=True)[0].item()
        # vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0]) + input_offset
        vision_token_start = len(input_ids[0][:index]) + input_offset
        vision_token_end = vision_token_start + int(sum(used_token_list))
        
        input_offset += int(sum(used_token_list))

        if stage == 1:
            # print("shape: ", llm_attn_matrix.shape)
            # print("text_len: ", len(tokenizer(prompt.split("<image>")[1], return_tensors='pt')["input_ids"][0]))
            stage1_dict = dict()
            vision_token_start_1 = vision_token_start
            for stage1_idx in range(4):
                stage1_dict[stage1_idx] = []
                used_list = used_token_list[num_patches * stage1_idx : num_patches * (stage1_idx + 1)]
                vision_token_end_1 = vision_token_start_1 + int(sum(used_list))
                
                # print("start: ", vision_token_start_1, "end : ", vision_token_end_1)
                
                # Connect with the vision encoder attention
                image_attentions = []
                for i, layer in enumerate(vision_tower.image_attentions):
                    layer = layer[stage1_idx + 1, ...].unsqueeze(0)
                    image_attentions.append(layer)

                vis_attn_matrix = aggregate_vit_attention(
                    image_attentions,
                    select_layer=vision_tower.select_layer,
                    all_prev_layers=True
                )

                # Output
                for i in range(len(output_token_inds)):
                    target_token_ind = output_token_inds[i]
                    attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][vision_token_start_1:vision_token_end_1]        
                    attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()
                    
                    assert attn_weights_over_vis_tokens.shape[0] == sum(used_list), f"{stage},{attn_weights_over_vis_tokens.shape[0]},{sum(used_list)}"

                    if sum(used_list) != 0:
                        attn_over_image = []
                        for idx in range(len(used_list)):
                            # append only used token * weight.
                            # len(attn_weights_over_vis_tokens) is same with sum(used_list).
                            if used_list[idx] == 1:
                                weight = attn_weights_over_vis_tokens[len(attn_over_image)]
                                vis_attn = vis_attn_matrix[idx]
                                vis_attn = vis_attn.reshape(grid_size//2, grid_size//2)
                                attn_over_image.append(vis_attn * weight)

                        attn_over_image = torch.stack(attn_over_image).sum(dim=0)        
                        
                        h, w = ret_attn_list[i].shape
                        attn_over_image = attn_over_image.to(device=model.device)
                        attn_over_image = torch.nn.functional.interpolate(attn_over_image.unsqueeze(0).unsqueeze(0), size=(h//2, w//2), mode='nearest').squeeze()
                        stage1_dict[stage1_idx].append(attn_over_image)
                    else:
                        attn_over_image = torch.zeros_like(ret_attn_list[i])
                        attn_over_image = torch.nn.functional.interpolate(attn_over_image.unsqueeze(0).unsqueeze(0), scale_factor=(0.5, 0.5), mode='nearest').squeeze()
                        stage1_dict[stage1_idx].append(attn_over_image)
                vision_token_start_1 += int(sum(used_list))
            vision_token_end = vision_token_end_1

            for idx, features in enumerate(zip(stage1_dict[0], stage1_dict[1], stage1_dict[2], stage1_dict[3])):
                f1, f2, f3, f4 = features
                top = torch.hstack([f1, f2])
                bot = torch.hstack([f3, f4])
                attn_over_image = torch.vstack([top, bot])
                attn_over_image = attn_over_image * image_mask[stage]
                attn_over_image = attn_over_image / attn_over_image.max()

                attn_over_image = attn_over_image * pow(4, stage + 2 - 1) # -1 to equalize with stage0
                
                if stage == current_stage:
                    stage_attn_list.append(attn_over_image)

                ret_attn_list[idx] = ret_attn_list[idx] + attn_over_image

        else:
            # Connect with the vision encoder attention
            image_attentions = []

            for i, layer in enumerate(vision_tower.image_attentions[0]):
                layer = layer[0, ...].unsqueeze(0)
                image_attentions.append(layer)

            vis_attn_matrix = aggregate_vit_attention(
                image_attentions,
                select_layer=vision_tower.select_layer,
                all_prev_layers=True
            )

            # Output
            for i in range(len(output_token_inds)):
                target_token_ind = output_token_inds[i]        
                attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][vision_token_start:vision_token_end]        
                attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()
                
                assert attn_weights_over_vis_tokens.shape[0] == sum(used_token_list), f"{stage},{attn_weights_over_vis_tokens.shape[0]},{sum(used_token_list)}"

                if len(used_token_list) != 0:
                    attn_over_image = []
                    for idx in range(len(used_token_list)):
                        # append only used token * weight.
                        # len(attn_weights_over_vis_tokens) is same with sum(used_token_list).
                        if used_token_list[idx] == 1:
                            weight = attn_weights_over_vis_tokens[len(attn_over_image)]
                            vis_attn = vis_attn_matrix[idx]
                            vis_attn = vis_attn.reshape(grid_size, grid_size)
                            attn_over_image.append(vis_attn * weight)

                    attn_over_image = torch.stack(attn_over_image).sum(dim=0)        
                    attn_over_image = attn_over_image / attn_over_image.max()
                    
                    h, w = ret_attn_list[i].shape
                    attn_over_image = attn_over_image.to(device=model.device)
                    attn_over_image = torch.nn.functional.interpolate(attn_over_image.unsqueeze(0).unsqueeze(0), size=(h, w), mode='nearest').squeeze()
                    attn_over_image = attn_over_image * image_mask[stage]
                else:
                    attn_over_image = torch.zeros_like(ret_attn_list[i])
                
                attn_over_image = attn_over_image * pow(4, stage + 2)

                if stage == current_stage:
                    stage_attn_list.append(attn_over_image)

                ret_attn_list[i] = ret_attn_list[i] + attn_over_image
            
        if stage == current_stage:
            break

    vis_attn_list = []
    for i in range(len(ret_attn_list)):
        tensor_norm = ret_attn_list[i].clone()
        tensor_norm = descending_indices_2d(tensor_norm)
        tensor_norm = tensor_norm - tensor_norm.min()
        tensor_norm = tensor_norm / tensor_norm.max()
        vis_attn_list.append(tensor_norm)
        
    # Get visual weight at each output token
    overall_attn_weights_over_vis_tokens = []
    vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
    for i, (row, token) in enumerate(
        zip(
            llm_attn_matrix[input_token_len:], 
            outputs["sequences"][0].tolist()
        )
    ):
        overall_attn_weights_over_vis_tokens.append(
            row[vision_token_start:vision_token_end].sum().item()
        )
        
    # ##### norm ####
    # if attn_norm is not None:
    #     ret_attn_list = attn_norm(ret_attn_list)
    # print(stage, current_stage, len(stage_attn_list), len(output_token_inds))
    if select_token is not None:
        ret_attn = ret_attn_list[select_token]
        stage_attn = stage_attn_list[select_token]
    else:
        temp_ret = []
        temp_stage = []
        attn_use_threshold = sum(overall_attn_weights_over_vis_tokens) / len(overall_attn_weights_over_vis_tokens)
        for attn_r, attn_s, w in zip(ret_attn_list, stage_attn_list, overall_attn_weights_over_vis_tokens):
            if w >= attn_use_threshold:
                temp_ret.append(attn_r)
                temp_stage.append(attn_s)
        ret_attn = sum(temp_ret)
        stage_attn = sum(temp_stage) / len(temp_stage)

    # stage_attn_nonzero = stage_attn.flatten()
    # stage_attn_nonzero = stage_attn_nonzero[stage_attn_nonzero != 0]
    # mean = torch.mean(stage_attn_nonzero)
    # std = torch.std(stage_attn_nonzero)
    # z_scores = (stage_attn - mean) / std
    # upper_threshold = 10
    # lower_threshold = -0.2
    # abnormal_mask = (torch.abs(z_scores) < upper_threshold).float()
    # upper_bound = upper_threshold * std + mean
    # upper_bound = 10000
    # lower_bound = lower_threshold * std + mean

    # abnormal_mask = (stage_attn <= 10000000).to(torch.float16)
    # abnormal_mask = (stage_attn <= 1000  * pow(4, current_stage + 2)).to(torch.float16)
    # ret_attn = ret_attn * abnormal_mask
    # stage_attn = stage_attn * abnormal_mask
    
    if image is not None and save_path is not None:
        ret_attn_list.append(ret_attn)
        tensor_norm = descending_indices_2d(ret_attn)
        tensor_norm = tensor_norm - tensor_norm.min()
        tensor_norm = tensor_norm / tensor_norm.max()
        vis_attn_list.append(tensor_norm)
        
        ret_attn_list.append(stage_attn)
        tensor_norm = descending_indices_2d(stage_attn)
        tensor_norm = tensor_norm - tensor_norm.min()
        tensor_norm = tensor_norm / tensor_norm.max()
        vis_attn_list.append(tensor_norm)
        for i, (raw_attn, vis_attn) in enumerate(zip(ret_attn_list, vis_attn_list)):
            if i == len(ret_attn_list) - 2:
                token_string = "total"
            elif i == len(ret_attn_list) - 1:
                token_string = "current_stage_only"
            else:
                token_string = tokenizer.decode(outputs['sequences'][0][i])
            
            # Save attention visualization for the current token and layer
            if token_string == "<s>":
                token_string = "sos"
            elif token_string == "</s>":
                token_string = "eos"
            elif "." in token_string:
                token_string.replace(".", "dot")
            elif "/" in token_string:
                token_string.replace("/", "slash")
            
            ###################
            ### draw histogream
            ###################
            tensor_flattened = raw_attn.flatten()

            nonzero_indices = torch.nonzero(tensor_flattened).squeeze()
            tensor_flattened = tensor_flattened[nonzero_indices].cpu()
            # tensor_flattened = tensor_flattened.cpu()

            bins = int(math.sqrt(len(tensor_flattened)))
            plt.hist(tensor_flattened, bins=500, alpha=0.7, edgecolor='black')
            # plt.xlim(right=max((tensor_flattened.max() * 1.3), upper_bound.cpu()))
            plt.xlim(right=tensor_flattened.max())
            # plt.xlim(left=min(0, lower_bound.cpu() - 0.3))
            # plt.axvline(x=upper_bound.cpu(), color='red', linestyle='--', linewidth=2, label='Upper Line')
            # plt.axvline(x=lower_bound.cpu(), color='blue', linestyle='--', linewidth=2, label='Lower Line')
            plt.title(f"Histogram of 2D Tensor Values, bin: {bins}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.savefig(f"{save_path}/{str(i).zfill(4)}_{token_string}_distribution_02.png")
            plt.close('all')
            plt.hist(tensor_flattened, bins=500, alpha=0.7, edgecolor='black')
            plt.xlim(right=tensor_flattened.max())
            plt.title(f"Histogram of 2D Tensor Values, bin: {bins}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.savefig(f"{save_path}/{str(i).zfill(4)}_{token_string}_distribution.png")
            plt.close('all')
            
            # token_attn = sum(ret_attn_list[:-1])
            # Resize the heatmap to match the original image size
            resized_heatmap = F.interpolate(
                vis_attn.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                size=(image.height, image.width),  # Resize to match image dimensions
                mode='nearest',  # Use bilinear for smoother heatmap
                # align_corners=True
            ).squeeze()
            
            # Normalize the resized heatmap
            # resized_heatmap -= resized_heatmap.min()
            # # resized_heatmap = torch.relu(resized_heatmap)
            # resized_heatmap /= resized_heatmap.max()
            resized_heatmap = resized_heatmap.cpu().numpy()

            np_img = np.array(image)[:, :, ::-1]
            heatmap = (cm.jet(resized_heatmap)[:, :, :3] * 255).astype(np.uint8)
            blended = cv2.addWeighted(np_img, 0.5, heatmap, 0.5, 0)
            img_with_attn = blended
            # img_with_attn = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            # img_with_attn, heatmap = show_mask_on_image(np_img, resized_heatmap)
            # img_with_attn = cv2.cvtColor(img_with_attn, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img_with_attn)
            plt.title(f"Token {token_string}")
            plt.axis("off")
            plt.savefig(f"{save_path}/{str(i).zfill(4)}_{token_string}.png")
            plt.close('all')

        # plot the trend of attention weights over the vision tokens
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(overall_attn_weights_over_vis_tokens)
        ax.set_xticks(range(len(overall_attn_weights_over_vis_tokens)))
        ax.set_xticklabels(
            [tokenizer.decode(token, add_special_tokens=False).strip() for token in outputs["sequences"][0].tolist()],
            rotation=75
        )
        ax.set_title("at each token, the sum of attention weights over all the vision tokens")
        plt.savefig(f"{save_path}/by_token_{token_string}.png")
        plt.close('all')


    return ret_attn


# old make_square
def make_square_center(im, min_size, smallest_grid_size, fill_color=(0, 0, 0)):
    x, y = im.size
    size = (max(min_size, x, y))
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im, size, x, y


# one-side padding
def make_square_top_left(im, min_size, smallest_grid_size, fill_color=(0, 0, 0)):
    x, y = im.size
    size = int(max(min_size, x, y))
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (0, 0))

    return new_im, size, x, y

# one-side padding
def make_square_bot_right(im, min_size, smallest_grid_size, fill_color=(0, 0, 0)):
    x, y = im.size
    size = int(max(min_size, x, y))
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (size - x, size - y))

    return new_im, size, x, y


def get_cropped_mask(image_mask, idx):
    h, w = image_mask.shape
    if idx == 0:
        return image_mask[:h//2, :w//2]
    elif idx == 1:
        return image_mask[:h//2, w//2:]
    elif idx == 2:
        return image_mask[h//2:, w//2:]
    elif idx == 3:
        return image_mask[h//2:, :w//2]
    else:
        return 0

