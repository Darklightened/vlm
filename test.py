import os
import sys
sys.path.append("./models")
import monai.losses
import numpy as np
import cv2

import torch
import torch.nn.functional as F
import torch.nn as nn

from models.llava.model.builder import load_pretrained_model
from models.llava.utils import disable_torch_init
from models.llava.mm_utils import get_model_name_from_path

from utils import (
    show_mask_on_image,
    preprocess_prompt,
    preprocess_image,
    get_heatmap,
)

from torchvision.datasets import CocoDetection
import random
import monai
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="simple parser")
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    # ===> specify the model path
    model_path = "liuhaotian/llava-v1.5-7b"

    # load the model
    load_8bit = False
    load_4bit = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device="cpu"

    print(f"device: {device}")
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        None, # model_base
        model_name, 
        load_8bit, 
        load_4bit, 
        device=device
    )
    print("model loaded")

    resized_image_size = args.image_size        

    # patch_size = 14
    # num_patches = (resized_image_size // patch_size) ** 2
    # num_positions = num_patches + 1
    # embed_dim = model.model.vision_tower.vision_tower.vision_model.embeddings.embed_dim

    # model.model.vision_tower.vision_tower.vision_model.embeddings.image_size = resized_image_size
    # model.model.vision_tower.vision_tower.vision_model.embeddings.num_patches = num_patches
    # model.model.vision_tower.vision_tower.vision_model.embeddings.num_positions = num_positions
    # model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(num_positions, embed_dim).to(device)
    # model.model.vision_tower.vision_tower.vision_model.embeddings.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent=False)
    # model.to(device)

    print(f"image_size: {args.image_size}")

    # Dataset path
    data_dir = './val2017'
    Caption_file = './annotations/captions_val2017.json' #Caption
    instances_file = './annotations/instances_val2017.json' #Segmentation, category_id, bbox
    keypoints_file = './annotations/person_keypoints_val2017.json' #Segmentation, keypoints, id, bbox

    with open(args.output_path, "w") as f:
        f.write(f"img_idx,cat_idx,hallucination,is_positive,cat_name\n")

    # Load dataset
    dataset = CocoDetection(root=data_dir, annFile=instances_file)
    coco = dataset.coco

    # Categories
    categories = coco.loadCats(coco.getCatIds())
    categories_list = [cat['name'] for cat in categories]

    for img_idx in tqdm(range(len(dataset)), desc="Processing"):
        image, target = dataset[img_idx]
        
        # Create semantic mask from instance mask.
        mask_dict = {}
        for obj in target:
            # Get category name
            category_id = coco.getCatIds(catIds=[obj["category_id"]])
            name = coco.loadCats(category_id)[0]['name']
            
            # Get instance mask
            segmentation = obj["segmentation"]
            if type(segmentation) != list:
                continue
            mask = np.zeros_like(image, dtype=np.uint8)
            poly = np.array(segmentation[0]).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], (255, 255, 255))
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Merge to semantic mask
            semantic_mask = mask_dict.get(name)
            if semantic_mask is not None:
                semantic_mask = semantic_mask | mask
            else:
                semantic_mask = mask
            
            mask_dict[name] = semantic_mask
        
        # Create sorted list by mask area
        positive_list = sorted(mask_dict, key=lambda x: mask_dict[x].sum(), reverse=True)
        # Create non-existent objects list
        negative_list = [name for name in categories_list if name not in positive_list]

        # Predict top-k-area positives & k-random negatives
        k = 0
        if len(positive_list) > 3:
            k = 3
        else:
            k = len(positive_list)

        predict_list = positive_list[:k]
        predict_list += random.sample(negative_list, k)

        for cat_idx, cat_name in enumerate(predict_list):
            mask = mask_dict.get(cat_name, np.zeros((image.height, image.width), dtype=np.uint8))
            prompt_text = f"Does there exist {cat_name} in the image? Answer in 'Yes' or 'No'"
            #print(f"\n{prompt_text}\n")

            input_ids, prompt = preprocess_prompt(model, model_name, prompt_text, tokenizer)
            image, image_tensor = preprocess_image(model, image_processor, image)
            image_size = image.size
            ################################################
            ids_list = input_ids.tolist()[0]
            ids_list.append(2)
            input_ids_temp = torch.tensor(ids_list)

            resized_image_tensor = F.interpolate(image_tensor, size=(resized_image_size, resized_image_size), mode='bilinear', align_corners=False)
        
            # generate the response
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    images=resized_image_tensor,
                    image_sizes=[image_size],
                    do_sample=False,
                    max_new_tokens=512,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_attentions=True,
                )

            text = tokenizer.decode(outputs["sequences"][0]).strip()
            # print(text)
                
            answer = ""
            if "Yes" in text:
                answer = "yes"
            elif "No" in text:
                answer = "no"
            
            is_positive = cat_idx < k
            hallucination = (is_positive and answer == "no") or (not is_positive and answer == "yes")

            with open(args.output_path, "a") as f:
                f.write(f"{img_idx},{cat_idx},{hallucination},{is_positive},{cat_name}\n")

if __name__ == '__main__':
    main()