import json
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasets import load_dataset
from transformers import Sam3Processor, Sam3Model
import numpy as np
import torch
from tqdm import tqdm 


def collate_batch(samples):
    return {
        "image": [sample["image"] for sample in samples],
        "idx": [sample["idx"] for sample in samples],
    }

bsize = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

#load noun phrases jsonl
noun_phrases = []
with open("noun_phrases.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        noun_phrases.append(data["generated_text"])
dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train")
dataloader = DataLoader(dataset, batch_size=bsize, shuffle=False, collate_fn=collate_batch)
for i, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing Images"): 
    if os.path.exists(f"segmentation_masks/{(i+1)*bsize}.png"):
        continue
    inputs = processor(images=sample["image"], text=noun_phrases[i*bsize:(i+1)*bsize], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.15,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )

    # save results as png
    for j, result in enumerate(results):
        if result is None:
            continue
        masks = result["masks"].cpu().numpy()
        if masks.shape[0] == 0:
            continue
        mask = np.clip(masks.sum(axis=0), 0, 1)
        foreground_ratio = mask.mean()
        if foreground_ratio > 0.6:
            continue
        Image.fromarray((mask * 255).astype(np.uint8)).save(f"segmentation_masks/{i*bsize+j}.png")