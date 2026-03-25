import json
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datasets import load_dataset
from accelerate import Accelerator
from transformers import Sam3Processor, Sam3Model
import numpy as np
import torch
from tqdm import tqdm 


def collate_batch(samples):
    return {
        "image": [sample["image"] for sample in samples],
    }

bsize = 24
accelerator = Accelerator()
device = accelerator.device
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

#load noun phrases jsonl
noun_phrases = []
with open("noun_phrases.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        noun_phrases.append(data["generated_text"])
        
os.makedirs("segmentation_masks", exist_ok=True)
max_idx = max([int(fn.split(".")[0]) for fn in os.listdir("segmentation_masks") if fn.endswith(".png")], default=-1)

dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train")
dataset_size = len(dataset)
dataloader = DataLoader(dataset, batch_size=bsize, shuffle=False, collate_fn=collate_batch)
model, dataloader = accelerator.prepare(model, dataloader)

for local_batch_idx, sample in tqdm(
    enumerate(dataloader),
    total=len(dataloader),
    desc="Processing Images",
    disable=not accelerator.is_local_main_process,
): 
    global_batch_idx = local_batch_idx * accelerator.num_processes + accelerator.process_index
    start_idx = global_batch_idx * bsize
    batch_size_current = len(sample["image"])
    batch_indices = [idx for idx in range(start_idx, start_idx + batch_size_current) if idx <= dataset_size]
    if not batch_indices:
        continue
    if all(idx <= max_idx for idx in batch_indices):
        continue
    batch_noun_phrases = [noun_phrases[idx] for idx in batch_indices]
    inputs = processor(images=sample["image"], text=batch_noun_phrases, return_tensors="pt").to(device)
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
        global_idx = batch_indices[j]
        if global_idx <= max_idx:
            continue
        if result is None:
            continue
        masks = result["masks"].cpu().numpy()
        if masks.shape[0] == 0:
            continue
        mask = np.clip(masks.sum(axis=0), 0, 1)
        foreground_ratio = mask.mean()
        if foreground_ratio > 0.6:
            continue
        Image.fromarray((mask * 255).astype(np.uint8)).save(f"segmentation_masks/{global_idx}.png")