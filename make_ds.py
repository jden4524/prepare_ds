from datasets import Dataset, Image, load_dataset
import os
from PIL import Image as PILImage
import json
from tqdm import tqdm

# Load source dataset and noun phrases
dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train")
noun_phrases = []
with open("noun_phrases.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        noun_phrases.append(data["generated_text"])

# Get list of file names in segmentation_masks
seg_mask_fn = set(os.listdir("segmentation_masks"))

def data_generator():
    """Generator function that yields processed data samples."""
    skipped = 0
    for i in tqdm(range(len(dataset)), desc="Processing dataset"):
        
        # Check if segmentation mask exists
        if f"{i}.png" not in seg_mask_fn:
            continue
        
        # Check if phrase is in answer
        phrase = noun_phrases[i]
        answer = dataset[i]["chosen"]
        if phrase.lower() not in answer.lower():
            skipped += 1
            continue
        
        # Yield the processed sample
        yield {
            "idx": i,
            "image": dataset[i]["image"],
            "question": dataset[i]["question"],
            "answer": answer,
            "mask": PILImage.open(f"segmentation_masks/{i}.png").convert("L"),
            "phrase": phrase
        }
    
    print(f"Skipped {skipped} samples")

# Create the dataset using from_generator
new_dataset = Dataset.from_generator(
    data_generator,
    features=None  # Auto-infer features, or specify manually if needed
)

print(f"Final dataset size: {len(new_dataset)}")
new_dataset.push_to_hub("Jackie2235/RLAIF-V-SEG", num_shards = 4)