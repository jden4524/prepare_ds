from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset
import json

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train")

sentences = dataset["chosen"] 
dataset_indices = dataset["idx"]

messages_list = [
    [{"role": "system", "content": "Extract exactly one specific, individual physical object or region from the image description as a noun, or a short noun phrase including key modifiers (e.g. color, size, location). Exclude categories and collections. The output must be a literal substring from the text. "},
     {"role": "user", "content": prompt}]
    for prompt in sentences
]
texts = tokenizer.apply_chat_template(
    messages_list,
    tokenize=False,
    add_generation_prompt=True,
)


def main():
    llm = LLM(model="Qwen/Qwen3-4B-Instruct-2507", 
        max_model_len=4096,
        gpu_memory_utilization=0.8)

    outputs = llm.generate(texts)
    output_mapping_path = "noun_phrases.jsonl"
    with open(output_mapping_path, "w", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            ds_idx = dataset_indices[i]
            generated_text = output.outputs[0].text
            f.write(json.dumps({"idx": ds_idx, "generated_text": generated_text}, ensure_ascii=False) + "\n")

    print(f"Saved mapping to {output_mapping_path}")
    
    
if __name__ == "__main__":
    main()