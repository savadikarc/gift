import json
import torch
from data import load_gsm8k
from utils import model_inference, initialize_text_to_text_model
import re
import os
from tqdm import tqdm
from peft import get_peft_model, PeftModelForCausalLM
import argparse
from pprint import pprint

def extract_num(text):
    # Regex pattern to find the number following '####'
    pattern = r'####\s*(\d+)'
    # Using re.search to find the first match
    match = re.search(pattern, text)
    if match:
        result = match.group(1)
    else:
        print(text)
        result = ""
    try:
        return int(result.replace(",", ""))
    except:
        print(f"'{result}' can't be converted")
        return 0

def main(model_name, checkpoint_dir, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_set = load_gsm8k()
    model_type = "CausalLM"
    model, tokenizer = initialize_text_to_text_model(
        model_name, model_type, True, tokenizer="meta-llama/Llama-2-7b-hf",flash_attention=True
    )
    model = model.to(device)
    wrapped_model: PeftModelForCausalLM = PeftModelForCausalLM.from_pretrained(model, checkpoint_dir, device_map=device)
    all = 0
    correct = 0
    generations = []
    t = tqdm(test_set)
    for example in t:
        pred_text = model_inference(wrapped_model, tokenizer, example['x'], model_type, max_target_length=512)
        d = {"x": example["x"], "y": example["y"], "pred": pred_text}
        pprint(d)
        generations.append(d)
        gt = extract_num(example["y"])
        pred = extract_num(pred_text)
        correct += int(gt==pred)
        all += 1
        t.set_description(f"Accuracy: {correct/all*100:02f}%")
    
    os.makedirs(output_dir, exist_ok=True)

    generation_file = os.path.join(output_dir, "generations.json")
    with open(generation_file, "w") as f:
        json.dump(generations, f, indent=4)

    print("Acc:", correct/all)
    # append to gsm8k_results.txt (create if not exists)
    metrics_file = os.path.join(output_dir, "gsm8k_results.txt")
    if not os.path.exists(metrics_file):
        with open("gsm8k_results.txt", "w") as f:
            f.write("Model Acc\n")
    with open(metrics_file, "a") as f:
        f.write(f"{model_name} {correct/all}\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    main(args.model_name, args.checkpoint_dir, args.output_dir)
