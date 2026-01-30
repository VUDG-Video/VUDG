import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import pandas as pd
import os, json
from tqdm import tqdm, trange
import cv2
import gc
import argparse


# ================== Model Path ==================
base_model_path = "Qwen2.5-VL-7B-Instruct"

print("Loading base model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
model.eval()

processor = AutoProcessor.from_pretrained(base_model_path)

def generate_question_prompt(question):
    return question + "\n\nAnswer with only the letter of the correct choice."

def chat(video_path, text_prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1,
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    try:
        image_inputs, video_inputs = process_vision_info(messages)
    except Exception as e:
        print(f"Error in processing vision info: {e}")
        return None

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=16)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # clean cache
    del inputs, generated_ids, generated_ids_trimmed, image_inputs, video_inputs
    gc.collect()
    torch.cuda.empty_cache()

    return output_text[0].strip().upper()

# ================== main ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Qwen2.5 Video QA with LoRA (no merge)')
    parser.add_argument('--input_path', type=str,
                        default='./data/test_multi.json',
                        help='Path to input JSON file')
    parser.add_argument('--output_path', type=str,
                        default='./results/Qwen2.5_7b.json',
                        help='Path to output JSON file')
    parser.add_argument('--ds_path', type=str, default="./data/testset")
    args = parser.parse_args()

    # Domains
    DOMAINS = {
        'style': ['cartoon', 'movie-TVshow', 'game', 'virtual'],
        'viewpoint': ['shaky', 'egocentric video', 'surveillance'],
        'env': ['rainy', 'snowy', 'night', 'foggy']
    }

    eval_domains = []
    for d in DOMAINS.values():
        eval_domains.extend(d)
    

    print("Domains to evaluate:", eval_domains)

    with open(args.input_path, 'r', encoding='utf-8') as f:
        rows = json.load(f)

    rows = [row for row in rows if row['domain'] in eval_domains]

    for idx, item in enumerate(tqdm(rows, desc="Processing")):
        video = item['video']
        domain = item['domain']
        video_path = os.path.join(args.ds_path, domain, video)

        if 'pred' in item and item['pred'] != "Error":
            continue

        try:
            question = item['question']
            text_prompt = generate_question_prompt(question)
            answer = chat(video_path, text_prompt)
            tqdm.write(f"Pred: {answer}")

            rows[idx]['pred'] = answer

            # save results every 10 run
            if (idx + 1) % 10 == 0:
                with open(args.output_path, 'w', encoding='utf-8') as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2, separators=(',', ': '))

        except Exception as e:
            tqdm.write(f"Error processing row {idx}: {str(e)}")
            rows[idx]['pred'] = "Error"

    # save results
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, separators=(',', ': '))

    print("✅ Inference completed.")