
import json
import requests
import re
import numpy as np
import torch

import time  




with open(
        '',
        'r', encoding='utf-8') as f:
    data = json.load(f)

base_prompt = """Evaluate a model-generated caption against a human-generated caption (ground truth) for a 3D model. Identify the aspects mentioned in the human caption and calculate the percentage of these aspects correctly mentioned or partially matched in the model caption. Score from 0 to 100, where each aspect contributes equally to the score. Consider similar concepts for partial score."

Provide your score (0-100) in the format of 'score'

Example:
Human: Pink statue looks like a cat with circular hair.
Model: This is a 3D model of a cartoon-style statue, designed in a white color. It bears an animated or comical appearance rather than a realistic one. The statue might be of a character or a funny interpretation of a common object. It could be used in animation, game development or as a digital mascot due to its cartoon nature. The white color makes it versatile, allowing the viewer to imagine it with any color according to the context.
Output: 50

Now score the following:"""
base = "Output ONLY a numerical score. Do not provide additional explanations."
model_outputs = [item["description"] for item in data]


API_KEY = ""  # replace your ChatGPT API
API_ENDPOINT = "https://api.atalk-ai.com/v2/chat/completions"  
headers = {
    "Authorization": API_KEY,  
    "Content-Type": "application/json"
}

results = []
with open('data/anno_data/PointLLM_brief_description_val_200_GT.json', 'r') as f:
    json_data = json.load(f)


for idx, output in enumerate(model_outputs):
    start_time = time.time() 
    max_retries = 10 
    retry_count = 0
    processed = False
    if idx < len(json_data):
        obj_id = json_data[idx]["object_id"]
        for conv in json_data[idx]["conversations"]:
            if conv["from"] == "gpt":
                gpt_description = conv["value"]
                break
    else:
        print(f"第 {idx} 个索引中gpy value无效")
        gpt_description = ""
    while not processed and retry_count < max_retries:
        try:


            PROMPT1 =  "Human: " + gpt_description
            PROMPT2 = "Model: " + output
            full_query = f"{base_prompt}\n{PROMPT1}\n{PROMPT2}\n\n{base}"
            payload = {
                "model": "gpt-4.1", 
                "messages": [{"role": "user", "content": full_query}],
                "temperature": 0.1,
                "stream": False
            }

            response = requests.post(API_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()

            response_data = response.json()
            raw_content = response_data["choices"][0]["message"]["content"].strip()

            match = re.search(r'\b(100|\d{1,2})\b', raw_content)
            if not match:
                raise ValueError(f"No valid score found in response: {raw_content}")

            score = int(match.group())

            if not 0 <= score <= 100:
                raise ValueError(f"Score out of range: {score}")

            results.append({
                "object_id": idx,
                "score": score,
                "raw_response": raw_content 
            })

            processed = True 
            print(f"Score of object {idx} is {score}")

        except Exception as e:
            current_time = time.time()
            elapsed = current_time - start_time

            if elapsed > 300: 
                print(f"Object {idx} timed out after {elapsed:.2f} seconds, retrying...")
                retry_count += 1
                start_time = time.time()  
                time.sleep(1)  
            else:
                print(f"Error processing object {idx} (attempt {retry_count + 1}): {str(e)}")
                if retry_count < max_retries - 1:
                    time.sleep(2)  
                retry_count += 1

            if retry_count >= max_retries and not processed:
                print(f"Object {idx} failed after {max_retries} attempts")
                results.append({
                    "object_id": idx,
                    "score": None,
                    "error": str(e),
                    "raw_response": raw_content if 'raw_content' in locals() else None
                })
                processed = True  
                break

with open('GPT_caption_score_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Processing completed. Results saved to classification_results.json")


