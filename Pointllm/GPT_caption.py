import json
import requests
import re
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import time 


def read_matrices_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    matrices = []
    current_matrix = []
    in_matrix = False

    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('[['):
            in_matrix = True
            current_matrix = [line[2:]] 
        elif line.endswith(']]'):
            in_matrix = False
            current_matrix.append(line[:-2])
            matrices.append(current_matrix)
            current_matrix = []
        elif in_matrix:
            current_matrix.append(line)

    matrix_arrays = []
    for matrix in matrices:
        matrix_str = ' '.join(matrix)
        matrix_str = matrix_str.replace('[', '').replace(']', '')
        matrix_data = np.fromstring(matrix_str, sep=' ')
        matrix_arrays.append(matrix_data)

    text_features = torch.tensor(np.vstack(matrix_arrays), dtype=torch.float32)
    return text_features


with open(
        'PointLLM-master/results/Prompt/caption/PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json',
        'r', encoding='utf-8') as f:
    data = json.load(f)

base_prompt = data["prompt"] + ""
base = "Output ONLY a 3D description. And don't describe too much."
model_outputs = [item["model_output"] for item in data["results"]]

API_KEY = ""  # replace your ChatGPT API
API_ENDPOINT = "https://api.atalk-ai.com/v2/chat/completions"  
headers = {
    "Authorization": API_KEY,  
    "Content-Type": "application/json"
}

file_path = '/home/kk/PointLLM/PointLLM-master/concat_f_values_caption.txt'
point_features = read_matrices_from_file(file_path)

nbrs = NearestNeighbors(n_neighbors=4, metric='euclidean')
nbrs.fit(point_features)

distances, indices = nbrs.kneighbors(point_features) 

knn_matrix = np.zeros((point_features.shape[0], point_features.shape[0]))

rows = np.arange(point_features.shape[0]).repeat(3)
cols = indices[:, 1:4].reshape(-1)
vals = np.exp(-distances[:, 1:4].reshape(-1))

knn_matrix[rows, cols] = vals

results = []

for idx, output in enumerate(model_outputs):
    start_time = time.time()
    max_retries = 10
    retry_count = 0
    processed = False

    while not processed and retry_count < max_retries:
        try:
            row_index = idx
            row = knn_matrix[row_index]
            top5_cols = np.argsort(row)[-3:][::-1] 

            PROMPT_LP2 = []
            first_item = True  

            for idx_in_data, item in enumerate(data["results"]):
                if idx_in_data in top5_cols:
                    if first_item:
                        PROMPT_LP2.append(item["model_output"])
                        first_item = False
                    else:
                        PROMPT_LP2.append("\n" + item["model_output"])
            PROMPT_LP2 = "".join(PROMPT_LP2)

            PROMPT1 = "The description requiring optimize: " + output
            PROMPT2 = "Descriptions of 3D objects with similar features: " + PROMPT_LP2
            full_query = f"{base_prompt}\n\n{PROMPT1}\n\n{PROMPT2}\n\n{base}"
            payload = {
                "model": "gpt-4o",  
                "messages": [{"role": "user", "content": full_query}],
                "temperature": 0.1,
                "stream": False
            }

            response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=60)
            response.raise_for_status()  

            response_data = response.json()

            raw_content = response_data["choices"][0]["message"]["content"].strip()

            print(f"API返回的文字内容: {raw_content}")

            results.append({
                "object_id": idx,
                "description": raw_content,
            })

            processed = True 
            print(f"Score of object {idx} is finished")

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

with open('caption_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Processing completed. Results saved to caption_results.json")