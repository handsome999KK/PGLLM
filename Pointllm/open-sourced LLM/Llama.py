import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import argparse


def read_matrices_from_file(file_path):
    """
    Read matrices stored in a plain-text file and convert them to a float32 NumPy array.
    Each matrix is expected to be wrapped by [[ ... ]].
    """
    with open(file_path, 'r') as file:
        content = file.read()

    # Split all matrices (assume each matrix is enclosed by [[ ... ]])
    matrices = []
    current_matrix = []
    in_matrix = False

    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('[['):
            in_matrix = True
            current_matrix = [line[2:]]  # Remove leading [[
        elif line.endswith(']]'):
            in_matrix = False
            current_matrix.append(line[:-2])  # Remove trailing ]]
            matrices.append(current_matrix)
            current_matrix = []
        elif in_matrix:
            current_matrix.append(line)

    # Convert each matrix string to a NumPy array
    matrix_arrays = []
    for matrix in matrices:
        matrix_str = ' '.join(matrix)
        matrix_str = matrix_str.replace('[', '').replace(']', '')
        matrix_data = np.fromstring(matrix_str, sep=' ')
        matrix_arrays.append(matrix_data)

    # Stack all matrices into one NumPy array
    text_features = np.vstack(matrix_arrays).astype(np.float32)
    return text_features


def main(args):
    # ================== Load data ==================
    with open(args.PointLLM_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Keep original prompt logic unchanged
    base_prompt = data["prompt"] + ""
    base = "Output ONLY a numerical <score></score>. Do not provide additional explanations."
    model_outputs = [item["model_output"] for item in data["results"]]

    # ================== Load local Llama model ==================
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,   # Or torch.float16
        device_map="auto",            # Multi-GPU model parallelism
    )
    model.eval()

    # ================== Read features & build KNN ==================
    point_features = read_matrices_from_file(args.features_path)   # shape = (N, D)
    print("Loaded shape:", point_features.shape)

    num_samples = point_features.shape[0]

    nbrs = NearestNeighbors(n_neighbors=11, metric='euclidean')
    nbrs.fit(point_features)

    # Get neighbor indices and distances for each sample
    distances, indices = nbrs.kneighbors(point_features)  # both shapes: (N, 11)

    # Initialize adjacency / similarity matrix
    knn_matrix = np.zeros((num_samples, num_samples), dtype=np.float32)

    # Vectorized fill
    rows = np.arange(num_samples).repeat(10)              # Repeat each row 10 times (excluding self)
    cols = indices[:, 1:11].reshape(-1)                   # Exclude first neighbor (self), flatten
    vals = np.exp(-distances[:, 1:11].reshape(-1))        # Compute exp(-distance), flatten

    knn_matrix[rows, cols] = vals

    # ================== Iterate over model outputs and run Llama inference ==================
    results = []

    for idx, output in enumerate(model_outputs):
        raw_content = None
        try:
            found_number = False  # Whether a valid score is parsed for this sample

            # Retry up to 5 times
            for attempt in range(5):
                # Select top-3 neighbors
                row_index = idx
                row = knn_matrix[row_index]
                top3_cols = np.argsort(row)[-3:][::-1]  # Sort descending

                PROMPT_LP2 = []
                first_item = True

                for item in data["results"]:
                    if item["object_id"] in top3_cols:
                        if first_item:
                            PROMPT_LP2.append(item["model_output"])
                            first_item = False
                        else:
                            PROMPT_LP2.append("\n" + item["model_output"])

                PROMPT_LP2 = "".join(PROMPT_LP2)

                PROMPT1 = "The description requiring probability calculation: " + output
                PROMPT2 = "Descriptions of 3D objects with similar features: " + PROMPT_LP2

                full_query = f"{base_prompt}\n\n{PROMPT1}\n\n{PROMPT2}\n\n{base}"

                # Build Llama chat format
                messages = [
                    {"role": "system", "content": "You are a helpful assistant for 3D object understanding."},
                    {"role": "user", "content": full_query},
                ]

                chat_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                model_inputs = tokenizer([chat_text], return_tensors="pt")
                # If you want to force a single GPU, you can do:
                # model_inputs = {k: v.to("cuda:0") for k, v in model_inputs.items()}

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=256,  # More than enough for a score + a few tokens
                        do_sample=False,
                        temperature=0.0
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(model_inputs["input_ids"], generated_ids)
                ]

                responses = tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                raw_content = responses[0].strip()
                print(f"object_id {idx}, attempt {attempt+1}/5, raw_content: {raw_content}")

                # Extract all numbers from raw_content and take the maximum one
                nums_str = re.findall(r"-?\d+(?:\.\d+)?", raw_content)
                if not nums_str:
                    if attempt < 4:
                        print(f"  -> No number found, retrying ({attempt+1}/5)...")
                        continue
                    else:
                        print(f"  -> No number found after 5 attempts, record raw_content only.")
                        break

                nums = [float(x) for x in nums_str]
                max_num = max(nums)
                score = int(max_num)  # You can change to round(max_num) if needed

                if not 0 <= score <= 100:
                    raise ValueError(f"Score out of range: {score}")

                found_number = True
                results.append({
                    "object_id": idx,
                    "score": score,
                    "raw_response": raw_content
                })
                print(f"  -> Parsed score: {score}")
                break  # Success for this sample, stop retry loop

            # Case: failed to parse a valid number after 5 attempts
            if not found_number:
                results.append({
                    "object_id": idx,
                    "score": None,
                    "error": "No valid number found after 5 attempts",
                    "raw_response": raw_content
                })

        except Exception as e:
            print(f"Error processing object_id {idx}: {str(e)}")
            results.append({
                "object_id": idx,
                "score": None,
                "error": str(e),
                "raw_response": raw_content if raw_content is not None else None
            })

    # ================== Save results ==================
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Processing completed. Results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features_path",
        type=str,
        required=True,
        help="Path to the feature text file (e.g., concat_f_values_MN.txt)"
    )
    parser.add_argument(
        "--PointLLM_results_path",
        type=str,
        required=True,
        help="Path to the JSON file containing prompt/results"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the local Llama model"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results_llama.json",
        help="Path to save the output JSON results"
    )

    args = parser.parse_args()
    main(args)