import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
import re
import torch
import argparse
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


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
    base = "Output ONLY a numerical score. Do not provide additional explanations."
    model_outputs = [item["model_output"] for item in data["results"]]

    # ================== Load local Qwen3-VL model ==================
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    # ================== Read features & build KNN ==================
    point_features = read_matrices_from_file(args.features_path)
    print("Loaded shape:", point_features.shape)

    num_samples = point_features.shape[0]

    nbrs = NearestNeighbors(n_neighbors=11, metric='euclidean')
    nbrs.fit(point_features)
    distances, indices = nbrs.kneighbors(point_features)

    # Initialize adjacency / similarity matrix
    knn_matrix = np.zeros((num_samples, num_samples), dtype=np.float32)

    # Vectorized fill
    rows = np.arange(num_samples).repeat(10)               # Repeat each row 10 times (excluding self)
    cols = indices[:, 1:11].reshape(-1)                    # Exclude self-neighbor, then flatten
    vals = np.exp(-distances[:, 1:11].reshape(-1))         # exp(-distance)

    knn_matrix[rows, cols] = vals

    # ================== Iterate over each sample and run Qwen3 inference ==================
    results = []

    for idx, output in enumerate(model_outputs):
        raw_content = None
        try:
            # Build full query with top-3 nearest neighbors
            row_index = idx
            row = knn_matrix[row_index]
            top3_cols = np.argsort(row)[-3:][::-1]  # Descending order, take top 3

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

            # Build Qwen3 chat format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": full_query
                        }
                    ],
                }
            ]

            # Prepare model inputs
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(model.device)

            # Inference (short output is enough)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_texts = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            raw_content = output_texts[0].strip()

            # Extract all valid scores (0~100) and take the maximum
            matches = re.findall(r'\b(100|\d{1,2})\b', raw_content)
            if not matches:
                raise ValueError(f"No valid score found in response: {raw_content}")

            scores = [int(s) for s in matches]
            score = max(scores)

            if not 0 <= score <= 100:
                raise ValueError(f"Score out of range: {score}")

            results.append({
                "object_id": idx,
                "score": score,
                "raw_response": raw_content
            })
            print(f"Score of object {idx} is {score}")

        except Exception as e:
            print(f"Error processing object_id {idx}: {str(e)}")
            results.append({
                "object_id": idx,
                "score": None,
                "error": str(e),
                "raw_response": raw_content
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
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen3-VL model name or local path"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results_qwen3vl.json",
        help="Path to save the output JSON results"
    )

    args = parser.parse_args()
    main(args)