import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
import re
import argparse

import torch
from transformers import pipeline


def read_matrices_from_file(file_path):
    """
    Read matrices stored in a plain-text file and convert them to a float32 NumPy array.
    Each matrix is expected to be wrapped by [[ ... ]].
    """
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

    text_features = np.vstack(matrix_arrays).astype(np.float32)
    return text_features


def main(args):
    # Load PointLLM / prompt results JSON
    with open(args.PointLLM_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Keep the original prompt logic unchanged
    base_prompt = data["prompt"] + ""
    base = "Output ONLY a numerical <score></sore> . Do not provide additional explanations."
    model_outputs = [item["model_output"] for item in data["results"]]

    # Build local generation pipeline
    pipe = pipeline(
        "text-generation",
        model=args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    # Load feature matrix file
    point_features = read_matrices_from_file(args.features_path)  # shape = (N, D)
    print("Loaded shape:", point_features.shape)

    num_samples = point_features.shape[0]

    # Build KNN graph
    nbrs = NearestNeighbors(n_neighbors=11, metric='euclidean')
    nbrs.fit(point_features)

    distances, indices = nbrs.kneighbors(point_features)

    knn_matrix = np.zeros((num_samples, num_samples), dtype=np.float32)
    rows = np.arange(num_samples).repeat(10)
    cols = indices[:, 1:11].reshape(-1)
    vals = np.exp(-distances[:, 1:11].reshape(-1))

    knn_matrix[rows, cols] = vals
    results = []

    # Score each object
    for idx, output in enumerate(model_outputs):
        raw_content = None
        try:
            found_number = False
            for attempt in range(5):
                # Select top-3 neighbors from the KNN row
                row_index = idx
                row = knn_matrix[row_index]
                top3_cols = np.argsort(row)[-3:][::-1]

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

                messages = [
                    {"role": "system", "content": "You are a helpful assistant for 3D object understanding."},
                    {"role": "user", "content": full_query},
                ]

                outputs = pipe(messages)

                gen = outputs[0]["generated_text"]
                if isinstance(gen, list):
                    last_msg = gen[-1]
                    if isinstance(last_msg, dict) and "content" in last_msg:
                        raw_content = last_msg["content"].strip()
                    else:
                        raw_content = str(last_msg).strip()
                else:
                    raw_content = str(gen).strip()

                print(f"object_id {idx}, attempt {attempt + 1}/5, raw_content: {raw_content}")

                score_pattern = r"<score>\s*(-?\d+(?:\.\d+)?)\s*</score>"
                nums_str = re.findall(score_pattern, raw_content, flags=re.IGNORECASE)

                if not nums_str:
                    if attempt < 4:
                        print(f"  -> No <score> number found, retrying ({attempt + 1}/5)...")
                        continue
                    else:
                        print(f"  -> No <score> number found after 5 attempts, record raw_content only.")
                        break

                nums = [float(x) for x in nums_str]
                max_num = max(nums)
                score = int(max_num)

                if not 0 <= score <= 100:
                    raise ValueError(f"Score out of range: {score}")

                found_number = True
                results.append({
                    "object_id": idx,
                    "score": score,
                    "raw_response": raw_content
                })
                print(f"  -> Parsed score: {score}")
                break

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

    # Save results
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
        "--output_path",
        type=str,
        default="results_gptoss.json",
        help="Path to save the output JSON results"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="openai/gpt-oss-20b",
        help="Hugging Face model id for text generation"
    )

    args = parser.parse_args()
    main(args)