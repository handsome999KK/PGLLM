import json
import requests
import re
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import time
import argparse
import os


def read_matrices_from_file(file_path):
  
    ext = os.path.splitext(file_path)[1].lower()

    # ---------- 1) NPY ----------
    if ext == ".npy":
        arr = np.load(file_path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return torch.tensor(arr, dtype=torch.float32)

    # ---------- 2) NPZ ----------
    if ext == ".npz":
        data = np.load(file_path)
        if "features" in data:
            arr = data["features"]
        else:
            # Use the first available array
            first_key = list(data.keys())[0]
            arr = data[first_key]
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return torch.tensor(arr, dtype=torch.float32)

    # ---------- 3) JSON ----------
    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # Common case 1: {"features": [[...], [...]]}
        if isinstance(obj, dict):
            if "features" in obj:
                arr = np.array(obj["features"], dtype=np.float32)
            else:
                # Try to find the first list-like field
                found = None
                for _, v in obj.items():
                    if isinstance(v, list):
                        found = v
                        break
                if found is None:
                    raise ValueError("No valid feature array found in JSON (e.g., key='features').")
                arr = np.array(found, dtype=np.float32)

        # Common case 2: [[...], [...]]
        elif isinstance(obj, list):
            arr = np.array(obj, dtype=np.float32)

        else:
            raise ValueError("Unsupported JSON feature format.")

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return torch.tensor(arr, dtype=torch.float32)

    # ---------- 4) PT / PTH ----------
    if ext in [".pt", ".pth"]:
        obj = torch.load(file_path, map_location="cpu")

        if isinstance(obj, torch.Tensor):
            t = obj.float()
            if t.ndim == 1:
                t = t.unsqueeze(0)
            return t

        if isinstance(obj, np.ndarray):
            arr = obj.astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return torch.tensor(arr, dtype=torch.float32)

        if isinstance(obj, dict):
            if "features" in obj:
                v = obj["features"]
            else:
                # Use the first tensor/array/list-like field
                v = None
                for _, vv in obj.items():
                    if isinstance(vv, (torch.Tensor, np.ndarray, list)):
                        v = vv
                        break
                if v is None:
                    raise ValueError("No valid feature data found in PT/PTH file (e.g., key='features').")

            if isinstance(v, torch.Tensor):
                t = v.float()
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                return t
            else:
                arr = np.array(v, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return torch.tensor(arr, dtype=np.float32)

        raise ValueError("Unsupported PT/PTH content format.")

    # ---------- 5) Fallback: parse original txt matrix format ----------
    with open(file_path, 'r', encoding='utf-8') as file:
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
        if matrix_data.size > 0:
            matrix_arrays.append(matrix_data)

    if len(matrix_arrays) == 0:
        raise ValueError(f"Failed to parse any matrix from file: {file_path}")

    text_features = torch.tensor(np.vstack(matrix_arrays), dtype=torch.float32)
    return text_features


def main(args):
    with open(args.PointLLM_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    model_outputs = [item["model_output"] for item in data["results"]]
    file_path = args.features_path
    point_features = read_matrices_from_file(file_path)

    prompt_class = "categories： airplane, bathtub, bed, bench, bookshelf, bottle, bowl, car, chair, cone, cup, curtain, desk, door, dresser, flower pot, glass box, guitar, keyboard, lamp, laptop, mantel, monitor, night stand, person, piano, plant, radio, range hood, sink, sofa, stairs, stool, table, tent, toilet, tv stand, vase, wardrobe, xbox."
    base_prompt = "Given a free-form description of a 3D object, the content described here belongs to one of the following 40 categories. Use this description to compute a similarity score (0-100) for each of the following 40 categories. The description of this 3D object is generated by an LLM and may be inaccurate. In addition, I will provide you with descriptions of other 3D objects that share similar features with this object of this category. 0=no relation, 100=perfect match."
    des_prompt = "3D object description:  "
    base = "\nPlease output the 40 corresponding similarity scores in the order of the above-mentioned categories, without any additional explanation."

    # Replace with your API configuration
    API_KEY = ""
    API_ENDPOINT = "https://api.atalk-ai.com/v2/chat/completions"
    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }

    nbrs = NearestNeighbors(n_neighbors=4, metric='euclidean')
    nbrs.fit(point_features)

    distances, indices = nbrs.kneighbors(point_features)

    knn_matrix = np.zeros((point_features.shape[0], point_features.shape[0]))

    rows = np.arange(point_features.shape[0]).repeat(3)
    cols = indices[:, 1:4].reshape(-1)
    vals = np.exp(-distances[:, 1:4].reshape(-1))
    knn_matrix[rows, cols] = vals

    results = []

    max_retries = 20
    TIMEOUT_LIMIT = 300

    for idx, output in enumerate(model_outputs):
        retry_count = 0
        num_retries = 0
        success = False
        object_start_time = time.time()
        last_valid_response = None

        while retry_count < max_retries and not success:
            try:
                row_index = idx
                row = knn_matrix[row_index]
                top5_cols = np.argsort(row)[-3:][::-1]

                PROMPT_LP2 = []
                first_item = True

                for item in data["results"]:
                    if item["object_id"] in top5_cols:
                        if first_item:
                            PROMPT_LP2.append(item["model_output"])
                            first_item = False
                        else:
                            PROMPT_LP2.append("\n" + item["model_output"])

                PROMPT_LP2 = "".join(PROMPT_LP2)
                PROMPT2 = "Descriptions of 3D objects with similar features: " + PROMPT_LP2
                full_query = f"{base_prompt}\n\n{prompt_class}\n\n{des_prompt}{output}\n\n{PROMPT2}\n{base}"

                payload = {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": full_query}],
                    "temperature": 0.1,
                    "stream": False
                }

                response = requests.post(
                    API_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()

                response_data = response.json()
                api_output = response_data["choices"][0]["message"]["content"].strip()

                numbers = []
                if '[' in api_output and ']' in api_output:
                    try:
                        numbers = json.loads(api_output)
                    except json.JSONDecodeError:
                        numbers = [float(num) for num in re.findall(r'\b\d{1,3}(?:\.\d+)?\b', api_output)]
                else:
                    numbers = [float(num) for num in re.findall(r'\b\d{1,3}(?:\.\d+)?\b', api_output)]

                last_valid_response = {
                    "api_output": api_output,
                    "numbers": numbers
                }

                if len(numbers) == 40:
                    results.append({
                        "object_id": idx,
                        "score": numbers,
                        "raw_response": api_output
                    })
                    success = True
                    print(f"成功处理对象 {idx}")
                    print(numbers)
                else:
                    num_retries += 1
                    if num_retries < 5:
                        print(f"对象 {idx} 返回了 {len(numbers)} 个数字 (应为40个)，将重试 ({num_retries}/5)")
                        time.sleep(1)
                        continue
                    else:
                        print(f"警告: 对象 {idx} 经过5次重试后仍返回 {len(numbers)} 个数字，将记录最后一次结果")
                        results.append({
                            "object_id": idx,
                            "score": numbers,
                            "raw_response": api_output,
                            "warning": f"返回了 {len(numbers)} 个数字 (应为40个)"
                        })
                        success = True

            except Exception as e:
                current_time = time.time()
                elapsed = current_time - object_start_time

                if elapsed > TIMEOUT_LIMIT:
                    print(f"对象 {idx} 处理超时 ({elapsed:.2f}秒)，将重试...")
                    retry_count += 1
                    object_start_time = time.time()
                    time.sleep(2)
                else:
                    print(f"处理对象 {idx} 时出错: {str(e)}")
                    time.sleep(1)

            if not success and retry_count >= max_retries and last_valid_response is not None:
                print(f"错误: 对象 {idx} 经过 {max_retries} 次重试仍失败，将记录最后一次有效响应")
                numbers = last_valid_response["numbers"]
                results.append({
                    "object_id": idx,
                    "score": numbers,
                    "raw_response": last_valid_response["api_output"],
                    "warning": f"处理失败后使用最后一次响应，返回了 {len(numbers)} 个数字"
                })
                success = True

    with open('GPT__results_cls_MN.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Processing completed. Results saved to classification_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--features_path", type=str, required=True,
                        help="Path to the features file")
    parser.add_argument("--PointLLM_results_path", type=str, required=True,
                        help="Path to PointLLM results")
    args = parser.parse_args()

    main(args)