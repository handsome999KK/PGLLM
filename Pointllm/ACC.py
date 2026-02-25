import json
import random
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse

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


def main(args):
    with open(args.LLM_results_path, 'r') as f:
        data = json.load(f)
    score_matrix = []
    for item in data:
        score_list = item['score']
        score_matrix.append(score_list)
    score_matrix = np.array(score_matrix)
    print(score_matrix.shape)
    pre = []
    labels_corres = {
        "airplane": 0,
        "bathtub": 1,
        "bed": 2,
        "bench": 3,
        "bookshelf" :4,
        "bottle": 5,
        "bowl": 6,
        "car": 7,
        "chair": 8,
        "cone": 9,
        "cup": 10,
        "curtain": 11,
        "desk": 12,
        "door": 13,
        "dresser": 14,
        "flower pot": 15,
        "glass box": 16,
        "guitar": 17,
        "keyboard": 18,
        "lamp": 19,
        "laptop": 20,
        "mantel": 21,
        "monitor": 22,
        "night stand": 23,
        "person": 24,
        "piano": 25,
        "plant": 26,
        "radio": 27,
        "range hood": 28,
        "sink": 29,
        "sofa": 30,
        "stairs": 31,
        "stool": 32,
        "table": 33,
        "tent": 34,
        "toilet": 35,
        "tv stand":36,
        "vase":37,
        "wardrobe": 38,
        "xbox": 39,
    }

    with open(args.PointLLM_results_path, 'r') as f:
        data = json.load(f)
    true_labels = []
    for item in data['results']:
        label_name = item['label_name']
        if label_name in labels_corres:
            true_labels.append(labels_corres[label_name])
        else:
            true_labels.append(-1) 
    file_path = args.features_path
    point_features = read_matrices_from_file(file_path)
    nbrs = NearestNeighbors(n_neighbors=11, metric='euclidean')
    nbrs.fit(point_features)
    distances, indices = nbrs.kneighbors(point_features)  
    knn_matrix = np.zeros((2468, 2468))
    rows = np.arange(2468).repeat(10)  
    cols = indices[:, 1:11].reshape(-1)  
    vals = np.exp(-distances[:, 1:11].reshape(-1)) 
    knn_matrix[rows, cols] = vals  
    np.fill_diagonal(knn_matrix, 1.0) \
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    W = knn_matrix + knn_matrix.T
    W = np.array(W)
    d = []

    for row in knn_matrix:
        row_sum = sum(row)  
        d.append(row_sum)  
    d = 1 / np.sqrt(d)
    D = np.diag(d)
    D_neg_sqrt = D
    W_tensor = torch.tensor(W, dtype=torch.float32, device=device)
    D_neg_sqrt_tensor = torch.tensor(D, dtype=torch.float32, device=device)
    Adajency_Matrix_tensor = D_neg_sqrt_tensor @ W_tensor @ D_neg_sqrt_tensor
    Y_array = torch.tensor(score_matrix, dtype=torch.float32, device=device)
    Y_array_0 = Y_array.clone()
    a = 0.5
    n = 5
    for _ in range(n):
        Y_array = a * (Adajency_Matrix_tensor @ Y_array) + (1 - a) * Y_array_0
    Y_array = Y_array.cpu().numpy()
    for scores in Y_array:
        max_score = max(scores) 
        max_indices = [i for i, score in enumerate(scores) if score == max_score]
        random_index = random.choice(max_indices)
        pre.append(random_index)  
    correct_count = 0
    for i in range(len(true_labels)):
        if pre[i] == true_labels[i]:
            correct_count += 1
    acc = correct_count / len(true_labels)
    print(f"准确率: {acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=str, required=True,
                        help="Path to the features JSON file")
    parser.add_argument("--LLM_results_path", type=str, required=True,
                        help="Path to PointLLM results")
    parser.add_argument("--PointLLM_results_path", type=str, required=True,
                        help="Path to PointLLM results")
    args = parser.parse_args()
    main(args)