import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import argparse
import sys


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
    if args.dataset_split == "MN1":
        valid_labels = {
            "airplane",
            "bathtub",
            "bed",
            "bench",
            "bookshelf",
            "bottle",
            "bowl",
            "car",
            "chair",
            "cone",
            "cup",
            "curtain",
            "desk"
        }
    elif args.dataset_split == "MN2":
        valid_labels = {
            "door",
            "dresser",
            "flower pot",
            "glass box",
            "guitar",
            "keyboard",
            "lamp",
            "laptop",
            "mantel",
            "monitor",
            "night stand",
            "person",
            "piano",
        }
    elif args.dataset_split == "MN3":
        valid_labels = {
            "plant",
            "radio",
            "range hood",
            "sink",
            "sofa",
            "stairs",
            "stool",
            "table",
            "tent",
            "toilet",
            "tv stand",
            "vase",
            "wardrobe",
            "xbox"
        }
    else:
        print(f"error dataset split choice")
        sys.exit(1)

    with open(args.PointLLM_results_path, 'r') as f:
        data = json.load(f)
    A = []
    for item in data['results']:
        label = item['label_name']
        if label in valid_labels:
            A.append(1)
        else:
            A.append(0)
    file_path = args.features_path
    point_features = read_matrices_from_file(file_path)

    nbrs = NearestNeighbors(n_neighbors=4, metric='euclidean')
    nbrs.fit(point_features)
    distances, indices = nbrs.kneighbors(point_features)  
    knn_matrix = np.zeros((point_features.shape[0], point_features.shape[0]))
    rows = np.arange(point_features.shape[0]).repeat(3)  
    cols = indices[:, 1:4].reshape(-1)  
    vals = np.exp(-distances[:, 1:4].reshape(-1))  

    knn_matrix[rows, cols] = vals

    np.fill_diagonal(knn_matrix, 1.0) \

    W = knn_matrix + knn_matrix.T
    W = np.array(W)
    d = []
    for row in knn_matrix:
        row_sum = sum(row)  
        d.append(row_sum) 
    d = 1 / np.sqrt(d)
    D = np.diag(d)
    D_neg_sqrt = D
    Adajency_Matrix = np.matmul(np.matmul(D_neg_sqrt, W), D_neg_sqrt)


    with open(args.LLM_results_path, 'r') as file:
        data = json.load(file)
    scores = [entry["score"] for entry in data]

    Y_array = np.array(scores)
    Y_array = Y_array.T
    Y_array_0 = Y_array
    a = 0.5
    n = 5

    for i in range(n):
        Y_array = a * Adajency_Matrix @ Y_array + (1 - a) * Y_array_0
    score = Y_array.T

    auroc = roc_auc_score(A, score)
    print(f'AUROC 值为: {auroc}')
    fpr, tpr, thresholds = roc_curve(A, score)
    closest_tpr_index = np.argmax(tpr >= 0.95)
    fpr95 = fpr[closest_tpr_index]
    print(f"FPR95: {fpr95:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_path", type=str, required=True,
                        help="Path to the features JSON file")
    parser.add_argument("--PointLLM_results_path", type=str, required=True,
                        help="Path to PointLLM results")
    parser.add_argument("--dataset_split", type=str,
                        choices=["MN1", "MN2", "MN3"],
                        help="Name of the dataset to split")
    parser.add_argument("--LLM_results_path", type=str, required=True,
                        help="Path to the features JSON file")
    args = parser.parse_args()
    main(args)