import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# 检查GPU是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 加载Sentence Transformer模型（替代SimCSE）
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

# 读取第一个文件（caption_results.json）
with open('caption_results.json', 'r') as f:
    file1_data = json.load(f)
descriptions1 = [item['description'] for item in file1_data]

# 读取第二个文件（PointLLM文件）
with open('PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json', 'r') as f:
    file2_data = json.load(f)
descriptions2 = [item['ground_truth'] for item in file2_data['results']]

# 确保两个列表长度相同（取最小长度）
min_length = min(len(descriptions1), len(descriptions2))
descriptions1 = descriptions1[:min_length]
descriptions2 = descriptions2[:min_length]

# 计算相似度
similarities = []
batch_size = 32  # 批处理大小，可根据GPU内存调整

# 分批处理以提高效率
for i in range(0, min_length, batch_size):
    batch1 = descriptions1[i:i + batch_size]
    batch2 = descriptions2[i:i + batch_size]

    # 编码两个批次的文本
    embeddings1 = model.encode(batch1, convert_to_tensor=True, device=device)
    embeddings2 = model.encode(batch2, convert_to_tensor=True, device=device)

    # 计算余弦相似度（使用点积因为向量已归一化）
    batch_similarities = torch.sum(embeddings1 * embeddings2, dim=1).tolist()
    similarities.extend(batch_similarities)

# 计算平均相似度
average_similarity = np.mean(similarities)

print(f"Processed {min_length} pairs of descriptions")
print(f"Average semantic similarity: {average_similarity:.4f}")