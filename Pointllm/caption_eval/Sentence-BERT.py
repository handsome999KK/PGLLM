from sentence_transformers import SentenceTransformer, util
import json

# 加载Sentence-BERT模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 读取第一个文件（caption_results.json）
with open('caption_results.json', 'r') as f:
    file1_data = json.load(f)
    # 提取所有描述
    descriptions1 = [item["description"] for item in file1_data]

# 读取第二个文件（PointLLM...json）
with open('PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json', 'r') as f:
    file2_data = json.load(f)
    # 提取所有ground_truth
    descriptions2 = [item["ground_truth"] for item in file2_data["results"]]

# 确保两个列表长度相同
assert len(descriptions1) == len(descriptions2), "文件包含的描述数量不匹配"

# 计算相似度
similarities = []
for desc1, desc2 in zip(descriptions1, descriptions2):
    # 编码句子
    embeddings = model.encode([desc1, desc2], convert_to_tensor=True)
    # 计算余弦相似度
    cos_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    similarities.append(cos_sim)

# 计算平均值
average_similarity = sum(similarities) / len(similarities)

print(f"句子对数量: {len(similarities)}")
print(f"平均Sentence-BERT相似度: {average_similarity:.4f}")