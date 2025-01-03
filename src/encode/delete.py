# 删除高度相似的节点
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embeddings_np = np.load('./data/encode_results/预防措施.npy')

# 计算余弦相似度矩阵
similarity_matrix = cosine_similarity(embeddings_np)

# 阈值：高相似度句子的相似度
threshold = 0.9
similar_pairs = []

# 遍历相似度矩阵，找到高相似度的句子对
for i in range(similarity_matrix.shape[0]):
    for j in range(i + 1, similarity_matrix.shape[1]):  # 只检查上三角矩阵
        if similarity_matrix[i, j] > threshold:
            similar_pairs.append((i, j, similarity_matrix[i, j]))

# 读取原始句子
with open('./data/raw_entity/预防措施.txt', 'r', encoding='utf-8') as file:
    sentences = file.readlines()

# 找到需要删除的句子索引
to_delete = set()

# 通过遍历相似度高的句子对，标记需要删除的句子
for i, j, sim in similar_pairs:
    to_delete.add(j)  # 将重复的句子（在对中另一个句子）标记为删除

# 将不需要删除的句子提取出来
remaining_sentences = [sentences[i] for i in range(len(sentences)) if i not in to_delete]

# 打印一下删除后的句子数
print(f"删除重复句子后的句子数: {len(remaining_sentences)}")

# 如果你有嵌入向量，可以通过相同的索引删除对应的嵌入
# 例如，假设 embeddings_np 是你原始的嵌入矩阵
remaining_embeddings = np.delete(embeddings_np, list(to_delete), axis=0)

# # 输出剩余句子的嵌入向量到 .npy 文件
# npy_output_file = '/root/Knowledge/encode_result/index_encode/疾病症状.npy'
# np.save(npy_output_file, remaining_embeddings)

# 输出剩余的句子到新的文本文件
output_file = './data/entity_modify/预防措施.txt'
with open(output_file, 'w', encoding='utf-8') as outfile:
    outfile.writelines(remaining_sentences)

print(f"删除重复句子后的文件已输出到：{output_file}")
# print(f"对应的嵌入已输出到：{npy_output_file}")