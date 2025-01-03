from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm  # 导入 tqdm 用于进度条显示
import numpy as np

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# 使用本地模型路径加载模型和分词器
local_model_path = './models/text2vec-base-chinese'  # 修改为本地模型路径

tokenizer = BertTokenizer.from_pretrained(local_model_path)
model = BertModel.from_pretrained(local_model_path)

# 读取文件中的每一行作为输入句子
file_path = './data/raw_entity/预防措施.txt'  # 此处依次替换为不同的实体文件
with open(file_path, 'r', encoding='utf-8') as f:
    sentences = f.readlines()

# 去除每行的换行符
sentences = [sentence.strip() for sentence in sentences]

# 创建一个空列表用于保存所有句子的嵌入
sentence_embeddings = []

# 使用 tqdm 显示进度条
for sentence in tqdm(sentences, desc="Processing sentences", unit="sentence"):
    # Tokenize sentences
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform pooling. In this case, mean pooling.
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings.append(embedding)

# 将嵌入转换为 PyTorch Tensor
sentence_embeddings = torch.cat(sentence_embeddings, dim=0)

# 输出嵌入
print("Sentence embeddings:")
print(sentence_embeddings)

# 保存嵌入为 NPY 文件
npy_file_path = "./data/encode_results/预防措施.npy"
np.save(npy_file_path, sentence_embeddings.numpy())  # 使用 np.save 保存为 NPY 文件

print(f"Embeddings saved to {npy_file_path}")
