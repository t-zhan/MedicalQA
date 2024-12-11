import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


class EntityMatcher:
    def __init__(self, model_path, disease_file, symptom_file, disease_embedding_file, symptom_embedding_file):
        """
        初始化EntityMatcher类，加载模型和数据
        :param model_path: 本地BERT模型路径
        :param disease_file: 疾病名称文本文件路径
        :param symptom_file: 症状名称文本文件路径
        :param disease_embedding_file: 疾病嵌入文件路径
        :param symptom_embedding_file: 症状嵌入文件路径
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        
        # 加载原文数据
        self.disease_texts = self.load_texts(disease_file)
        self.symptom_texts = self.load_texts(symptom_file)
        
        # 加载现有的疾病和症状嵌入
        self.disease_embeddings = np.load(disease_embedding_file)
        self.symptom_embeddings = np.load(symptom_embedding_file)

    @staticmethod
    def load_texts(file_path):
        """加载文本数据"""
        with open(file_path, 'r', encoding='utf-8') as file:
            texts = file.readlines()
        return [text.strip() for text in texts]  # 去掉换行符

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        """Mean Pooling - Take attention mask into account for correct averaging"""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def match_entity_to_embedding(self, entity: str, entity_type: str = 'disease', top_k_dict: dict = None):
        """
        根据输入的实体进行嵌入，然后在疾病或症状嵌入中找到最相似的原文。
        :param entity: 输入实体（疾病或症状名称）
        :param entity_type: 实体类型，'disease' 或 'symptom'
        :param top_k_dict: 一个字典，定义不同实体类型的 top_k
        :return: 匹配的最相似原文和相似度分数
        """
        # 设置默认的 top_k 配置
        if top_k_dict is None:
            top_k_dict = {'disease': 1, 'symptom': 10}  # 默认疾病匹配1个，症状匹配10个
        
        # Step 1: 对输入实体进行嵌入
        encoded_input = self.tokenizer(entity, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        entity_embedding = self.mean_pooling(model_output, encoded_input['attention_mask']).numpy()

        # Step 2: 根据实体类型选择对应的嵌入和原文数据
        if entity_type == 'disease':
            embeddings = self.disease_embeddings
            texts = self.disease_texts
            top_k = top_k_dict.get('disease', 1)  # 默认返回 1 个最相似的结果
        elif entity_type == 'symptom':
            embeddings = self.symptom_embeddings
            texts = self.symptom_texts
            top_k = top_k_dict.get('symptom', 10)  # 默认返回 10 个最相似的结果
        else:
            raise ValueError("entity_type should be either 'disease' or 'symptom'")

        # 确保 texts 列表不为空
        if not texts:
            raise ValueError(f"原文数据为空，请确保已经加载了{entity_type}数据")

        # Step 3: 使用余弦相似度计算嵌入之间的相似度
        similarity_matrix = cosine_similarity(entity_embedding, embeddings)  # 计算相似度矩阵

        # Step 4: 获取最相似的top_k个结果
        top_k_indices = similarity_matrix.argsort()[0][-top_k:][::-1]  # 获取 top_k 个最相似的索引
        top_similarities = similarity_matrix[0][top_k_indices]  # 对应的相似度得分
        top_texts = [texts[i] for i in top_k_indices]  # 获取最相似的文本

        return top_texts, top_similarities
