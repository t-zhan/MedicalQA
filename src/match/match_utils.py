from src.match.index_type import EntityMatcher  # 假设你的EntityMatcher类已经定义在entity_matcher.py中
import json
from typing import List, Dict, Any, Union


def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def match_entities_and_save(entry: Dict[str, Any])-> Union[List[str], Dict[str, Any]]:
    """读取 JSON 文件，匹配实体并保存结果到文件"""

    # 创建EntityMatcher实例
    model_path = 'models/text2vec-base-chinese'
    disease_file = 'data/match_info/disease.txt'
    symptom_file = 'data/match_info/symptom.txt'
    disease_embedding_file = 'data/match_info/disease.npy'
    symptom_embedding_file = 'data/match_info/symptom.npy'

    # 实例化 EntityMatcher
    entity_matcher = EntityMatcher(model_path, disease_file, symptom_file, disease_embedding_file, symptom_embedding_file)

    entity_type = entry['type']
    
    # 中文类型映射到英文类型
    if entity_type == '疾病':
        entity_type = 'disease'
    elif entity_type == '症状':
        entity_type = 'symptom'
    else:
        raise ValueError(f"Invalid entity type: {entity_type}, must be either '疾病' or '症状'")
    
    match_lists = []  # 用于存储当前 entry 中每个实体的匹配结果

    for entity in entry['entity']:
        if entity_type == 'disease':
            # 匹配疾病，返回 1 个最相似的结果，过滤相似度低于阈值的匹配
            matching_disease_texts, disease_similarities = entity_matcher.match_entity_to_embedding(entity, 'disease', {'疾病': 1, '症状': 5})
            match_list = [match[0] for match in zip(matching_disease_texts, disease_similarities) if match[1] > 0.8]
        elif entity_type == 'symptom':
            # 匹配症状，返回 5 个最相似的结果，过滤相似度低于阈值的匹配
            matching_symptom_texts, symptom_similarities = entity_matcher.match_entity_to_embedding(entity, 'symptom', {'疾病': 1, '症状': 5})
            match_list = [match[0] for match in zip(matching_symptom_texts, symptom_similarities) if match[1] > 0.8]

        # 如果匹配结果不为空，添加到当前实体的匹配结果列表
        if match_list:
            match_lists.append(match_list)

    return {"Type": entity_type, "match": match_lists}