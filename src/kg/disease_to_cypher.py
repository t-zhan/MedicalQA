import py2neo
from typing import List, Dict, Any, Tuple
import json
import os

class DiseaseToCypher:
    # 预定义所有可能的关系类型
    RELATION_TYPES = {
        "推荐药品": "疾病——推荐药品",
        "推荐食物": "疾病——推荐食物",
        "忌讳食物": "疾病——忌讳食物",
        "检查项目": "疾病——检查项目",
        "科室": "疾病——科室",
        "症状": "疾病——疾病症状",
        "治疗方法": "疾病——治疗方法",
        "传染方式": "疾病——传染方式",
        "并发症": "疾病——并发症",
        "发病部位": "疾病——发病部位",
        "医保": "疾病——医保状态",
        "生病原因": "疾病——生病原因",
        "预防措施": "疾病——预防措施",
        "易感人群": "疾病——易感人群"
    }

    def __init__(self,
                 db_name='neo4j',
                 neo4j_host="bolt://10.176.28.10:7687",
                 neo4j_user="neo4j",
                 neo4j_password="medical123456"):
        # 初始化neo4j连接
        self.graph = py2neo.Graph(
            neo4j_host,
            auth=(neo4j_user, neo4j_password),
            name=db_name
        )

    def get_disease_info(self, disease_names: List[str]) -> List[Dict[str, Any]]:
        """
        获取疾病列表的所有属性和关系信息，并按疾病整合到字典中
        
        Args:
            disease_names: 疾病名称列表
            
        Returns:
            List[Dict]: 每个疾病的所有信息字典的列表
            字典格式：{
                "属性": {"疾病名称": xx, "疾病简介": xx, ...},
                "疾病——推荐药品": [(start_type, start_name, relation_type, end_type, end_name), ...],
                "疾病——疾病症状": [...],
                ...
            }
        """
        result = []
        
        # Cypher查询语句：查找以疾病为起点的所有关系
        relation_query = """
        MATCH (start:疾病 {name: $disease_name})-[r]->(end)
        RETURN labels(start)[0] as start_type, 
               start.name as start_name,
               type(r) as relation_type,
               labels(end)[0] as end_type,
               end.name as end_name
        """
        
        # Cypher查询语句：查找疾病节点的所有属性
        property_query = """
        MATCH (d:疾病 {name: $disease_name})
        RETURN d.name as name,
               d.intro as intro,
               d.period as period,
               d.rate as rate,
               d.cost as cost,
               d.notice as notice,
               d.ratio as ratio,
               d.accompany_disease as accompany_disease,
               d.alias as alias
        """
        
        # 对每个疾病执行查询
        for disease_name in disease_names:
            disease_info = {}
            
            # 初始化所有关系类型为空列表
            for relation_type in self.RELATION_TYPES.values():
                disease_info[relation_type] = []

            # 查询属性
            property_records = self.graph.run(property_query, disease_name=disease_name).data()
            print(disease_name)
            if not property_records:  # 如果查询结果为空列表，跳过该疾病
                print('*'*100)
                print(f"注意: 疾病实体 '{disease_name}' 在数据库中未找到，已跳过")
                print('*'*100)
                continue

            property_record = property_records[0]
            disease_info["属性"] = {
                "疾病名称": property_record["name"],
                "疾病简介": property_record["intro"] or "",
                "治疗周期": property_record["period"] or "",
                "治愈率": property_record["rate"] or "",
                "治疗费用": property_record["cost"] or "",
                "注意事项": property_record["notice"] or "",
                "患病比例": property_record["ratio"] or "",
                "并发症": property_record["accompany_disease"] or "",
                "别名": property_record["alias"] or ""
            }

            # 查询关系
            relation_records = self.graph.run(relation_query, disease_name=disease_name).data()
            for record in relation_records:
                relation = (
                    record['start_type'],    # 起始节点类型
                    record['start_name'],    # 起始节点名称
                    record['relation_type'], # 关系类型
                    record['end_type'],     # 终止节点类型
                    record['end_name']      # 终止节点名称
                )
                # 将关系添加到对应类型的列表中
                relation_category = self.RELATION_TYPES.get(record['relation_type'])
                if relation_category:
                    disease_info[relation_category].append(relation)
            
        
            result.append(disease_info)
                
        return result

    def get_similar_symptoms(self, symptom_groups: List[List[str]]) -> List[List[str]]:
        """
        对给定的症状组进行模糊匹配，每组包含同一症状的多种表述
        
        Args:
            symptom_groups: 二维列表，每个子列表包含同一症状的不同表述
            例如: [
                ["心悸", "心慌", "心跳紊乱"],
                ["头疼", "头疼剧烈", "头痛"],
                ["发热", "发烧", "高烧"]
            ]
                
        Returns:
            List[List[str]]: 返回每组症状的模糊匹配结果
            例如: [
                ["心悸", "心慌", "心悸伴胸闷", ...], # 第一组所有表述匹配到的症状的并集
                ["头疼", "头痛", "剧烈头痛", ...],   # 第二组所有表述匹配到的症状的并集
                ["发热", "发烧", "高热", ...]        # 第三组所有表述匹配到的症状的并集
            ]
        """
        result = []
        
        # Cypher查询语句：使用CONTAINS进行模糊匹配
        query = """
        MATCH (s:疾病症状)
        WHERE s.name CONTAINS $symptom_name
        RETURN DISTINCT s.name as symptom_name
        """
        
        # 对每组症状的每个表述进行模糊匹配
        for symptom_group in symptom_groups:
            group_similar_symptoms = set()  # 使用集合来自动去重
            
            # 对组内每个症状表述进行模糊匹配
            for symptom_name in symptom_group:
                # 执行查询
                records = self.graph.run(query, symptom_name=symptom_name).data()
                
                # 提取匹配到的症状名称并添加到该组的集合中
                matched_symptoms = [record['symptom_name'] for record in records]
                
                # 如果没有找到相似症状，则将原始症状加入集合
                if not matched_symptoms:
                    group_similar_symptoms.add(symptom_name)
                else:
                    group_similar_symptoms.update(matched_symptoms)
            
            # 将集合转换为列表并添加到结果中
            result.append(list(group_similar_symptoms))
        
        return result

    def get_diseases_by_fuzzy_symptoms(self, symptom_groups: List[List[str]], debug: bool = False) -> List[str]:
        """
        通过模糊匹配症状组查找疾病
        
        Args:
            symptom_groups: 二维列表，每个子列表包含同一症状的不同表述
            例如: [
                ["心悸", "心慌", "心跳紊乱"],  # 第一组症状
                ["头疼", "头痛", "头晕"],      # 第二组症状
            ]
            debug: 是否打印调试信息，默认为False
            
        Returns:
            List[str]: 最终匹配到的疾病名称列表
            匹配规则：
            1. 对于同一组症状（如第一组中的"心悸"、"心慌"、"心跳紊乱"），疾病只需具有其中任意一个症状即满足
            2. 对于不同组的症状之间取交集
        """
        # 1. 获取每组症状的模糊匹配结果
        similar_symptoms_groups = self.get_similar_symptoms(symptom_groups)
        
        if debug:
            print("\n模糊匹配后的症状组:")
            for i, group in enumerate(similar_symptoms_groups):
                print(f"第{i+1}组症状: {group}\n")

        
        # 2. 对每组相似症状分别查询疾病
        diseases_per_group = []
        
        for i, symptom_group in enumerate(similar_symptoms_groups):
            # Cypher查询语句：查找与症状组中任一症状相关的疾病
            query = """
            MATCH (d:疾病)-[r:症状]->(s:疾病症状)
            WHERE s.name IN $symptom_group
            WITH DISTINCT d.name as disease_name
            RETURN disease_name
            """
            
            # 执行查询
            records = self.graph.run(query, symptom_group=symptom_group).data()
            
            # 提取疾病名称
            group_diseases = set(record['disease_name'] for record in records)
            
            if group_diseases:  # 只添加非空结果
                diseases_per_group.append(group_diseases)
        
        # 3. 对不同症状组的疾病取交集
        if not diseases_per_group:
            return []
        
        # 从第一组开始，逐步与后面的组取交集
        result_diseases = diseases_per_group[0]
        if debug:
            print("\n各组症状匹配到的疾病:")
            print(f"第1组疾病数量: {len(diseases_per_group[0])}")
            print(list(diseases_per_group[0]))
            print("-" * 100)
            
        for i, diseases in enumerate(diseases_per_group[1:], 1):
            result_diseases = result_diseases.intersection(diseases)
            if debug:
                print(f"第{i+1}组疾病数量: {len(diseases)}")
                print(list(diseases))
                print(f"\n前{i+1}个症状组查到的疾病取交集后的结果:")
                print(f"剩余疾病数量: {len(result_diseases)}")
                print(list(result_diseases))
                print("-" * 100)
        
        return list(result_diseases)