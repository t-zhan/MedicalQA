import os
import copy
import re
import py2neo
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from extract_info import *


# 配置参数
MAX_LINES = None  # prevent, cause, easy_get提取行数 # 提取全部设置为None
MAX_PAPERS = None   # 提取几篇论文 # 提取全部设置为None

class MedicalKGBuilder:
    _ENTITY_TEMPLATE = {
        "疾病": [], "药品": [], "食物": [],
        "检查项目": [], "科室": [], "疾病症状": [],
        "治疗方法": [], "传染方式": [], "并发症": [],
        "生病原因": [], "预防措施": [], "易感人群": [],
        "医保状态": ['是'], "发病部位": []
    }

    _MEDICAL_ENTITY_MAPPING = {
        # 预定义的entity_type: 数据中对应的index
        "检查项目": "检查项目",
        "科室": "科室",
        "疾病症状": "疾病症状",
        "治疗方法": "治疗方法",
        "传染方式": "传染方式",
        "发病部位": "发病部位"
    }
    _OTHER_ENTITY_MAPPING = {
        "检查项目": "检查项目",
        "科室": "科室",
        "疾病症状": "疾病症状",
        "治疗方法": "治疗方法",
        "传染方式": "传染方式",
        "发病部位": "发病部位",
        "易感人群": "易感人群"
    }

    _MEDICAL_RELATION_MAPPING = {
        # 预定义的entity_type: 知识图谱中预定义的relation_type
        "检查项目": "检查项目",
        "科室": "科室",
        "疾病症状": "症状",
        "治疗方法": "治疗方法",
        "传染方式": "传染方式",
        "发病部位": "发病部位"
    }
    _OTHER_RELATION_MAPPING = {
        "检查项目": "检查项目",
        "科室": "科室",
        "疾病症状": "症状",
        "治疗方法": "治疗方法",
        "传染方式": "传染方式",
        "发病部位": "发病部位",
        "易感人群": "易感人群"
    }

    _PAPER_ENTITY_MAPPING = {
        'disease_has_drug': '药品',
        'disease_has_treatment': '治疗方法', 
        'disease_no_eat': '食物',
        'disease_do_check': '检查项目',
        'disease_has_symptom': '疾病症状',
        'disease_easy_get_population': '易感人群',
        'disease_has_prevent': '预防措施',
        'disease_acompany_disease': '并发症',
        'disease_has_cause': '生病原因',
        'disease_has_spreads_way': '传染方式'
    }

    def __init__(self, 
                 db_name='neo4j',
                 neo4j_host="bolt://10.176.28.10:7687",
                 neo4j_user="neo4j",
                 neo4j_password="medical123456",
                 disease_data_path='data/kg_info/other_medical_merge.csv',
                 cause_path='data/kg_info/result_cause.json',
                 easy_get_path='data/kg_info/result_easy_get.json',
                 prevent_path='data/kg_info/result_prevent.json',
                 output_dir='data/kg_info/entity_relation'):
        # 初始化实体和关系
        self.all_entity = copy.deepcopy(self._ENTITY_TEMPLATE)
        self.relationship = []
        self.accompany_rel = []
        
        # 保存路径配置
        self.disease_data_path = disease_data_path
        self.cause_path = cause_path
        self.easy_get_path = easy_get_path
        self.prevent_path = prevent_path
        self.output_dir = output_dir
        
        # 初始化neo4j连接
        self.graph = py2neo.Graph(
            neo4j_host,
            auth=(neo4j_user, neo4j_password),
            name=db_name
            )
        
        # 加载预处理的cause、easy_get、prevent数据
        self._load_preprocessed_data()
        
        # 初始化disease_set
        df_disease_other_medical = pd.read_csv(self.disease_data_path)
        self.disease_set = set(df_disease_other_medical['疾病名称'])

    def _load_preprocessed_data(self):
        """加载预处理的数据"""
        # 提取关系
        cause_relationship = extract_rel(self.cause_path, "has_cause", max_lines=MAX_LINES)
        easy_get_relationship = extract_rel(self.easy_get_path, "easy_get", max_lines=MAX_LINES)
        prevent_relationship = extract_rel(self.prevent_path, "has_prevent", max_lines=MAX_LINES)
        
        # 提取实体
        self.all_entity["生病原因"] = extract_entity(cause_relationship)
        self.all_entity["易感人群"] = extract_entity(easy_get_relationship)
        self.all_entity["预防措施"] = extract_entity(prevent_relationship)
        
        # 添加关系
        self.relationship.extend([("疾病", disease, "生病原因", "生病原因", cause) 
                                for disease, _, cause in cause_relationship])
        self.relationship.extend([("疾病", disease, "预防措施", "预防措施", prevent) 
                                for disease, _, prevent in prevent_relationship])
        self.relationship.extend([("疾病", disease, "易感人群", "易感人群", easy_get) 
                                for disease, _, easy_get in easy_get_relationship])

    def process_medical_data(self, data, is_other=False):
        entity_dict = self.all_entity
        disease_name = data.get("疾病名称", "")
        
        # 添加疾病实体
        disease_entity = self._create_disease_entity(data, is_other)
        entity_dict["疾病"].append(disease_entity)
        
        # 处理药品和食物
        self._process_medicine_food(data, disease_name, entity_dict, is_other)
        
        # 处理医保状态
        self._process_medical_insurance(data, disease_name)

        # 处理并发症
        self._process_accompany_disease(data, disease_name, entity_dict)
        
        # 处理其他实体和关系
        entity_mapping = self._OTHER_ENTITY_MAPPING if is_other else self._MEDICAL_ENTITY_MAPPING
        relation_mapping = self._OTHER_RELATION_MAPPING if is_other else self._MEDICAL_RELATION_MAPPING
        self._process_entities_relations(data, disease_name, entity_dict, entity_mapping, relation_mapping)


    def _process_accompany_disease(self, data, disease_name, entity_dict):
        """处理并发症实体和关系"""
        if "并发症" in data and data["并发症"]:
            accompany_diseases = data["并发症"] if isinstance(data["并发症"], list) else [data["并发症"]]
            
            # 处理并发症实体
            for disease in accompany_diseases:
                # 只有当并发症不在disease_set中时才添加为实体

                # todo!!!!!
                # 剩下要考虑的就是在建Node的时候，是否将disease_set以外的疾病也创建Node

                if disease not in self.disease_set:
                    entity_dict["并发症"].append(disease)
                
                # 无论是否在disease_set中都添加关系
                self.accompany_rel.append(
                    ("疾病", disease_name, "并发症", "疾病", disease)
                )

    def _process_medical_insurance(self, data, disease_name):
        """处理医保状态关系"""
        insurance_status = data.get("是否医保", "")
        if insurance_status == "是":
            self.relationship.append(
                ("疾病", disease_name, "医保", "医保状态", "是")
            )

    def _process_medicine_food(self, data, disease_name, entity_dict, is_other):
        """单独处理药品和食物的实体与关系"""
        # 处理药品
        medicine_keys = ["常用药品", "推荐药品"] if not is_other else ["常用药品"]
        medicines = []
        for key in medicine_keys:
            if key in data and data[key]:
                value = data[key]
                if isinstance(value, list):
                    medicines.extend(value)
                else:
                    medicines.append(value)
        if medicines:
            entity_dict["药品"].extend(medicines)
            self.relationship.extend([
                ("疾病", disease_name, "推荐药品", "药品", medicine)
                for medicine in medicines
            ])

        # 处理食物
        if not is_other:  # 只在medical数据中处理食物
            # 推荐食物
            recommend_food = data.get("推荐食物", [])
            if recommend_food:
                if isinstance(recommend_food, list):
                    entity_dict["食物"].extend(recommend_food)
                    self.relationship.extend([
                        ("疾病", disease_name, "推荐食物", "食物", food)
                        for food in recommend_food
                    ])
                else:
                    entity_dict["食物"].append(recommend_food)
                    self.relationship.append(
                        ("疾病", disease_name, "推荐食物", "食物", recommend_food)
                    )

            # 忌讳食物
            avoid_food = data.get("忌讳食物", [])
            if avoid_food:
                if isinstance(avoid_food, list):
                    entity_dict["食物"].extend(avoid_food)
                    self.relationship.extend([
                        ("疾病", disease_name, "忌讳食物", "食物", food)
                        for food in avoid_food
                    ])
                else:
                    entity_dict["食物"].append(avoid_food)
                    self.relationship.append(
                        ("疾病", disease_name, "忌讳食物", "食物", avoid_food)
                    )

    def _create_disease_entity(self, data, is_other):
        """创建疾病实体"""
        if is_other:
            return {
                "疾病名称": data.get("疾病名称", ""),
                "疾病简介": data.get("疾病简介", ""),
                "治疗周期": data.get("治疗周期", ""),
                "治愈率": data.get("治愈率", ""),
                "治疗费用": data.get("治疗费用", ""),
                "别名": data.get("别名", ""),
                "并发症": data.get("并发症", "")
            }
        else:
            return {
                "疾病名称": data.get("疾病名称", ""),
                "疾病简介": data.get("疾病简介", ""),
                "患病比例": data.get("患病比例", ""),
                "治疗周期": data.get("治疗周期", ""),
                "治愈率": data.get("治愈率", ""),
                "治疗费用": data.get("治疗费用", ""),
                "注意事项": data.get("注意事项", ""),
                "别名": data.get("别名", ""),
                "并发症": data.get("并发症", "")
            }

    def _process_entities_relations(self, data, disease_name, entity_dict, entity_mapping, relation_mapping):
        """处理实体和其他关系"""
        for entity_type, source_key in entity_mapping.items():       
            if source_key in data:
                value = data[source_key]
                if value:  # 检查非空
                    entities = value if isinstance(value, list) else [value]
                    
                    # 添加实体
                    entity_dict[entity_type].extend(entities)
                    
                    # 添加关系
                    relation_name = relation_mapping[entity_type]
                    self.relationship.extend([
                        ("疾病", disease_name, relation_name, entity_type, entity)
                        for entity in entities
                    ])

    def deduplicate_entities_relations(self):
        # 对非疾病实体去重
        for entity_type in self.all_entity:
            if entity_type != '疾病':
                self.all_entity[entity_type] = list(set(self.all_entity[entity_type]))

        # 对关系去重
        self.relationship = list(set(tuple(rel) for rel in self.relationship))
        self.accompany_rel = list(set(tuple(rel) for rel in self.accompany_rel))

    def process_paper_data(self, instruction_path):
        """处理论文数据"""
        # 提取论文关系
        rel_list = ['disease_has_drug', 'disease_has_treatment',
                'disease_no_eat','disease_do_check',
                'disease_has_symptom', 'disease_easy_get_population', 
                'disease_has_prevent', 'disease_acompany_disease', 
                'disease_has_cause', 'disease_has_spreads_way']
        rel_dict = {rel:[] for rel in rel_list}

        # 提取论文关系
        paper_relationship, instruction_rel_dict = extract_paper_rel(instruction_path, rel_dict, self.disease_set, max_papers=MAX_PAPERS)
        
        # 提取论文疾病并添加到all_entity中
        paper_disease = set()
        for key in instruction_rel_dict:
            paper_disease.update(extract_disease(instruction_rel_dict[key]))
        
        # 将论文中的每个疾病添加到all_entity中
        for disease_name in paper_disease:
            disease_dict = {"疾病名称": disease_name}
            self.all_entity["疾病"].append(disease_dict)
        
        # 添加paper中非疾病的新实体
        for rel_type, entities in instruction_rel_dict.items():
            entity_type = self._PAPER_ENTITY_MAPPING[rel_type]
            paper_entities = extract_paper_entity(entities)
            for entity in paper_entities:
                if entity not in self.all_entity[entity_type]:
                    self.all_entity[entity_type].append(entity)

        # 添加paper中的关系
        self.relationship.extend(paper_relationship)


    def clean_entities(self):
        invalid_ways = ['无传染性', '不传染']
        for way in invalid_ways:
            if way in self.all_entity["传染方式"]:
                self.all_entity["传染方式"].remove(way)

    def save_to_files(self, output_dir=None):
        """保存实体和关系到文件并导入neo4j"""
        if output_dir is None:
            output_dir = self.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        entity_dir = os.path.join(output_dir, 'entity')
        os.makedirs(entity_dir, exist_ok=True)
        
        # 保存关系
        self._save_relationships(output_dir)
        
        # 保存实体
        self._save_entities(entity_dir)
               
    def _save_relationships(self, output_dir):
        """保存关系到文件"""
        rel_path = os.path.join(output_dir, 'last_rel.txt')
        with open(rel_path, 'w', encoding='utf-8-sig') as f:
            for rel in self.accompany_rel:
                f.write(" ".join(rel) + '\n')
            for rel in self.relationship:
                f.write(" ".join(rel) + '\n')
                
    def _save_entities(self, entity_dir):
        # 保存非疾病实体
        for k, v in self.all_entity.items():
            if k != '疾病':
                entity_path = os.path.join(entity_dir, f'{k}.txt')
                with open(entity_path, 'w', encoding='utf-8-sig') as f:
                    f.write('\n'.join(v))
                    
        # 保存疾病实体
        disease_path = os.path.join(entity_dir, '疾病.txt')
        with open(disease_path, 'w', encoding='utf-8-sig') as f:
            # 写入all_entity中的疾病名称(medical + other + paper)
            disease_names = [disease['疾病名称'] for disease in self.all_entity['疾病']]
            f.write('\n'.join(disease_names))

    def import_to_neo4j(self):
        """将实体和关系导入neo4j"""
        print("正在清空数据库")
        # 先删除所有索引
        indexes = self.graph.run("SHOW INDEXES").data()
        # 删除所有索引
        for index in indexes:
            if index.get('state') == 'ONLINE':  # 确保索引处于可删除状态
                index_name = index.get('name')
                print(f"删除索引: {index_name}")
                self.graph.run(f"DROP INDEX {index_name}")

        # 清空数据库
        self.graph.run("MATCH (n) DETACH DELETE n")

        print("正在创建索引")
        # 创建索引
        for entity_type in ["疾病", "药品", "食物", "检查项目", "科室", "疾病症状", 
                           "治疗方法", "传染方式", "并发症", "生病原因", "预防措施", 
                           "易感人群", "医保状态", "发病部位"]:
            self.graph.run(f"""
                CREATE INDEX {entity_type}_index IF NOT EXISTS
                FOR (n:{entity_type})
                ON (n.name)
            """)

        # 创建实体节点
        print("正在创建实体节点")
        for entity_type, entities in tqdm(self.all_entity.items(), desc="处理实体类型"):
            if entity_type == "疾病":
                for disease in tqdm(entities, desc="创建疾病节点"):
                    # 处理别名，将列表转换为字符串
                    alias = disease.get("别名", "")
                    if isinstance(alias, list):
                        alias = '、'.join(alias)  # 使用中文顿号连接别名
                        
                    # 使用py2neo创建节点和属性
                    node = py2neo.Node(entity_type,
                                     name=disease["疾病名称"],
                                     intro=disease.get("疾病简介", ""),
                                     period=disease.get("治疗周期", ""),
                                     rate=disease.get("治愈率", ""),
                                     cost=disease.get("治疗费用", ""),
                                     notice=disease.get("注意事项", ""),
                                     ratio=disease.get("患病比例", ""),
                                     accompany_disease=disease.get("并发症", ""),
                                     alias=alias)
                    self.graph.create(node)
            else:
                # 如果是并发症节点 直接跳过（我们只考虑disease_set之间的并发症）
                if entity_type != "并发症":  # 跳过并发症节点
                    for entity in tqdm(entities, desc=f"创建{entity_type}节点"):
                        node = py2neo.Node(entity_type, name=entity)
                        self.graph.create(node)

        # 创建关系
        print("正在导入所有关系，时间很长，请耐心等候")
        all_relations = self.relationship + self.accompany_rel
        for rel in tqdm(all_relations, desc="创建关系"):
            start_type, start_name, rel_type, end_type, end_name = rel
            
            # 使用py2neo查找起始和终止节点
            start_node = self.graph.nodes.match(start_type, name=start_name).first()
            end_node = self.graph.nodes.match(end_type, name=end_name).first()
            
            if start_node and end_node:
                # 使用py2neo创建关系
                rel = py2neo.Relationship(start_node, rel_type, end_node)
                self.graph.create(rel)


def load_json_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line)) 
    return data

def main():
    # 警告用户程序将清空neo4j数据库内容
    print("警告: 该程序将清空neo4j数据库中的所有内容!")
    user_input = input("是否继续执行? (yes/no): ").lower()
    
    if user_input != 'yes':
        print("程序已终止")
        return
        
    # 读取数据
    other_medical_merge_data = load_json_data('data/kg_info/other_medical_merge.json')
    
    # 创建知识图谱构建器
    kg_builder = MedicalKGBuilder()
    
    # 处理medical数据
    for data in other_medical_merge_data[:8803]:
        kg_builder.process_medical_data(data)
        
    # 处理other数据
    for data in other_medical_merge_data[8803:]:
        kg_builder.process_medical_data(data, is_other=True)
        
    # 非疾病实体去重、关系去重
    kg_builder.deduplicate_entities_relations()

    # 处理论文数据
    instruction_path = 'data/kg_info/result_instruction.json'
    kg_builder.process_paper_data(instruction_path)

    # 再次去重
    # 非疾病实体去重、关系去重
    kg_builder.deduplicate_entities_relations()

    # 清理实体(度数很高的无用节点)
    kg_builder.clean_entities()

    # 保存实体和关系
    kg_builder.save_to_files()

    # 导入neo4j
    kg_builder.import_to_neo4j()

if __name__ == "__main__":
    main()