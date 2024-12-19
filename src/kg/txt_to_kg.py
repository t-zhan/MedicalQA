import os
import py2neo
import json
from tqdm import tqdm
from extract_info import *


# 配置参数
MAX_LINES = None  # prevent, cause, easy_get提取行数 # 提取全部设置为None
MAX_PAPERS = None   # 提取几篇论文 # 提取全部设置为None

class MedicalKGBuilder:
    def __init__(self,
                 db_name='neo4j',
                 neo4j_host="bolt://10.176.28.10:7687",
                 neo4j_user="neo4j",
                 neo4j_password="medical123456",
                 disease_data_path='data/kg_info/other_medical_merge.json',
                 instruction_path='data/kg_info/result_instruction.json',
                 entity_base_path='data/kg_info/fixed_entity_relation/entity',
                 relation_base_path='data/kg_info/fixed_entity_relation/relation'
                 ):
        
        # 实体和关系的基础路径
        self.entity_base_path = entity_base_path
        self.relation_base_path = relation_base_path
        
        # 定义所有实体类型
        self.entity_types = [
            "生病原因", "易感人群", "预防措施", "并发症", "医保状态",
            "药品", "食物", "检查项目", "科室",
            "疾病症状", "治疗方法", "传染方式", "发病部位"
        ]
        
        # 定义所有关系类型
        self.relation_types = [
            "生病原因", "易感人群", "预防措施", "并发症", "医保",
            "推荐药品", "忌讳食物", "推荐食物", "检查项目", "科室",
            "症状", "治疗方法", "传染方式", "发病部位"
        ]
        
        # 初始化存储结构
        self.disease_entities = []  # 存储疾病节点及其属性
        self.other_entities = {entity_type: [] for entity_type in self.entity_types}
        self.relationships = []
        
        # 初始化neo4j连接
        self.graph = py2neo.Graph(
            neo4j_host,
            auth=(neo4j_user, neo4j_password),
            name=db_name
        )
        
        # 保存data_path
        self.disease_data_path = disease_data_path
        self.instruction_path = instruction_path

    def load_disease_data(self):
        """从JSON文件加载疾病数据"""
        other_medical_merge_data = load_json_data(self.disease_data_path)
        
        # 处理medical数据
        for data in other_medical_merge_data[:8803]:
            disease_entity = self._create_disease_entity(data, is_other=False)
            self.disease_entities.append(disease_entity)
            
        # 处理other数据
        for data in other_medical_merge_data[8803:]:
            disease_entity = self._create_disease_entity(data, is_other=True)
            self.disease_entities.append(disease_entity)

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

    def load_other_entities(self):
        """从txt文件加载非疾病实体"""
        for entity_type in self.entity_types:
            file_path = os.path.join(self.entity_base_path, f'{entity_type}.txt')
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    entities = [line.strip() for line in f if line.strip()]
                    self.other_entities[entity_type] = entities

    def load_relationships(self):
        """从txt文件加载关系"""
        for relation_type in self.relation_types:
            file_path = os.path.join(self.relation_base_path, f'{relation_type}.txt')
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            self.relationships.append(tuple(parts))
        print(f"总关系数量: {len(self.relationships)}")
    def load_paper_disease_data(self):
        """从论文数据中提取疾病实体"""
        print("正在从论文数据中提取疾病实体...")
        
        # 提取论文关系类型
        rel_list = ['disease_has_drug', 'disease_has_treatment',
                'disease_no_eat','disease_do_check',
                'disease_has_symptom', 'disease_easy_get_population', 
                'disease_has_prevent', 'disease_acompany_disease', 
                'disease_has_cause', 'disease_has_spreads_way']
        rel_dict = {rel:[] for rel in rel_list}

        # 提取论文关系和关系字典
        _, instruction_rel_dict = extract_paper_rel(self.instruction_path, rel_dict, set(), max_papers=MAX_PAPERS)
        
        # 提取论文疾病
        paper_disease = set()
        for key in instruction_rel_dict:
            paper_disease.update(extract_disease(instruction_rel_dict[key]))
        
        # 将论文中的每个疾病添加到disease_entities中
        for disease_name in paper_disease:
            disease_dict = {"疾病名称": disease_name}
            self.disease_entities.append(disease_dict)


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
        
        # 创建疾病节点
        print("正在创建疾病节点")
        for disease in tqdm(self.disease_entities, desc="创建疾病节点"):
            alias = disease.get("别名", "")
            if isinstance(alias, list):
                alias = '、'.join(alias)
                
            node = py2neo.Node("疾病",
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
        
        # 创建其他实体节点
        print("正在创建非疾病实体节点")
        for entity_type, entities in tqdm(self.other_entities.items(), desc=f"创建{entity_type}实体节点"):
            for entity in entities:
                node = py2neo.Node(entity_type, name=entity)
                self.graph.create(node)
        
        # 创建关系
        print("正在创建关系，时间较长，请耐心等待")
        for rel in tqdm(self.relationships, desc="创建所有关系"):
            start_type, start_name, rel_type, end_type, end_name = rel
            
            start_node = self.graph.nodes.match(start_type, name=start_name).first()
            end_node = self.graph.nodes.match(end_type, name=end_name).first()
            
            if start_node and end_node:
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
    
    # 创建知识图谱构建器
    kg_builder = MedicalKGBuilder()
    
    # 加载疾病数据
    print("正在加载疾病数据")
    kg_builder.load_disease_data()
    
    # 加载论文中的疾病数据
    kg_builder.load_paper_disease_data()
    
    # 加载其他实体
    print("正在加载其他实体")
    kg_builder.load_other_entities()
    
    # 加载关系
    print("正在加载关系")
    kg_builder.load_relationships()
    
    # 导入neo4j
    kg_builder.import_to_neo4j()

if __name__ == "__main__":
    main()
    # 总关系数量:393713