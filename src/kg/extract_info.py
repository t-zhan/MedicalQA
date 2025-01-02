import json
import pandas as pd

def extract_paper_rel(file_path, rel_dict, disease_set, max_papers=None): # paper
    with open (file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    paper_relationship = []
    error_count = 0
    paper_count = 0
    
    # 如果指定了max_papers,则只处理指定数量的papers
    papers_to_process = data[:max_papers] if max_papers else data
    
    for paper in papers_to_process:
        paper_count += 1
        for line in paper:
            if line:
                try:
                    line = json.loads(line)
                except json.JSONDecodeError as e:
                    error_count += 1
                    # print(f"Error parsing JSON: {e} for paper {paper_count} line {paper.index(line) + 1}")
                    continue

                for rel in rel_dict.keys():
                    if rel in line.keys():
                        for item in line[rel]:
                            if not isinstance(item, dict):
                                continue
                            subj = item.get("subject")
                            if subj in disease_set:
                                continue
                            obj = item.get("object")
                            rel = rel
                            if rel == 'disease_has_drug':
                                rel_dict[rel].append(('疾病', subj, '推荐药品', '药品', obj))
                                paper_relationship.append(('疾病', subj, '推荐药品', '药品', obj))
                            if rel == 'disease_has_treatment':
                                rel_dict[rel].append(('疾病', subj, '治疗方法', '治疗方法', obj))
                                paper_relationship.append(('疾病', subj, '治疗方法', '治疗方法', obj))
                            if rel == 'disease_no_eat':
                                rel_dict[rel].append(('疾病', subj, '忌讳食物', '食物', obj))
                                paper_relationship.append(('疾病', subj, '忌讳食物', '食物', obj))
                            if rel == 'disease_do_check':
                                rel_dict[rel].append(('疾病', subj, '检查项目', '检查项目', obj))
                                paper_relationship.append(('疾病', subj, '检查项目', '检查项目', obj))
                            if rel == 'disease_has_symptom':
                                rel_dict[rel].append(('疾病', subj, '症状', '疾病症状', obj))
                                paper_relationship.append(('疾病', subj, '症状', '疾病症状', obj))
                            if rel == 'disease_easy_get_population':
                                rel_dict[rel].append(('疾病', subj, '易感人群', '易感人群', obj))
                                paper_relationship.append(('疾病', subj, '易感人群', '易感人群', obj))
                            if rel == 'disease_has_prevent':
                                rel_dict[rel].append(('疾病', subj, '预防措施', '预防措施', obj))
                                paper_relationship.append(('疾病', subj, '预防措施', '预防措施', obj))
                            if rel == 'disease_acompany_disease':
                                rel_dict[rel].append(('疾病', subj, '并发症', '并发症', obj))
                                paper_relationship.append(('疾病', subj, '并发症', '并发症', obj))
                            if rel == 'disease_has_cause':
                                rel_dict[rel].append(('疾病', subj, '生病原因', '生病原因', obj))
                                paper_relationship.append(('疾病', subj, '生病原因', '生病原因', obj))
                            if rel == 'disease_has_spreads_way':
                                rel_dict[rel].append(('疾病', subj, '传染方式', '传染方式', obj))
                                paper_relationship.append(('疾病', subj, '传染方式', '传染方式', obj))

    # print(f"total errors: {error_count}")
    return list(set(paper_relationship)), { rel_name: list(set(rel_list_of_tuple)) for rel_name, rel_list_of_tuple in rel_dict.items() }


def extract_rel(file_path, rel, max_lines=None): # prevent, cause, easy_get
    with open (file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 如果指定了max_lines,则只读取指定行数
    if max_lines is not None:
        data = data[:max_lines]

    result = []
    error_count = 0

    for line in data:
        if line:
            try:
                line = json.loads(line)
            except json.JSONDecodeError as e:
                error_count += 1
                print(f"Error parsing JSON: {e} for {rel} in line {data.index(line) + 1}")
                continue

            for item in line[rel]:
                subj = item.get("subject")
                obj = item.get("object")
                rel = rel
                if subj and obj:
                    result.append((subj, rel, obj))
    # print(f"total errors: {error_count}")
    return list(set(result))

def extract_entity(rel_result): # rel_result是一个list, 每个元素是一个三元组
    ent = set()
    for i in range(len(rel_result)):
        if rel_result[i][2] not in ent:
            ent.add(rel_result[i][2])
    return list(ent)

def extract_paper_entity(rel_result): # rel_result是一个list, 每个元素是一个五元组
    ent = set()
    for i in range(len(rel_result)):
        if rel_result[i][4] not in ent:
            ent.add(rel_result[i][4])
    return list(ent)

def extract_disease(rel_result): # rel_result是一个list, 每个元素是一个五元组
    ent = set()
    for item in rel_result:
        ent.add(item[1])
    return ent


if __name__ == '__main__':
    rel_list = ['disease_has_drug', 'disease_has_treatment',
            'disease_no_eat','disease_do_check',
             'disease_has_symptom', 'disease_easy_get_population', 
             'disease_has_prevent', 'disease_acompany_disease', 
             'disease_has_cause', 'disease_has_spreads_way']
    rel_dict = {rel:[] for rel in rel_list}
    instruction_path = 'data/kg_info/result_instruction.json'


    disease_other_medical_path = 'data/kg_info/other_medical_merge.csv'
    df_disease_other_medical = pd.read_csv(disease_other_medical_path)
    disease_set = set(df_disease_other_medical['疾病名称'])

    paper_relationship, instruction_rel_dict = extract_paper_rel(instruction_path, rel_dict, disease_set, max_papers=None)
    paper_disease = set()
    for key in instruction_rel_dict:
        paper_disease.update(extract_disease(instruction_rel_dict[key]))

    has_drug = extract_paper_entity(instruction_rel_dict['disease_has_drug'])
    has_treatment = extract_paper_entity(instruction_rel_dict['disease_has_treatment'])
    no_eat = extract_paper_entity(instruction_rel_dict['disease_no_eat'])
    do_check = extract_paper_entity(instruction_rel_dict['disease_do_check'])
    has_symptom = extract_paper_entity(instruction_rel_dict['disease_has_symptom'])
    easy_get_population = extract_paper_entity(instruction_rel_dict['disease_easy_get_population'])
    has_prevent = extract_paper_entity(instruction_rel_dict['disease_has_prevent'])
    acompany_disease = extract_paper_entity(instruction_rel_dict['disease_acompany_disease'])
    has_cause = extract_paper_entity(instruction_rel_dict['disease_has_cause'])
    has_spreads_way = extract_paper_entity(instruction_rel_dict['disease_has_spreads_way'])



    cause_path = 'data/kg_info/result_cause.json'
    easy_get_path = 'data/kg_info/result_easy_get.json'
    prevent_path = 'data/kg_info/result_prevent.json'
    cause_relationship = extract_rel(cause_path, "has_cause", max_lines=100)
    easy_get_relationship = extract_rel(easy_get_path, "easy_get", max_lines=100)
    prevent_relationship = extract_rel(prevent_path, "has_prevent", max_lines=100)
    cause = extract_entity(cause_relationship)
    easy_get = extract_entity(easy_get_relationship)
    prevent = extract_entity(prevent_relationship)
    print(len(cause))
    print(len(easy_get))
    print(len(prevent))