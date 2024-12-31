import json
import pandas as pd
import numpy as np
import re


def process_NA_null(x):
    if isinstance(x, str):
        if x == 'N/A':
            return ''
        else:
            return x
    else:
        return ''
    

def other_preprocess(x):
    if isinstance(x, str):
        if x == 'N/A':
            return []
        else:
            a = re.split(r'[,，]',x)
            return [i.strip() for i in a]

    else:
        return []
    
def other_preprocess_dun(x):
    if isinstance(x, str):
        if x == 'N/A':
            return []
        else:
            a = x.split('、')
            return [i.strip() for i in a]

    else:
        return []

def all_preprocess_get_way(x):
    if isinstance(x, str):
        if x == '':
            return []
        else:
            a = re.split(r'[、,. (, )，或]', x)
            tmp = [re.sub(r'^\d+\s*|\s*\d+|(。)|(。 )$', '', i.strip()) for i in a]
            return [j for j in tmp if j]

    else:
        return []

if __name__ == '__main__':
    otherdata = []
    otherdata_path = 'data/data_preprocess/other_medical_line.json'
    with open(otherdata_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line:
                otherdata.append(json.loads(line))

    # 从otherdata里面提取有用的信息
    otherdata_flat = []
    for item in otherdata:

        # 先判断这条数据是不是空的
        disease_name = item.get('名字', '')
        if not disease_name:
            continue

        othername = item.get('别名', '')
        position = item.get('发病部位', '') 
        yibao_status = item.get('是否医保', '') 

        row = {
            "疾病名称": disease_name,
            "科室": item.get('挂号科室', ''),
            "疾病简介": item.get('基本信息', ''),
            "易感人群": item.get('多发人群', ''),
            "传染方式": item.get('传染性', ''),
            "治疗方法": item.get('治疗方法', ''),
            "治疗周期": item.get('治疗周期', ''),
            "治愈率": item.get('治愈率', ''),
            "治疗费用": item.get('治疗费用', ''),
            "疾病症状": item.get('典型症状', ''),
            "检查项目": item.get('临床检查', ''),
            "常用药品": item.get('常用药品', ''),
            "并发症": item.get('并发症', ''),
            "别名": item.get('别名', ''),
            "发病部位": item.get('发病部位', ''),
            "是否医保": item.get('是否医保', '') 
        }
        otherdata_flat.append(row)

    # 转换成DataFrame
    df_other = pd.DataFrame(otherdata_flat)

    df_other = df_other[~((df_other.drop(columns=["疾病名称"]) == "N/A").all(axis=1))]
    for col in ['科室', '易感人群', '疾病症状', '检查项目', '常用药品', '并发症', '别名', '发病部位']:
        df_other[col] = df_other[col].apply(other_preprocess)
    for col in ['治疗方法']:
        df_other[col] = df_other[col].apply(other_preprocess_dun)

    for col in ['患病比例', '注意事项', '生病原因', '预防措施', '推荐食物', '忌讳食物', '推荐药品']:
        df_other.loc[:,col] = np.nan
    for col in ['疾病简介', '传染方式', '治疗周期', '治愈率', '治疗费用']:
        df_other[col] = df_other[col].apply(process_NA_null)

    # 寻医问药主体疾病信息
    data_medical = []
    data_medical_path = 'data/data_preprocess/old_new_merge.json'
    with open(data_medical_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line:
                data_medical.append(json.loads(line))

    df_medical = pd.DataFrame(data_medical)

    df_result = df_medical.merge(df_other[['疾病名称', '别名', '发病部位', '是否医保']], on = '疾病名称', how = 'left')

    # 找出 df_other 没有和 df_medical 重合的疾病, 把这些新的病单独存到一个DataFrame里
    df_other_non_overlap = df_other[~df_other['疾病名称'].isin(df_medical['疾病名称'])]

    # 把 df_other 的新病和 df_medical, 合并
    df_result = pd.concat([df_result, df_other_non_overlap])

    for col in ['传染方式']:
        df_result[col] = df_result[col].apply(all_preprocess_get_way)

    # 把 df_result 导出为 csv 文件
    out_file_path = 'data/kg_info/other_medical_merge.csv'
    df_result.to_csv(out_file_path, index=False, encoding='utf-8-sig')

    # 把 df_result 转换为 json 文件
    json_output = df_result.to_json(orient='records', lines=True, force_ascii=False)
    with open('data/kg_info/other_medical_merge.json', 'w', encoding='utf-8-sig') as f:
        f.write(json_output)