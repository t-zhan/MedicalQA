import json
import pandas as pd
import re

data = []
data_path = 'data/raw_data/new_medical_line.json'
with open(data_path, 'r', encoding='utf-8-sig') as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

data_flat = []
for item in data:

    # 加载 json 文件里面疾病的 information
    basic_info = item['basic_info']

    # 先判断这条数据是不是空的
    disease_name = basic_info.get("name", '')
    if not disease_name:
        continue


    cause_info = item['cause_info'] 
    prevent_info = item['prevent_info'] 
    sysptom_info = item['symptom_info'] 
    treat_info = item['treat_info'] 
    food_info = item['food_info'] 
    cure_way = basic_info["attributes"][6].split('：')[1].strip()

    row = {

        # basci_info
        "疾病名称": basic_info.get("name", ''),
        "科室": basic_info.get("category")[1:],
        "疾病简介": basic_info.get("desc")[0].strip(),
        "患病比例": basic_info["attributes"][1].split('：')[1],
        "易感人群": basic_info["attributes"][2].split('：')[1],
        "传染方式": basic_info["attributes"][3].split('：')[1],
        "并发症": basic_info["attributes"][4].split('：')[1],
        "治疗方法": re.split(r'[、 ,，]+', cure_way),
        "治疗周期": basic_info["attributes"][7].split('：')[1],
        "治愈率": basic_info["attributes"][8].split('：')[1],
        "治疗费用": basic_info["attributes"][9].split('：')[1],
        "注意事项": basic_info["attributes"][10] if 10 < len(basic_info["attributes"]) else '',

        # cause_info
        "生病原因": cause_info,

        # prevent_info
        "预防措施": prevent_info,

        # sysptom_info
        "疾病症状": sysptom_info[0] if sysptom_info[0] else [],

        # food_info
        "推荐食物": food_info.get('good', []) + food_info.get('recommand', []),
        "忌讳食物": food_info.get('bad', [])

    }
    data_flat.append(row)

# 转换成DataFrame
df_guo = pd.DataFrame(data_flat)



data2 = []
data2_path = 'data/raw_data/old_medical.json'
with open(data2_path, 'r', encoding='utf-8-sig') as f:
    for line in f:
        line = line.strip().rstrip(',')
        if line:
            data2.append(json.loads(line))

df2 = pd.DataFrame(data2)
df2.drop(columns=['_id', 'yibao_status', 'drug_detail'], inplace=True)
df2.rename(columns = {'name': '疾病名称', 'check': '检查项目', 'common_drug': '常用药品', 'recommand_drug': '推荐药品', 'acompany': '并发症'}, inplace=True)

df2 = df2.drop_duplicates(subset='疾病名称')

df_result = df_guo.merge(df2[['疾病名称', '检查项目', '常用药品', '推荐药品', '并发症']], on = '疾病名称', how = 'left')
df_result['并发症'] = df_result['并发症_y']
df_result.drop(columns=['并发症_x', '并发症_y'], inplace=True)

# 针对特定列替换 DataFrame 中的 NaN 为空列表
for column in ["检查项目", "常用药品", "推荐药品"]:
    df_result[column] = df_result[column].apply(lambda x: x if isinstance(x, list) else [])

# 把merge后的 dataframe 导出为 csv 文件 # 预览查看格式
# out_file_path = 'data/data_preprocess/old_new_merge.csv'
# df_result.to_csv(out_file_path, index=False, encoding='utf-8-sig')


# 把 old new merge 后的 dataframe 转换为 json 文件
json_output = df_result.to_json(orient='records', lines=True, force_ascii=False)
with open('data/raw_data/old_new_merge.json', 'w', encoding='utf-8-sig') as f:
    f.write(json_output)
