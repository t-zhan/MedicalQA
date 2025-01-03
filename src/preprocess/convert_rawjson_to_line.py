import json

# convert new_medical_to_line
with open('data/raw_data/new_medical_rawjson.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('data/raw_data/new_medical_line.json', 'w', encoding='utf-8') as f:
    for record in data:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


# convert other_medical_to_line
data = []
with open('data/raw_data/other_medical_rawjson.json.json', 'r', encoding='utf-8') as f:
    content = f.read()  # 读取整个文件内容
    records = content.split('}\n{')  # 以 '}\n{' 为分隔符进行分割

    # 处理每条记录，确保它们是有效的 JSON
    for record in records:
        # 添加缺失的开头和结尾的括号
        record = record.strip()
        if not record.startswith('{'):
            record = '{' + record
        if not record.endswith('}'):
            record = record + '}'
        
        # 解析 JSON 对象
        data.append(json.loads(record))

with open('data/raw_data/other_medical_line.json', 'w', encoding='utf-8') as f:
    for record in data:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')