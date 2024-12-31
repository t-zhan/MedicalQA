import torch
import transformers
import re
import json


def generate_intent(input, model, tokenizer):
    
    messages = [
        {"role": "system", "content": "你是一个疾病领域的专家，请识别用户问题中的查询意图。"},
        {"role": "user", "content": f"""
阅读下列提示，回答问题（问题在输入的最后）:
当你试图识别用户问题中的查询意图时，你需要仔细分析问题，并在16个预定义的查询类别中一一进行判断。对于每一个类别，思考用户的问题是否含有与该类别对应的意图。如果判断用户的问题符合某个特定类别，就将该类别加入到输出列表中。这样的方法要求你对每一个可能的查询意图进行系统性的考虑和评估，确保没有遗漏任何一个可能的分类。

**查询类别**
- "查询疾病简介"
- "查询疾病病因"
- "查询疾病预防措施"
- "查询疾病治疗周期"
- "查询治愈概率"
- "查询疾病易感人群"
- "查询疾病所需药品"
- "查询疾病宜吃食物"
- "查询疾病忌吃食物"
- "查询疾病所需检查项目"
- "查询疾病所属科目"
- "查询疾病的症状"
- "查询疾病的治疗方法"
- "查询疾病的并发疾病"
- "查询症状对应疾病"
- "查询症状治疗措施"
- "查询症状的预防措施"
- "查询症状的原因"

在处理用户的问题时，请按照以下步骤操作：
- 仔细阅读用户的问题。
- 对照上述查询类别列表，依次考虑每个类别是否与用户问题相关。
- 如果用户问题明确或隐含地包含了某个类别的查询意图，请将该类别的描述添加到输出列表中。
- 确保最终的输出列表包含了所有与用户问题相关的类别描述。

以下是一些含有隐晦性意图的例子，每个例子都采用了输入和输出格式，并包含了对你进行思维链形成的提示：
**示例1：**
输入："睡眠不好，这是为什么？"
输出：["查询症状对应疾病","查询症状的原因","查询疾病简介"]  # 这个问题隐含地询问了睡眠不好的病因
**示例2：**
输入："感冒了，怎么办才好？"
输出：["查询疾病简介","查询疾病所需药品", "查询疾病的治疗方法"]  # 用户可能既想知道应该吃哪些药品，也想了解治疗方法
**示例3：**
输入："跑步后膝盖痛，需要吃点什么？"
输出：["查询症状对应疾病","查询疾病宜吃食物", "查询疾病所需药品"]  # 这个问题可能既询问宜吃的食物，也可能在询问所需药品
**示例4：**
输入："我怎样才能避免冬天的流感和感冒？"
输出：["查询疾病简介","查询疾病简介","查询疾病预防措施"]  # 询问的是预防措施，但因为提到了两种疾病，这里隐含的是对共同预防措施的询问
**示例5：**
输入："头疼是什么原因，应该怎么办？"
输出：["查询症状对应疾病","查询症状的原因","查询疾病简介","查询疾病病因", "查询疾病的治疗方法"]  # 用户询问的是头疼的病因和治疗方法
**示例6：**
输入："如何知道自己是不是有艾滋病？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有艾滋病，一定一定要进行相关检查，这是根本性的！其次是查看疾病的病因，看看自己的行为是不是和病因重合。
**示例7：**
输入："我该怎么知道我自己是否得了21三体综合症呢？"
输出：["查询疾病简介","查询疾病所需检查项目","查询疾病病因"]  # 用户想知道自己是不是有21三体综合症，一定一定要进行相关检查(比如染色体)，这是根本性的！其次是查看疾病的病因。
**示例8：**
输入："感冒了，怎么办？"
输出：["查询疾病简介","查询疾病的治疗方法","查询疾病所需药品","查询疾病所需检查项目","查询疾病宜吃食物"]  # 问怎么办，首选治疗方法。然后是要给用户推荐一些药，最后让他检查一下身体。同时，也推荐一下食物。
**示例9：**
输入："癌症会引发其他疾病吗？"
输出：["查询疾病简介","查询疾病的并发疾病","查询疾病简介"]  # 显然，用户问的是疾病并发疾病，随后可以给用户科普一下癌症简介。
通过上述例子，我们希望你能够形成一套系统的思考过程，以准确识别出用户问题中的所有可能查询意图。请仔细分析用户的问题，考虑到其可能的多重含义，确保输出反映了所有相关的查询意图。

**注意：**
- 你的所有输出，都必须在这个范围内上述**查询类别**范围内，不可创造新的名词与类别！
- 参考上述5个示例：在输出查询意图对应的列表之后，请紧跟着用"#"号开始的注释，简短地解释为什么选择这些意图选项。注释应当直接跟在列表后面，形成一条连续的输出。
- 你的输出的类别数量不应该超过5，如果确实有很多个，请你输出最有可能的5个！同时，你的解释不宜过长，但是得富有条理性。

现在，你已经知道如何解决问题了，请你解决下面这个问题并将结果输出！
问题输入："{input}"
输出的时候请确保输出内容都在**查询类别**中出现过。确保输出类别个数**不要超过5个**！确保你的解释和合乎逻辑的！注意，如果用户询问了有关疾病的问题，一般都要先介绍一下疾病，也就是有"查询疾病简介"这个需求,如果用户询问了有关症状的问题，一般都先介绍一下症状对应疾病，也就是有"查询症状对应疾病"这个需求。
再次检查你的输出都包含在**查询类别**:"查询疾病简介"、"查询疾病病因"、"查询疾病预防措施"、"查询疾病治疗周期"、"查询治愈概率"、"查询疾病易感人群"、"查询疾病所需药品"、"查询疾病宜吃食物"、"查询疾病忌吃食物"、"查询疾病所需检查项目"、"查询疾病所属科目"、"查询疾病的症状"、"查询疾病的治疗方法"、"查询疾病的并发疾病"、"查询药品的生产商"。
"""},
    ]
    pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16
            )
    sequences = pipeline(
            messages,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            max_length=4096,
            pad_token_id=tokenizer.eos_token_id)

    output = sequences[0]['generated_text'][-1]['content']
    
    output = re.search(r"\[.*?\]", output).group()
    output = output.replace("'", '"')
    output = json.loads(output)
    print(f"intent:{output}")

    return output


def generate_keywords(input, model, tokenizer):
    
    messages = [
        {"role": "system", "content": "你是一个疾病领域的实体识别的专家，从用户的问题中识别出实体关键词。"},
        {"role": "user", "content": f"根据问题，提取所有实体关键词，回答格式：['entity1','entity2'].例子1:user：‘眼睛干涩和疲劳的原因是什么？’这句话包含的实体有？assistant：['眼睛干涩'，‘疲劳’]。例子2:user：‘手脚冰凉可能是什么问题？’这句话包含的实体有？assistant：['手脚冰凉']。例子3:user：‘感冒应该吃什么药？’这句话包含的实体有？assistant：['流行性感冒']。例子4:user：‘偏头痛是由什么导致的？’这这句话包含的实体有？assistant：['偏头痛']。例子5:user：‘韧带拉伤和髂胫束综合症会有什么症状？’这句话包含的实体有？assistant：['韧带拉伤','髂胫束综合症']。user：‘{input}’这句话包含的实体有？assistant："},
    ]
    pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            max_length=1024)
    sequences = pipeline(
            messages,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            max_length=1024,
            pad_token_id=tokenizer.eos_token_id)

    output = sequences[0]['generated_text'][-1]['content']
    
    output = re.search(r"\[.*?\]", output).group()
    output = output.replace("'", '"')
    output = json.loads(output)

    print(f"keywords:{output}")
    return output


def generate_answer(input, context, intent, model, tokenizer):
    
    messages = [
        {"role": "system", "content": """
    你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。
    回答逻辑要根据用户意图，回答内容要按照数据并结合任何相关的通用知识.
    提供尽可能多的信息以满足用户意图。
    回答中不要出现‘根据提供的信息’‘问题’‘数据’等字眼.
    """},
        {"role": "user", "content": f"""
    ---目标---

    生成一个符合要求的回答，以回应用户的问题，总结输入数据中的所有相关信息，数据中信息以['主体类型', '主体', '关系', '客体类型', '客体']的形式呈现，并结合任何相关的通用知识。回答包含用户意图中的每一项。

    ---意图---

    {intent}

    ---数据---

    {context}

    ---问题---

    {input}

    ---回答---
    """},
    ]
    pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16
            )
    sequences = pipeline(
            messages,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            max_length=8192,
            pad_token_id=tokenizer.eos_token_id)

    output = sequences[0]['generated_text'][-1]['content']
    
    # output = re.search(r"\[.*?\]", output).group()
    # output = output.replace("'", '"')
    # output = json.loads(output)
    print(f"answer:\n{output}\n")

    return output


def output_kw(input, model, tokenizer):
    # 问题实体提取
    intent = generate_intent(input, model, tokenizer)
    output = generate_keywords(input, model, tokenizer)
    data = {}
    if "查询症状对应疾病" in intent:
        data['type'] = "症状"
        data['entity'] = output
    else:
        data['type'] = "疾病"
        data['entity'] = output
    data['intent'] = intent
    return data

def output_kw_d(input, model, tokenizer):
    # 问题实体提取
    intent = generate_intent(input, model, tokenizer)
    output = generate_keywords(input, model, tokenizer)
    data = {}
    data['type'] = "疾病"
    data['entity'] = output
    data['intent'] = intent
    return data

def output_kw_s(input, model, tokenizer):
    # 问题实体提取
    intent = generate_intent(input, model, tokenizer)
    output = generate_keywords(input, model, tokenizer)
    data = {}

    data['type'] = "症状"
    data['entity'] = output

    data['intent'] = intent
    return data

def g_new_context(new_context, context, data, i):
    new_context[f'{i}'] = {}
    new_context[f'{i}']['疾病'] = context[i]['属性']['疾病名称']
    if "查询疾病简介" in data['intent']:
        new_context[f'{i}']['简介'] = context[i]['属性']['疾病简介']
    if "查询疾病病因" in data['intent']:
        new_context[f'{i}']['生病原因'] = [i[4] for i in context[i]['疾病——生病原因']]
    if "查询疾病预防措施" in data['intent']:
        new_context[f'{i}']['预防措施'] = [i[4] for i in context[i]['疾病——预防措施']]
    if "查询疾病治疗周期" in data['intent']:
        new_context[f'{i}']['治疗周期'] = context[i]['属性']['治疗周期']
    if "查询治愈概率" in data['intent']:
        new_context[f'{i}']['治愈率'] = context[i]['属性']['治愈率']
    if "查询疾病易感人群" in data['intent']:
        new_context[f'{i}']['易感人群'] = [i[4] for i in context[i]['疾病——易感人群']]
    if "查询疾病所需药品" in data['intent']:
        new_context[f'{i}']['推荐药品'] = [i[4] for i in context[i]['疾病——推荐药品']]
    if "查询疾病宜吃食物" in data['intent']:
        new_context[f'{i}']['推荐食物'] = [i[4] for i in context[i]['疾病——推荐食物']]
    if "查询疾病忌吃食物" in data['intent']:
        new_context[f'{i}']['忌讳食物'] = [i[4] for i in context[i]['疾病——忌讳食物']]
    if "查询疾病所需检查项目" in data['intent']:
        new_context[f'{i}']['检查项目'] = [i[4] for i in context[i]['疾病——检查项目']]
    if "查询疾病所属科目" in data['intent']:
        new_context[f'{i}']['科室'] = [i[4] for i in context[i]['疾病——科室']]
    if "查询疾病的症状" in data['intent']:
        new_context[f'{i}']['症状'] = [i[4] for i in context[i]['疾病——疾病症状']]
    if "查询疾病的治疗方法" in data['intent']:
        new_context[f'{i}']['治疗方法'] = [i[4] for i in context[i]['疾病——治疗方法']]
    if "查询疾病的并发疾病" in data['intent']:
        new_context[f'{i}']['并发症'] = [i[4] for i in context[i]['疾病——并发症']]
    if "查询症状对应疾病" in data['intent']:
        new_context[f'{i}']['症状'] = [i[4] for i in context[i]['疾病——疾病症状']]
    if "查询症状治疗措施" in data['intent']:
        new_context[f'{i}']['症状'] = [i[4] for i in context[i]['疾病——疾病症状']]
        new_context[f'{i}']['治疗方法'] = [i[4] for i in context[i]['疾病——治疗方法']]
    if "查询症状的预防措施" in data['intent']:
        new_context[f'{i}']['症状'] = [i[4] for i in context[i]['疾病——疾病症状']]
        new_context[f'{i}']['预防措施'] = [i[4] for i in context[i]['疾病——预防措施']]
    if "查询症状的原因" in data['intent']:
        new_context[f'{i}']['症状'] = [i[4] for i in context[i]['疾病——疾病症状']]
        new_context[f'{i}']['生病原因'] = [i[4] for i in context[i]['疾病——生病原因']]
    return new_context


def select_context(context, data):
    new_context = {}
    for i in range(len(context)):
        new_context = g_new_context(new_context,context,data,i)
    return new_context


def generate_direct_answer(input, model, tokenizer):
    
    messages = [
        {"role": "system", "content": "你是一个乐于助人的医学专家，根据提供的信息以医生的身份回答问题。"},
        {"role": "user", "content": f"{input}"},
    ]
    pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16
            )
    sequences = pipeline(
            messages,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            max_length=8192,
            pad_token_id=tokenizer.eos_token_id)

    output = sequences[0]['generated_text'][-1]['content']
    

    print(f"direct_answer:\n{output}\n")
    return output