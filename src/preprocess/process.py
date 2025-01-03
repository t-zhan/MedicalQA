#coding=utf-8
##导包
import torch
import json
import os
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)
import os
from modelscope import snapshot_download

# 设置可见的 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

##加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = snapshot_download('ZJUNLP/OneKE', local_dir='models/OneKE', cache_dir='models/OneKE/cache')
config = AutoConfig.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 4bit量化OneKE
quantization_config=BitsAndBytesConfig(     
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    device_map='auto',  
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()

## 任务选取
split_num_mapper = {'NER':6, 'RE':1, 'EE':4, 'EET':4, 'EEA':4, 'KG':1}

## 角色定位
instruction_mapper = {
    'NERzh': '你是专门进行实体抽取的专家.请从input中抽取出符合schema定义的实体,不存在的实体类型返回空列表.请按照JSON字符串的格式回答.',
    'REzh': '你是专门进行关系抽取的专家.请从input中抽取出符合schema定义的关系三元组,不存在的关系返回空列表.请按照JSON字符串的格式回答.',
    'EEzh': '你是专门进行事件提取的专家.请从input中抽取出符合schema定义的事件,不存在的事件返回空列表,不存在的论元返回NAN,如果论元存在多值请返回列表.请按照JSON字符串的格式回答.',
    'EETzh': '你是专门进行事件提取的专家.请从input中抽取出符合schema定义的事件类型及事件触发词,不存在的事件返回空列表.请按照JSON字符串的格式回答.',
    'EEAzh': '你是专门进行事件论元提取的专家.请从input中抽取出符合schema定义的事件论元及论元角色,不存在的论元返回NAN或空字典,如果论元存在多值请返回列表.请按照JSON字符串的格式回答.',
    'KGzh': '你是一个图谱实体知识结构化专家.根据输入实体类型(entity type)的schema描述,从文本中抽取出相应的实体实例和其属性信息,不存在的属性不输出, 属性存在多值就返回列表,并输出为可解析的json格式.',
    'NERen': 'You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string.',
    'REen': 'You are an expert in relationship extraction. Please extract relationship triples that match the schema definition from the input. Return an empty list for relationships that do not exist. Please respond in the format of a JSON string.',
    'EEen': 'You are an expert in event extraction. Please extract events from the input that conform to the schema definition. Return an empty list for events that do not exist, and return NAN for arguments that do not exist. If an argument has multiple values, please return a list. Respond in the format of a JSON string.',
    'EETen': 'You are an expert in event extraction. Please extract event types and event trigger words from the input that conform to the schema definition. Return an empty list for non-existent events. Please respond in the format of a JSON string.',
    'EEAen': 'You are an expert in event argument extraction. Please extract event arguments and their roles from the input that conform to the schema definition, which already includes event trigger words. If an argument does not exist, return NAN or an empty dictionary. Please respond in the format of a JSON string.', 
    'KGen': 'You are an expert in structured knowledge systems for graph entities. Based on the schema description of the input entity type, you extract the corresponding entity instances and their attribute information from the text. Attributes that do not exist should not be output. If an attribute has multiple values, a list should be returned. The results should be output in a parsable JSON format.',
}



##生成指令
def get_instruction(language, task, schema, input):
    sintructs = []
    split_num = split_num_mapper[task]
    if type(schema) == dict:
        sintruct = json.dumps({'instruction':instruction_mapper[task+language], 'schema':schema, 'input':input}, ensure_ascii=False)
        sintructs.append(sintruct)
    else:
        split_schemas = [schema[i:i+split_num] for i in range(0, len(schema), split_num)]
        for split_schema in split_schemas:
            sintruct = json.dumps({'instruction':instruction_mapper[task+language], 'schema':split_schema, 'input':input}, ensure_ascii=False)
            sintructs.append(sintruct)
    return sintructs

task = 'RE'
language = 'zh'
schema ={
    # 'drug_of':'该关系类型描述的是疾病和药物的关系。展示了药物所治疗的疾病。主体是药物，客体是疾病。疾病必须出现在以下列表中：{disease_list}',
    # 'do_eat':'该关系类型描述的是疾病和食物的关系。展示了疾病患病期间宜吃的食物。主体是疾病，客体是食物。',
    # 'no_eat':'该关系类型描述的是疾病和食物的关系。展示了疾病患病期间不能吃的食物。主体是疾病，客体是食物。',
    # 'do_check':'该关系类型描述的是疾病和检查的关系。展示了疾病所需要的检查方式。主体是疾病，客体是检查项目。',
    # 'belong_to':'这个关系类型用来表示疾病，科室的层级关系。展示了疾病所属父类疾病，疾病所属科室，科室所属父类科室等。存在三类关系，分别是：疾病属于的父类疾病，疾病属于的科室，科室属于的父类科室。',
    # 'has_symptom':'该关系类型描述的是疾病和症状的关系。展示了疾病患病时所展现的症状。主体是疾病，客体是症状。',
    'easy_get':'该关系类型描述的是疾病和不同群体的关系，展现了疾病的易患病人群。主体是疾病，其前后由@@确认，如“@肺炎@”可确定主体疾病为“肺炎”，客体是不同群体名称，如老人，男性，女性，青少年等。',
    # 'has_prevent':'该关系描述的是疾病和预防措施的关系，展现了预防疾病所需采取的措施。主体是疾病，其前后由@@确认，如“@肺炎@”可确定主体疾病为“肺炎”，客体是预防措施，预防措施限制字数为10字以下短语。',
    # 'acompany':'该关系类型描述的是疾病间的关系，展现了疾病所可能伴随的并发症。主体是疾病，客体是另一种疾病。',
    # 'has_cause':'该关系描述的是疾病和病因的关系，展现了导致患病的可能原因。主体是疾病，其前后由@@确认，如“@肺炎@”可确定主体疾病为“肺炎”，客体是病因，病因是各种名词或名词短语。',
    }



# task = 'NER'
# language = 'zh'
# schema ={
#     '疾病':'比如：感冒,肺炎,癌症,非典等。',
#     '药品':'如:阿司匹林、莲花清温胶囊等。',
#     '食物':'如:香蕉,韭菜,辣椒等.',
#     '检查项目':'如:胸部CT检查,肺活检,耳、鼻、咽拭子细菌培养,白细胞分类计数等，注意不包括各种生物蛋白分子',
#     '科室':'如:内科，外科，骨科，口腔科等.以及各种子科室，如：消化内科，肾内科等，注意不包括各种等级医院',
#     '症状':'如:恶心,抽搐,感觉障碍等。以及相关的描述的名词',
#     '治疗方法':'如:支气管肺泡灌洗，药物治疗,支持性治疗等。',
#     '药品商':'如：北京同仁堂，白云山医药，东新药业等.',
#     '传播方式':'如:呼吸道传播，性传播，母婴传播等',
#     '病因':'包括各种细菌感染，病毒感染，外伤，不当饮食，器官病变等。例如：肺泡壁破裂,肺炎球菌感染,吸入有一氧化碳等.',
#     '预防措施':'如：控制传染源，切断传播途径，定期产前检查，注意饮食等.',
#     '人群':'该实体类型描述的是疾病的易感染人群。例如:新生儿，青少年，老人，孕妇，女性，男性等.',
#     }
# schema ={
#     '疾病':'该实体类型描述人体病症,比如：感冒,肺炎,癌症,非典等。',
#     '药品':'该实体类型描述的是各种疾病的治疗药物,例如:阿司匹林、莲花清温胶囊等。',
#     '食物':'该实体类型描述的是可食用物品。例如:香蕉,韭菜,辣椒等.',
#     '检查项目':'该实体类型描述的是各种检测疾病或者检测身体状况的检查项目。例如:胸部CT检查,肺活检,耳、鼻、咽拭子细菌培养,白细胞分类计数等',
#     '科室':'该实体类型描述的是各种疾病在医院中所属科室。例如:内科，外科，骨科，口腔科等.以及各种子科室，如：消化内科，肾内科等',
#     '症状':'该实体类型描述的是各种疾病患病时出现的特征。例如:恶心,抽搐,感觉障碍等。以及相关的描述的名词',
#     '治疗方法':'该实体类型描述的是各品种疾病对应的治疗方法。例如:支气管肺泡灌洗，药物治疗,支持性治疗等。',
#     '药品商':'该实体类型描述的是药品的生产商。例如：北京同仁堂，白云山医药，东新药业等.',
#     '传播方式':'该实体类型描述的是疾病的传播方式。例如:呼吸道传播，性传播，母婴传播等',
#     '病因':'该实体类型描述的是疾病发生的原因。包括各种细菌感染，病毒感染，外伤，不当饮食，器官病变等。例如：肺泡壁破裂,肺炎球菌感染,吸入有一氧化碳等.',
#     '预防措施':'该实体类型描述的是个体在患病前可以采取的的避免疾病的措施。例如：控制传染源，切断传播途径，定期产前检查，注意饮食等.',
#     '人群':'该实体类型描述的是疾病的易感染人群。例如:新生儿，青少年，老人，孕妇，女性，男性等.',
#     }


# inputs = '病虫害综合防控技术萝卜主要病害（一）病毒病病毒病是萝卜的主要病害,各地均有分布,发生普遍,夏、秋季发病重.一般病株率 10%左右,危害轻时影响产量,严重发病率30%~50%,对产量和质量都有明显影响.1.症状病株生长不良.心叶表现明脉症,并逐渐形成花叶叶片皱缩,畸形,严重病株出现疱疹状叶.染病萝卜生长缓慢、品质低劣.另一种症状是叶片上出现许多直径2~4毫米的圆形黑斑,茎、花梗上产生黑色条斑.病株受害表现为植株矮化,但很少出现畸形,结荚少且不饱满.2.病原其病原有芜菁花叶病毒(TuMV )、 黄瓜花叶病毒(CMV)和萝卜耳突花叶病毒( REMV ).此病毒寄主范围广,可侵染十字花科、藜科、茄科植物.3.传播途径病毒主要在病株和叶子中越冬,可通过摩擦方式进行计液传播.在周年栽培十字花科蔬菜的地区,病毒能不断地从病株传到健康植株上引起发病.此外,REMV和RMV由黄条跳甲等传毒.TuMV和CMV可由桃蚜、萝卜蚜传毒.4.发生规律萝卜病毒病的发病条件与萝卜的发育阶段、有翅蚜的迁飞活动、气候、品种的抗性和萝卜的邻作等都有一的关系.萝卜苗期植株柔嫩,若遇蚜 虫迁飞高峰或高温干星,则容易引起病毒病的感染和流行,且受害严重.适于病害发生流行的温度为28℃左右,潜育期8~14天.高温干旱对蚜虫的敏殖和活动有利,对萝卜生长发育不利,植株抗病力弱,发病较严重.不同的萝卜品种对病毒的抵抗力差异很大,同一品种的不同个体发病程度也不一致.十字花科蔬菜互为邻作时病毒相互传染,发病重.萝卜与非十字花科蔬菜邻作时发病轻.另外,夏、秋季节不适当的早播也常引起病毒病的流行.5.综合防治(1)农业防治选用抗病品种,一般青皮系统较抗病,要根据茬口和市场需求选用抗病品种.秋茬萝卜干旱年份宜早播.高畦直播时,苗期多浇水,以降低地温.适当晚定苗,选留无病株.与大田作物间套作,可明显减轻病害.苗期用银灰膜或塑料反光膜、铝光纸反光遮蚜.(2)化学防治发病初期喷20%吗胍·乙酸铜可湿性粉剂500倍液,或1.5%烷醇·硫酸铜乳剂1000倍液.每隔10天左右防治1次,连续防治3~4次.在苗期防治蚜虫和跳甲.（二）黑腐病萝卜黑腐病俗称黑心病、烂心病,是萝卜最常见的病害之一.各地均有发生,秋播比春播发病重.生长期和贮藏期均可起黑腐病危害.主要危害萝卜的叶和根,萝卜根内部变黑失去商品性,造成很大损失.1.症状主要危害叶和根.(1)叶片幼苗期发病子叶感病,病原菌从叶缘侵人引起发病,叶初呈黄色菱蔫状,之后逐渐枯死.幼苗发病严重时,可导动幼苗萎蔫、枯死或病情迅速蔓延至真叶.真叶感病时会形成黄褐色坏死斑,病斑具有明显的黄绿色晕边,病健界限不明显,且病斑由叶缘逐渐向内部扩展,呈“V”形,部分叶片发病后向一边扭曲.之后继续向内发展,叶脉变黑呈网纹状,逐渐整叶变黄干枯.病原沿叶脉和维管束向短缩茎根部发展,最后使全株叶片变黄枯死	(2)根萝卜肉质根受侵染,透过日光可看出暗灰色病变.横切看,维管束呈黑褐色放射线状,严重发病时呈干缩的空洞.黑腐病导致维管束溢出菌脓,可与缺硼引起的生理性变黑相区别.另外,留种株发病严重时,叶片枯死,茎上密布病斑,种荚瘦小,种子干瘪.2.病原黑腐病病原为野油菜黄单胞杆菌野油菜黑腐病致病型,属细菌性病害.这种病原可以侵染萝卜、白菜类、甘蓝等多种十字花科蔬菜.3.发生规律初侵染源主要来自以下几个方面.(1)带菌种子萝卜细菌性黑腐病是一种种传病害,种子带菌率为0.03%时就能造成该病害的大规模暴发.在染病的种株上,病菌可从果柄维管束或种脐进人种荚或种皮,使种子带菌.种子是黑腐病的重要初侵染源之一.(2)土壤及病残体在田间,黑腐病菌可以存活于土壤中或土表的植物病残体上,该病原菌在植株病残体上存活时间可达2~3年,而离开植株残体,该细菌在土壤中存活时间不会超过6周,带茎的植物病残体是该病在田间最主要的初侵染源.(3)杂草尤其是一些十字花科杂草是细菌性黑腐病菌的寄主,如芜菁、印度芥菜、黑芥、芥菜、野生萝卜、大蒜芥等,田间及田块周围的带菌的杂草也是该病的初侵染源之一.4.传播途径(1)种子传播从黑腐病侵染循环中可以看出,种子是病害发生的重要初侵梁染源.商品种子的快速流通,使得该病在我国大面积发生.(2)雨水飞溅和滥溉水传播雨季来临时,随着雨水的地夹桥流及雨滴的飞溅,导致该病原菌传播到感病寄主上,从其伤口、气孔及水孔进行侵染:田间灌溉时,灌溉水水滴飞溅将土壤、病残体中的病原菌传播到感病寄主上进行侵染.在潮湿条件下、叶缘形成吐水液滴,病菌聚集在吐水液滴中,水滴飞溅也可导致病原菌传播到相邻植株上.(3)生物媒介传播田间昆虫取食感病植株,可将该病原菌传播至其他作物上导致感病.此外,部分昆虫取食时在作物叶片上造成伤口,为病原菌的侵染也创造了条件.	(4)农事操作传播植株种植过密或生长过旺时进行农事操作,使株间叶片频繁摩擦造成大量伤口,增加了病原菌侵染的机会.农事操作人员在操作后未及时更换鞋子、手套,未对农机具消毒等,使得病原菌从有病株传播到无病株,或传播到另一个田块,使得该病原菌在田间传播蔓延.同时,不恰当的农事操作也会造成该病原菌在田间的进一步传播,如田间病残体及杂草未及时清除,或清除后仍然堆放于田块周围,没及时进行焚烧或深埋等处理,进一步增加了该病原菌传播与侵染的机会.5.流行因素细菌性黑腐病在温暖、潮湿的环境下易暴发流行.温度25℃~30℃、地势低洼、排水不良,尤其是早播、与十字化科作物连作、种植过密、粗放管理、植株徒长、虫害发生严重的田块发病较重.6.综合防治（1）农业防治目前农业防治仍然是细菌性黑腐病防控的主要方式.①使用无菌种子且对种子进行消毒从无病田或无病株上采种.播前对种子进行消毒,用50℃热水浸种25分钟或50%代森锌水剂200倍液浸种15分钟以杀死种子表而携带的多种致病菌.②注意田园清洁发现病株作物或杂草,应立即拔除,并将其深埋或带到到田块外烧毁.	③加强田间管理平整地势,改善田间灌溉系统,与非十字花科作物轮作,避免种植过密,植株徒长,加强田间虫害的防控.(2)综合防治细菌性病害传播很快,短时间内就能在生产田中造成大规模暴发流行.对该病害的防治应以预防为主,在作物发病前或发病初期施药,能较好地控制病害的发生和病原菌的传播.	①生物防治使用生物农药,用3%中生菌素可湿性粉剂600倍液于幼苗2~4叶期进行叶面喷雾,隔3天喷1次,连续喷2~3次.	②化学防治常用防治药剂及方法:用50%噁霉灵可湿性粉剂1200~1500倍液,或70%甲基硫菌灵可湿性粉剂1500倍液,或77%氢氧化铜可湿性粉剂400~500倍液,或20%噻菌铜悬浮剂600~700倍液喷雾或灌根.田间喷药可在一定程度上减慢黑腐病的传播.同时,用药量及用药时间应严格掌握,中午及采收前禁止用药,否则易造成药害.黑腐病发病初期也可喷施50%多菌灵可湿性粉剂1000倍液,隔7天喷1次,连续喷2~3次.感病前可喷施植物抗病诱导剂苯并噻二唑,该药剂离体条件无杀菌活性,但能够诱导一些植物的免疫活性,起到抗病、防病的作用.大田喷施50%苯并噻二唑水分散粒剂,每公顷使用该药剂有效成分不超过35克,隔7天喷1次,连续喷4次,能够减少作物发病.（三）软腐病软腐病又称白腐病,是萝卜的一般性般性病害. 各地都有发生,多在高温时期发生.主要危害根、茎、叶柄和叶片.1.症状软腐病主要危害根茎,叶柄和叶片也会发病.苗期发病,叶基部星水渍状,时柄软化,叶片黄化萎蔫.成熟期发病,叶柄基部水渍状软化,叶片黄化下重.短缩茎发病后向萝卜根发展,引起中心部腐烂,发生恶臭,根部多从根尖开始发病,出现油渍状的褐色软腐,发展后根变软腐烂,继而向上蔓延使心叶呈黑褐色软腐,烂成黏滑的稀泥状.肉质根在贮藏期染病会使部分或整体变成黑褐软腐状.采种株染病后外部形态往往无异常,但髓部完全溃烂变空,仅留肉质根的空壳.植株所有发病部位除表现黏滑烂泥状外,均发出一股难闻的臭味.萝卜得软腐病时维管束不变黑,以此可与黑腐病相区别.2.病原病原为胡萝卜软腐欧文菌胡萝卜软腐致病型.这种病原可以侵染十字花科、茄科、百合科、伞形花科及菊科蔬菜.病原主要在土壤中生存,条件适宜时从伤口侵人进行初侵染和再侵染.3.发生规律病菌主要在留种株、 病残体和土壤里越冬,成为翌年的初侵染源.剪卜软腐病的发病与气候、害虫和栽培条件有一定的关系.该菌发育温度范围为2°C~41℃,适温为25℃ ~ 30℃,50℃条件下经10分钟可将其致死.耐酸碱度范围为pH值53~9.2,适宜pH值7.2.多雨高温天气,病害容易流行.植株体表机械伤、虫伤、自然伤口皆利于病菌的侵入.同时,有的害虫体内外携带病菌,是传播病害的媒介.此外,栽培条件也与病害发生有一定的关系, 如高畦校培比平畦栽培发病轻.凡施用未腐熟的有机肥料,土壤黏重,'
# from docx import Document
# def read_docx(file_path):
#     doc = Document(file_path)
#     text = [paragraph.text.strip().replace("\t", "").replace("\n", "").replace("\xa0", "") for paragraph in doc.paragraphs]
#     return text
def append_to_json_file(data, output_file_path):
    try:
        # 读取现有数据
        with open(output_file_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = []

    # 将新数据追加到现有数据中
    existing_data.append(data)

    # 写入更新后的数据
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

# 函数：使用模型生成输出
def generate_output(texts):
    sintructs = get_instruction(language, task, schema, texts)  
    system_prompt = '<<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手.\n<</SYS>>\n\n'
    sintruct = '[INST] ' + system_prompt + sintructs[0] + '[/INST]'
    input_ids = tokenizer.encode(sintruct, return_tensors='pt').to(device)
    input_length = input_ids.size(1)
    generation_output = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_length=1024, max_new_tokens=512, return_dict_in_generate=True), pad_token_id=tokenizer.eos_token_id)
    generation_output = generation_output.sequences[0]
    generation_output = generation_output[input_length:]
    output = tokenizer.decode(generation_output, skip_special_tokens=True)
    print(output)
    return output

# 函数：处理文件夹中所有.docx文件
# def process_folder(input_folder, output_folder):
#     for filename in os.listdir(input_folder):
#         if filename.endswith('.docx'):
#             file_path = os.path.join(input_folder, filename)
#             texts = read_docx(file_path)
#             results = generate_output(texts)
#             output_file_path = os.path.join(output_folder, filename.replace('.docx', '_results.json'))
#             with open(output_file_path, 'w', encoding='utf-8') as f:
#                 json.dump(results, f, ensure_ascii=False, indent=4)
#             print(f'Results saved to {output_file_path}')

def json_load():
    # 使用with open()打开名为"data.json"的文件并以utf-8编码读取其内容
    file = 'data/raw_data/old_medical.json'
    with open(file, 'r', encoding='utf-8') as fh:
        for line in fh:
            yield json.loads(line.strip())
def process_json():
    text = []
    for item in json_load():
        name, cause, prevent, easy_get = '', '', '', ''
        if 'name' in item:
            name = item['name']
        # if 'cause' in item:
        #     cause = item['cause']
        # if 'prevent' in item:
        #     prevent = item['prevent']
        if 'easy_get' in item:
            easy_get = item['easy_get']
        text.append("@"+name+"@的易患病人群有：" + easy_get)
        results = generate_output(text)
        output_file_path = os.path.join(output_folder, f'result.json')
        append_to_json_file(results, output_file_path)
        text = []
        torch.cuda.empty_cache()

# NER
# input_folder = './example/llm/input'
# output_folder = './example/llm/output'
# process_folder(input_folder, output_folder)

# RE
# output文件夹：output_easy_get，output_cause，output_prevent,output_RE
input_folder = 'data/raw_data'
output_folder = 'data/raw_data'
# process_folder(input_folder, output_folder)
process_json()