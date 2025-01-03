# 爬取疾病百科网站的数据
import urllib.request
import json
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 函数：获取网页 HTML 内容
def get_html(url):
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/51.0.2704.63 Safari/537.36')
    }
    try:
        req = urllib.request.Request(url=url, headers=headers)
        res = urllib.request.urlopen(req)
        html = res.read().decode('utf-8', errors='ignore')  # 使用 utf-8 并忽略非法字符
        return html
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

# 函数：提取疾病信息
def extract_information(html):
    soup = BeautifulSoup(html, 'html.parser')
    
    # 创建一个字典来存储信息
    info = {
        '名字': 'N/A',
        '别名': [],  # 列表
        '发病部位': [],  # 列表
        '挂号科室': [],  # 列表
        '传染性': [],  # 列表
        '治疗方法': [],  # 列表
        '治愈率': [],  # 列表
        '治疗周期': [],  # 列表
        '多发人群': [],  # 列表
        '治疗费用': [],  # 列表
        '典型症状': [],  # 列表
        '常用药品': [],  # 列表
        '基本信息': 'N/A',
        '是否医保': [],  # 列表
        '临床检查': [],  # 列表
        '并发症': []  # 列表
    }
    
    # 提取疾病名字
    disease_name = soup.find('h1').get_text(strip=True)
    info['名字'] = disease_name
    
    # 提取基本信息
    basic_info_tag = soup.find('p', class_='information_l')
    if basic_info_tag:
        basic_info = basic_info_tag.get_text(strip=True)
        basic_info = basic_info.replace('详细', '')  # 去掉"详细"链接的文本
        info['基本信息'] = basic_info
    
    # 找到信息所在的所有 <li> 标签，并过滤有用信息
    li_tags = soup.find_all('li')
    
    for li in li_tags:
        i_tag = li.find('i')
        if i_tag:  # 确保找到 <i> 标签
            key = i_tag.get_text(strip=True).replace('：', '')
            if key in info:  # 仅提取与疾病信息相关的字段
                a_tags = li.find_all('a')  # 提取所有 <a> 标签

                # 确保 info[key] 是列表
                if not isinstance(info[key], list):
                    info[key] = []

                # 将 <a> 标签内容添加到列表
                for a in a_tags:
                    info[key].append(a.get_text(strip=True))

                # 如果没有 <a> 标签，尝试获取 <span> 或直接获取 <li> 的文本
                if not a_tags:
                    value = li.find('span')
                    if value:
                        info[key].append(value.get_text(strip=True))
                    else:
                        # 添加文本内容
                        info[key].append(li.get_text(strip=True).replace(key + '：', '').strip())

    # 确保所有相关字段都转换为字符串，避免后续操作中的错误
    info['发病部位'] = ', '.join(info['发病部位']) if info['发病部位'] else 'N/A'
    info['常用药品'] = ', '.join(info['常用药品']) if info['常用药品'] else 'N/A'
    info['并发症'] = ', '.join(info['并发症']) if info['并发症'] else 'N/A'
    info['临床检查'] = ', '.join(info['临床检查']) if info['临床检查'] else 'N/A'
    info['典型症状'] = ', '.join(info['典型症状']) if info['典型症状'] else 'N/A'
    info['挂号科室'] = ', '.join(info['挂号科室']) if info['挂号科室'] else 'N/A'
    info['别名'] = ', '.join(info['别名']) if info['别名'] else 'N/A'
    info['传染性'] = ', '.join(info['传染性']) if info['传染性'] else 'N/A'
    info['治疗方法'] = ', '.join(info['治疗方法']) if info['治疗方法'] else 'N/A'
    info['治愈率'] = ', '.join(info['治愈率']) if info['治愈率'] else 'N/A'
    info['治疗周期'] = ', '.join(info['治疗周期']) if info['治疗周期'] else 'N/A'
    info['多发人群'] = ', '.join(info['多发人群']) if info['多发人群'] else 'N/A'
    info['治疗费用'] = ', '.join(info['治疗费用']) if info['治疗费用'] else 'N/A'
    info['是否医保'] = ', '.join(info['是否医保']) if info['是否医保'] else 'N/A'

    
    return info


# 从 CSV 文件读取 unique_links
unique_links_df = pd.read_csv('./data/raw_data/unique_disease_links.csv')

# 去除列名中的空格（如果有的话）
unique_links_df.columns = unique_links_df.columns.str.strip()

# 使用正确的列名提取链接
unique_links = list(zip(unique_links_df['Title'], unique_links_df['Link']))

# unique_links =  unique_links_1[:2000]

# disease_info = []


# 遍历 unique_links 并显示总进度条


# 将信息写入 JSON 文件
with open('data/raw_data/jbk39.json.json', 'w', encoding='utf-8') as json_file:
    
    for title, link in tqdm(unique_links, desc='Processing diseases', total=len(unique_links)):
        html_content = get_html(link)
        
        if html_content:
            disease_info_entry = extract_information(html_content)
            disease_info_entry['链接'] = link  # 添加链接信息
            # disease_info.append(disease_info_entry)  # 将字典添加到列表中
        json_file.write(json.dumps(disease_info_entry, ensure_ascii=False, indent=4) + '\n')
            
    # json.dump(disease_info, json_file, ensure_ascii=False, indent=4)

# 最终结果打印
print(f"Processed {len(unique_links)} diseases and saved to JSON file.")

logging.info(f"Processed {len(unique_links)} diseases and saved to JSON file.")