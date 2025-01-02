# 提取所有页面的疾病链接
import requests
from bs4 import BeautifulSoup
import time
import csv
from tqdm import tqdm 


# 基础URL和对应的页数
base_urls = {
    'https://jbk.39.net/bw/neike_p{}/': 100,  # 内科网页，有100页
    'https://jbk.39.net/bw/waike_p{}/': 100,  # 外科网页，有100页
    'https://jbk.39.net/bw/erke_p{}/': 90,  # 儿科网页，90页
    'https://jbk.39.net/bw/fuchanke_p{}/': 86,  # 妇产科，86页
    'https://jbk.39.net/bw/nanke_p{}/': 21,  # 男科，21页
    'https://jbk.39.net/bw/wuguanke_p{}/': 100,  # 五官科，100页
    'https://jbk.39.net/bw/pifuxingbing_p{}/': 92,  # 皮肤病，92页
    'https://jbk.39.net/bw/shengzhijiankang_p{}/': 13,  # 生殖健康，13页
    'https://jbk.39.net/bw/zhongxiyijieheke_p{}/': 97,  # 中西医结合，97页
    'https://jbk.39.net/bw/ganbing_p{}/': 14,  # 肝病，14页
    'https://jbk.39.net/bw/jingshenxinlike_p{}/': 39,  # 精神科，39页
    'https://jbk.39.net/bw/zhongliuke_p{}/': 67,  # 肿瘤科，67页
    'https://jbk.39.net/bw/chuanranke_p{}/': 43,  # 传染科，43页
    'https://jbk.39.net/bw/laonianke_p{}/': 18,  # 老年科，18页
    'https://jbk.39.net/bw/tijianbaojianke_p{}/': 70,  # 体检保健，70页
    # 'https://jbk.39.net/bw/chengyinyixueke/': 1, # 成瘾科，1
    'https://jbk.39.net/bw/heyixueke_p{}/': 3,  # 核医学科，3页
    'https://jbk.39.net/bw/jizhenke_p{}/': 40,  # 急诊科，40页
    'https://jbk.39.net/bw/yingyangke_p{}/': 14,  # 营养科，14页
}

# 请求头
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# 函数：提取每页中的疾病链接
def get_disease_links(url, page):
    full_url = url.format(page)
    try:
        response = requests.get(full_url, headers=headers)
        response.raise_for_status()  # 检查是否成功获取页面
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch page {page}, error: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找包含疾病链接的 <a> 标签
    disease_links = []
    for item in soup.select('div.result_item p.result_item_top_l a'):
        link = item.get('href')  # 获取 href 属性
        title = item.get('title')  # 获取 title 属性，即疾病名称
        if link and link.startswith('http'):  # 确保是完整的链接
            disease_links.append((title, link))

    return disease_links

# 存储所有提取的链接
all_disease_links = []

# 计算总页数
total_pages = sum(base_urls.values())

# 创建进度条
with tqdm(total=total_pages, desc='Total Progress') as pbar:
    # 遍历每个基础URL及其对应的页数
    for base_url, total_pages_for_url in base_urls.items():
        for page in range(1, total_pages_for_url + 1):
            links = get_disease_links(base_url, page)
            all_disease_links.extend(links)  # 将提取的链接添加到变量中
            
            # 更新进度条
            pbar.update(1)
            
            # 请求间隔，防止请求过于频繁
            time.sleep(1)  # 等待1秒

# 最终的疾病链接存储在 all_disease_links 变量中
print(f"Total diseases extracted: {len(all_disease_links)}")  # 显示提取的总数

# 打印前几项看看结果
for title, link in all_disease_links[:100]:  # 仅输出前10个
    print(f"{title}: {link}")

# 保存到 CSV 文件
with open('/remote-home/share/guokaiqian/disease_links.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Title', 'Link'])  # 写入表头
    writer.writerows(all_disease_links)  # 写入所有链接



# 最终的疾病链接存储在 all_disease_links 变量中
print(f"Total diseases extracted: {len(all_disease_links)}")  # 显示提取的总数

# #### 链接去重

# 使用字典保持顺序并去重
unique_links = list(dict.fromkeys(all_disease_links))

# 输出结果
print("Unique links:")
for title, link in unique_links:
    print(f"{title}: {link}")

# 如果你想要一个集合，可以这样做
# unique_links_set = set(all_disease_links)
# print(f"Total unique links (set): {len(unique_links_set)}")

print(f"Total diseases extracted: {len(unique_links)}")


with open('/remote-home/share/guokaiqian/unique_disease_links.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Title', 'Link'])  # 写入表头
    writer.writerows(unique_links)  # 写入所有链接
