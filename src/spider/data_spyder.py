# 爬取寻医问药网站的数据
import urllib.request
from urllib.parse import urlparse
from lxml import etree
import re
import time
import random
from tqdm import tqdm
import os
import json


class CrimeSpider:
    def __init__(self, output_file='data/raw_data/old_medical.json'):
        self.output_file = output_file
        self.data = []  # 用于存储抓取的数据
        
    def get_html(self, url):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) '
                                        'Chrome/51.0.2704.63 Safari/537.36'}
        req = urllib.request.Request(url=url, headers=headers)
        res = urllib.request.urlopen(req)
        html = res.read().decode('gbk')
        return html
    
    def url_parser(self, content):
        selector = etree.HTML(content) 
        urls = ['http://www.anliguan.com' + i for i in  selector.xpath('//h2[@class="item-title"]/a/@href')]
        return urls
    
    def spider_main(self):
    # all_data = []  # Store all scraped data
        pbar = tqdm(total=10137, desc="抓取进度")  # Initialize the progress bar

        for page in tqdm(range(1, 10138), desc="抓取进度"): 
            try:
                basic_url = 'http://jib.xywy.com/il_sii/gaishu/%s.htm'%page # 简介
                cause_url = 'http://jib.xywy.com/il_sii/cause/%s.htm'%page #病因
                prevent_url = 'http://jib.xywy.com/il_sii/prevent/%s.htm'%page #预防
                # neopathy_url = 'http://jib.xywy.com/il_sii/neopathy/%s.htm'%page #（新增）并发症
                symptom_url = 'http://jib.xywy.com/il_sii/symptom/%s.htm'%page # 症状
                inspect_url = 'http://jib.xywy.com/il_sii/inspect/%s.htm'%page # 检查
                # diagnosis_url = 'http://jib.xywy.com/il_sii/diagnosis/%s.htm'%page #（新增）诊断鉴别
                treat_url = 'http://jib.xywy.com/il_sii/treat/%s.htm'%page # 治疗
                food_url = 'http://jib.xywy.com/il_sii/food/%s.htm'%page # 饮食保健
                # nursing_url = 'http://jib.xywy.com/il_sii/nursing/%s.htm'%page # (修改)drug——>nursing，护理
                    
                data = {}
                data['url'] = basic_url
                data['basic_info'] = self.basicinfo_spider(basic_url)
                data['cause_info'] =  self.common_spider(cause_url)
                data['prevent_info'] =  self.common_spider(prevent_url)
                data['symptom_info'] = self.symptom_spider(symptom_url)
                data['inspect_info'] = self.inspect_spider(inspect_url)
                data['treat_info'] = self.treat_spider(treat_url)
                data['food_info'] = self.food_spider(food_url)
                # data['drug_info'] = self.drug_spider(drug_url)
                # print(page, basic_url)
                self.data.append(data)
                # all_data.append(data)
                pbar.update(1)  # Update the progress bar

            except Exception as e:
                pbar.set_description(f"错误: {e}, 页面: {page}, URL: {basic_url}")
                
        self.save_to_file()
        pbar.close()  # Close the progress bar
        return 
    
    def save_to_file(self):
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)  # 确保目录存在
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
            
    def basicinfo_spider(self, url):
        html = self.get_html(url)
        selector = etree.HTML(html)
        title = selector.xpath('//title/text()')[0]
        category = selector.xpath('//div[@class="wrap mt10 nav-bar"]/a/text()')
        desc = selector.xpath('//div[@class="jib-articl-con jib-lh-articl"]/p/text()')
        ps = selector.xpath('//div[@class="mt20 articl-know"]/p')
        infobox = []
        for p in ps:
            info = p.xpath('string(.)').replace('\r','').replace('\n','').replace('\xa0', '').replace('   ', '').replace('\t','')
            infobox.append(info)
        basic_data = {}
        basic_data['category'] = category
        basic_data['name'] = title.split('的简介')[0]
        basic_data['desc'] = desc
        basic_data['attributes'] = infobox
        # print("基本信息：",basic_data)
        return basic_data
    
    def treat_spider(self, url):
        html = self.get_html(url)
        selector = etree.HTML(html)
        ps = selector.xpath('//div[starts-with(@class,"mt20 articl-know")]/p')
        infobox = []
        for p in ps:
            info = p.xpath('string(.)').replace('\r','').replace('\n','').replace('\xa0', '').replace('   ', '').replace('\t','')
            infobox.append(info)
        return infobox  
    
    def food_spider(self, url):
        html = self.get_html(url)
        selector = etree.HTML(html)
        divs = selector.xpath('//div[@class="diet-img clearfix mt20"]')
        try:
            food_data = {}
            food_data['good'] = divs[0].xpath('./div/p/text()')
            food_data['bad'] = divs[1].xpath('./div/p/text()')
            food_data['recommand'] = divs[2].xpath('./div/p/text()')
        except:
            return {}

        return food_data
    
    def symptom_spider(self, url):
        html = self.get_html(url)
        selector = etree.HTML(html)
        symptoms = selector.xpath('//a[@class="gre" ]/text()')
        ps = selector.xpath('//p')
        detail = []
        for p in ps:
            info = p.xpath('string(.)').replace('\r','').replace('\n','').replace('\xa0', '').replace('   ', '').replace('\t','')
            detail.append(info)
        symptoms_data = {}
        symptoms_data['symptoms'] = symptoms
        symptoms_data['symptoms_detail'] = detail
        return symptoms, detail
    
    def inspect_spider(self, url):
        html = self.get_html(url)
        selector = etree.HTML(html)
        inspects  = selector.xpath('//li[@class="check-item"]/a/@href')
        return inspects
    
    def common_spider(self, url):
        html = self.get_html(url)
        selector = etree.HTML(html)
        ps = selector.xpath('//p')
        infobox = []
        for p in ps:
            info = p.xpath('string(.)').replace('\r', '').replace('\n', '').replace('\xa0', '').replace('   ','').replace('\t', '')
            if info:
                infobox.append(info)
        return '\n'.join(infobox)
    
    def inspect_crawl(self):
        for page in tqdm(range(1, 10138), desc="检查项抓取进度"):
            try:
                url = 'http://jck.xywy.com/jc_%s.html'%page
                html = self.get_html(url)
                data = {}
                data['url']= url
                data['html'] = html
                self.db['jc'].insert_one(data)
                print(url)
            except Exception as e:
                print(e)
                
    

handler = CrimeSpider()
handler.spider_main()