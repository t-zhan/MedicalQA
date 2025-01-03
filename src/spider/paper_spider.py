from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
from tqdm import tqdm
import json
import os


chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-gpu')
# chrome_options.add_argument("user-agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'")
browser = webdriver.Chrome(options=chrome_options)
browser2 = webdriver.Chrome(options=chrome_options)

browser.get('https://www.yiigle.com/Paper/Search?type=&q=指南')
sleep(1)

num_pages = 50  # Number of pages to crawl

for i in tqdm(range(num_pages)):
    year_box = browser.find_element(By.XPATH, './/input[@type="text" and @autocomplete="off" and @maxlength="4" and @placeholder="  年" and contains(@class, "el-input__inner")]')
    year_box = browser.find_element(By.CLASS_NAME, value='el-input__inner')

    
    links = browser.find_elements(By.XPATH, './/a[@class="el-link el-link--default" and @href]')
    links = set(link.get_attribute('href') for link in links if link.get_attribute('href') is not None and 'index' not in link.get_attribute('href'))

    print('Loading existing json...')
    file_path = 'data/raw_data/instructions.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    print('Starting to scrape...')
    for link in tqdm(links):
        browser2.get(link)
        sleep(1)
        title = browser2.find_element(By.CLASS_NAME, 'art_title')
        sleep(1)
        content = browser2.find_element(By.CLASS_NAME, 'body_content')
        sleep(1)
        entry = {"title": title.text, "content": content.text}
        existing_data.append(entry)

    # Write updated data back to file
    with open('data/raw_data/instructions.json', 'w') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    print('Data saved to file')

    next_page = browser.find_element(By.CLASS_NAME, 'btn-next')
    next_page.click()
    sleep(1)
