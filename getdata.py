import time
import requests
from selenium import webdriver
from PIL import Image
from io import BytesIO
# 我们想要浏览的 URL 链接
url = "https://unsplash.com"
# 使用 Selenium 的 webdriver 来打开这个页面
driver = webdriver.Firefox(executable_path=r'geckodriver.exe')
driver.get(url)
driver.execute_script("window.scrollTo(0,1000);")
time.sleep(5)
#选择图片然后打印URL
image_elements=driver.find_element_by_css_selector("#gridMulti img")
i=0
for image_element in image_elements:
    image_url=image_element.get_attribute("src")
    print(image_url)


