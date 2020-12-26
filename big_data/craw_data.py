import requests
import bs4
from bs4 import BeautifulSoup
import csv

url = "http://lite.cnn.com/en"
r = requests.get(url)
path ="http://lite.cnn.com"
text_list = []
title_list = []
if (r.status_code == 200):
    r.encoding = r.apparent_encoding
    soup = BeautifulSoup(r.text, "html.parser")
    tag_a = soup.find_all('a') #获取所有的a标签
    print("200")
    for a in range(12):
        if a < 2 and a <= len(tag_a) - 2: #前两条和最后两条并不是正文，不读
            continue
        title = str(tag_a[a]).split('>')[1].split('<')[0]
        text_url = path + tag_a[a].get('href')
        #接下来，读取正文
        r2 = requests.get(text_url)
        r2.encoding = r2.apparent_encoding
        soup2 = BeautifulSoup(r2.text, "html.parser")
        text = ""
        for tr in soup2.find('div').children:
            if (isinstance(tr, bs4.element.Tag)):
                tds = tr('p')
                text += str(tds)
        final = text.replace("<p>", "")
        final = final.replace("</p>","")
        final = final.replace("<a>", "")
        final = final.replace("</a>", "") #去杂

        title_list.append(title)
        text_list.append(final)

        print("finished{}".format(a))

with open('cnn_news.csv', 'a', newline='', encoding='utf-8')as f:
        write = csv.writer(f)
        write.writerows(text_list)
        write.writerows(title_list)