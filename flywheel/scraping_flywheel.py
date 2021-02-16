# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import re
import time
from tqdm import tqdm_notebook as tqdm
import requests
from bs4 import BeautifulSoup


# %%
### GETリクエスト
def getRequest(url):
    time.sleep(1)
    return requests.get(url)


# %%
urlBase = 'https://www.flywheel.jp/blog/page/'
pageNum = 7


# %%
### blogの一覧からURL取得
blogUrls = []
for i in tqdm(range(pageNum)):
    url = urlBase + str(i+1) + '/'
    response = getRequest(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    aTags = soup.select('li.blog-item a')
    blogUrls += [aTag['href'] for aTag in aTags]


# %%
i = 1
for url in tqdm(blogUrls):
    saveFileName = 'work/raw/'+str(i)+'.html'
    response = getRequest(url)
    with open(saveFileName, 'wb') as saveFile:
        saveFile.write(response.content)
    i+=1


# %%



# %%



# %%



