# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from tqdm import tqdm_notebook as tqdm
import re
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'IPAexGothic'

from scipy.cluster.hierarchy import linkage, dendrogram

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel, CoherenceModel, TfidfModel

import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()

from wordcloud import WordCloud


# %%
import glob
from bs4 import BeautifulSoup

files = glob.glob('./work/raw/*html')

def parse(fileName):
    with open(fileName) as f:
        soup = BeautifulSoup(f, 'html.parser')

    title      = soup.select_one('article.blog-entry-article h1.blog-title').get_text()
    date       = soup.select_one('article.blog-entry-article div.blog-date').get_text()
    category   = soup.select_one('article.blog-entry-article li.blog-category').get_text()
    text       = soup.select_one('article.blog-entry-article div.content').get_text()
    
    
    return [fileName, title, date, category, text]


data = [parse(fileName) for fileName in tqdm(files)]


# %%
df = pd.DataFrame(data, columns=['file', 'title', 'date', 'category', 'text'])
df.head()


# %%
## 前処理
import MeCab
m = MeCab.Tagger('-d /usr/local/lib/mecab/dic/ipadic')

HINSHI = ['名詞']
STOPWORDS = ['フライウィール', 'flywheel', 'var', 'main', 'test', 'time', 'src', 'com', 'jp', 'れる', 'これ', 'なっ', 'それ', 'もの', 'たち', 'さん']

def parseText(text):
    node = m.parseToNode(text)
    words = []
    while node:
        fields = node.feature.split(",")
        word = node.surface.lower() # 小文字化
        word = re.sub(r'\d+', '0', word) # 数字置き換え        
        word = re.sub(r'[\.\/\(\){}\[\]:,?!;\*=_\-\'"@<>#\^%]+', '', word) # 記号除去
        if fields[0] in HINSHI and word not in STOPWORDS and len(word) > 1:
            words.append(word)
        node = node.next
    
    return words


# %%
df['words'] = df['text'].map(lambda text: parseText(text))
# df['words'] = df['title'].map(lambda text: parseText(text))
df['words'].head()


# %%
## 辞書とコーパスの作成
dictionary = Dictionary(df['words'])
dictionary.filter_extremes(no_below=3, no_above=0.7)
print(len(dictionary))

# BoWコーパス
corpus = [dictionary.doc2bow(words) for words in df['words']]

# tfidfコーパス
tfidf = TfidfModel(corpus)
corpus = tfidf[corpus]


# %%
from collections import defaultdict
from gensim.models.keyedvectors import KeyedVectors
from sklearn.cluster import KMeans


# %%
# 以下から最新の学習済みモデルをダウンロード
# https://github.com/singletongue/WikiEntVec/releases
# 今回利用したのは20190520のjawiki.all_vectors.100d.txt.bz2

model = KeyedVectors.load_word2vec_format('work/jawiki.all_vectors.100d.txt')


# %%
word = '理科'
results = model.wv.most_similar(word)
print(word, "と類似度の高い単語") 
for result in results:
    print(result)


# %%
data = [[word, model.wv[word]] for word in dictionary.values() if word in model.wv]
df = pd.DataFrame(data, columns=['word', 'vectors'])


# %%
df.head()


# %%
df.shape


# %%
distortions = []

for i  in tqdm(range(1,21)):
    km = KMeans(n_clusters=i, verbose=1, random_state=42, n_jobs=-1)
    km.fit(list(df['vectors']))
    distortions.append(km.inertia_)


# %%
plt.figure(figsize=(6, 6))
plt.plot(range(1,21),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
plt.savefig('work/word_kmeans_sse.png')


# %%



# %%
n_clusters = 6
kmeans_model = KMeans(n_clusters=n_clusters, verbose=1, random_state=42, n_jobs=-1)
kmeans_model.fit(list(df['vectors']))


# %%
df['cluster'] = kmeans_model.labels_


# %%
for i in range(n_clusters):
    print('## Label '+str(i))
    print(','.join(df[df['cluster']==i]['word']))
    print()


# %%
### 主成分分析
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(list(df['vectors']))


# %%
color = list(matplotlib.colors.TABLEAU_COLORS.items())[:10]
color = [x[0] for x in color]

feature = pca.transform(list(df['vectors']))
plt.figure(figsize=(6, 6))
for x, y, name, cluster in zip(feature[:, 0], feature[:, 1], df['word'], df['cluster']):
    plt.text(x, y, name, alpha=0.8, size=5, color=color[cluster])
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
plt.title("Principal Component Analysis")
plt.xlabel("The first principal component score")
plt.ylabel("The second principal component score")
plt.show()
plt.savefig('work/word_kmeans.png')


