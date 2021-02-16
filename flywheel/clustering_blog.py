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
df.groupby('category').agg(['count'])


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
## トピック数の探索
start = 2
limit = 10
step = 1

coherence_vals = []
perplexity_vals = []

for n_topic in tqdm(range(start, limit, step)):
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topic, random_state=0)
    perplexity_vals.append(np.exp2(-lda_model.log_perplexity(corpus)))
    coherence_model_lda = CoherenceModel(model=lda_model, texts=df['words'], dictionary=dictionary, coherence='c_v')
    coherence_vals.append(coherence_model_lda.get_coherence())


# %%
# evaluation
x = range(start, limit, step)

fig, ax1 = plt.subplots(figsize=(12,5))

# coherence
c1 = 'darkturquoise'
ax1.plot(x, coherence_vals, 'o-', color=c1)
ax1.set_xlabel('Num Topics')
ax1.set_ylabel('Coherence', color=c1); ax1.tick_params('y', colors=c1)

# perplexity
c2 = 'slategray'
ax2 = ax1.twinx()
ax2.plot(x, perplexity_vals, 'o-', color=c2)
ax2.set_ylabel('Perplexity', color=c2); ax2.tick_params('y', colors=c2)

# Vis
ax1.set_xticks(x)
fig.tight_layout()
plt.show()
plt.savefig('work/metrics.png')


# %%
NUM_TOPICS = 4
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=NUM_TOPICS, random_state=0)
lda_model.save('work/lda.model')


# %%
for i in range(lda_model.num_topics):
        print('TOPIC:', i, '__', lda_model.print_topic(i))


# %%
# WordCloud
# 日本語フォントをダウンロードしてwork以下に設置
fig, axs = plt.subplots(ncols=2, nrows=math.ceil(lda_model.num_topics/2), figsize=(16,20))
axs = axs.flatten()

def color_func(word, font_size, position, orientation, random_state, font_path):
    return 'darkturquoise'

for i, t in enumerate(range(lda_model.num_topics)):

    x = dict(lda_model.show_topic(t, 30))
    im = WordCloud(
        background_color='black',
        color_func=color_func,
        max_words=4000,
        width=300, height=300,
        random_state=0,
        font_path='./work/ipaexg.ttf'
    ).generate_from_frequencies(x)
    axs[i].imshow(im.recolor(colormap= 'Paired_r' , random_state=244), alpha=0.98)
    axs[i].axis('off')
    axs[i].set_title('Topic '+str(t))

# vis
plt.tight_layout()
plt.show()

# save as png
plt.savefig('work/wordcloud.png') 


# %%
# Vis PCoA
vis_pcoa = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
vis_pcoa

# save as html
pyLDAvis.save_html(vis_pcoa, 'work/pyldavis_output_pcoa.html')


# %%
data = []
for c, words, fileName, title, category in zip(corpus, df['words'], df['file'], df['title'], df['category']):
    topics = []
    for topic, score in lda_model[c]:
        if (score > 0.7):
            topics.append(str(topic))
    data.append([fileName, title, category, ','.join(topics)])

df_topic = pd.DataFrame(data, columns=['file', 'title', 'category', 'topics'])
df_topic.head()


# %%
for i in range(lda_model.num_topics):
        print('## TOPIC:', i)
        print('\n'.join(df_topic[df_topic['topics'].str.contains(str(i))]['category']))
        print()


# %%



# %%



