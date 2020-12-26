import nltk
import pandas as pd
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

stop = stopwords.words('english')
true = pd.read_csv("archive/True.csv")
false = pd.read_csv("archive/Fake.csv")

SAVAPATH = '/home/hadoop/project/result'
def basic_clean(text):
  """
  对数据做最基本的清洗，包括用正则表达式、nltk的停用词
  这一部分与annModel中的类似
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  words = re.sub(r'[^\w\s]', '', text).split()
  for t in words:
      if len(t) < 4:
          words.remove(t)
  return [wnl.lemmatize(word) for word in words if word not in stopwords]

def tokenizeandstopwords(text):
    tokens = nltk.word_tokenize(text)
    token_words = [w for w in tokens if w.isalpha()]
    meaningful_words = [w for w in token_words if not w in stop]
    joined_words = ( " ".join(meaningful_words))
    return joined_words

def generate_word_cloud(text, tf):
    wordcloud = WordCloud(
        width = 3000,
        height = 2000,
        background_color = 'black').generate(str(text))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('visualize_{}.png'.format(tf))

'''
由于数据集中的数据内容有多种分类，我们根据不同的分类分析其对应的分类词
'''

'''
真新闻中的政治新闻和世界新闻常用词
'''
# politics = true[true['subject']=="politicsNews"]
# politics['text'] = politics['text'].apply(tokenizeandstopwords)
# politics_text = basic_clean(str(politics.text.values))
#
# generate_word_cloud(politics_text, "politics_text")

# worldnews = true[true['subject']=="worldnews"]
# worldnews['text'] = worldnews['text'].apply(tokenizeandstopwords)
# worldnews_text = basic_clean(str(worldnews.text.values))
# generate_word_cloud(worldnews_text, "world_news_true")

# politics = false[false['subject']=="politics"]
# politics['text'] = politics['text'].apply(tokenizeandstopwords)
# politics_text = basic_clean(str(politics.text.values))
# generate_word_cloud(politics_text, "politics_text_fake")

worldnews = false[false['subject']=="News"]
worldnews['text'] = worldnews['text'].apply(tokenizeandstopwords)
worldnews_text = basic_clean(str(worldnews.text.values))
generate_word_cloud(worldnews_text, "world_news_takes")

