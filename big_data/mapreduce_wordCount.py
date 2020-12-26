from pyspark import SparkConf, SparkContext
from nltk.corpus import stopwords
import nltk
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

conf = SparkConf().setAppName("proj").setMaster("spark://master:7077")
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")   # 设置日志级别
spark = SparkSession(sc)

stop = stopwords.words('english') ## 从nltk包里获取的stopwords

true = sc.textFile("archive/True.csv") # csh这两条路径你需要修改
false = sc.textFile("archive/Fake.csv")
SAVEPATH = " " ##csh这里你需要修改

def retrieve_data(data):
    '''
    文件格式：title,text,subject,date
    :param path: 路径
    :return:
    '''
    word_list = []
    for line in data:
        word_list += line[0]
    return word_list

def rdd2dic(resRdd,topK):
    resDic = resRdd.collectAsMap()
    # 截取字典前K个
    K = 0
    wordDicK = {}
    for key, value in resDic.items():
        if K < topK:
            wordDicK[key] = value
            K += 1
        else:
            break
    return wordDicK

def draw_word_cloud(wordDic):
    wc = WordCloud(
                    background_color='black',
                    width=3000,
                    height=2000,)
    wc.generate_from_frequencies(wordDic)
    # 保存结果
    wc.to_file(os.path.join(SAVAPATH, '词云可视化.png'))

def tokenizeandstopwords(text):
    tokens = nltk.word_tokenize(text)
    token_words = [w for w in tokens if w.isalpha()]
    meaningful_words = [w for w in token_words if not w in stop]
    joined_words = ( " ".join(meaningful_words))
    return joined_words


def clean_data(data):
    data = data.collect()
    word_list = retrieve_data(data)
    wordsRdd = sc.parallelize(word_list)
    resRdd = wordsRdd.filter(lambda word: word not in stop) \
                        .filter(lambda word: len(word) < 4) \
                            .map(lambda word: (word, 1)) \
                                .reduceByKey(lambda a, b: a + b) \
                                    .sortBy(ascending=False, numPartitions=None, keyfunc=lambda x: x[1])

    worddick =  rdd2dic(resRdd,60)
    draw_word_cloud(worddick)

if __name__ == '__main__':
    clean_data(true)
    clean_data(false)