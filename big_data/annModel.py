import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import punkt
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def get_subject(data):
    data["subject"].value_counts()

def pure_dataset(data):
    '''
    数据清洗
    :param data:
    :return:
    '''
    new_data = []
    pattern = "[^a-zA-Z]"
    lemma = nltk.WordNetLemmatizer()
    for txt in data:
        txt = re.sub(pattern, " ", txt)
        txt = txt.lower() #大小写不做区分，同一个单词
        txt = nltk.word_tokenize(txt)
        txt = [lemma.lemmatize(word) for word in txt]
        for t in txt:
            if len(t) < 4:
                txt.remove(t)
        txt = " ".join(txt)
        new_data.append(txt)
    return new_data

# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        ### four convolution layer
        self.mlp1 = torch.nn.Linear(2*2*64,100)
        self.mlp2 = torch.nn.Linear(100,10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))
        x = self.mlp2(x)
        return x


class news_classification(nn.Module):
    def __init__(self):
        super(news_classification, self).__init__()
        self.linear1 = nn.Linear(5008, 2000)
        self.relu1 = nn.ReLU() #激活函数

        self.linear2 = nn.Linear(2000, 500)
        self.relu2 = nn.ReLU()

        self.linear3 = nn.Linear(500, 100)
        self.relu3 = nn.ReLU()

        self.linear4 = nn.Linear(100, 20)
        self.relu4 = nn.ReLU()

        self.linear5 = nn.Linear(20, 2)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.relu3(out)
        out = self.linear4(out)
        out = self.relu4(out)
        out = self.linear5(out)
        return out

'''
建立新闻分类模型
'''
model = news_classification()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
error = nn.CrossEntropyLoss()


'''
data set from kaggle, true & false
'''
true_data = pd.read_csv("archive/True.csv")
fake_data = pd.read_csv("archive/Fake.csv")

'''
为数据集添加标签
'''
true_data["label"] = np.ones(len(true_data),dtype=int)
fake_data["label"] = np.zeros(len(fake_data), dtype=int)

'''
数据合并
'''
data = pd.concat((true_data,fake_data),axis=0)
data = data.sample(frac=1)
data = pd.get_dummies(data,columns=["subject"]) #做成独热的形式
data = data.drop("date",axis=1)

print("--------start puring data--------")
new_text = pure_dataset(data.text)
new_title = pure_dataset(data.title)
print("--------data pured!--------")

'''
将数据转化为矩阵的形式
并
'''
print("--------vetorizing data--------")
vectorizer_title = CountVectorizer(stop_words="english",max_features=1000)
vectorizer_text = CountVectorizer(stop_words="english",max_features=4000)

title_matrix = vectorizer_title.fit_transform(new_title).toarray()
text_matrix = vectorizer_text.fit_transform(new_text).toarray()

data.drop(["title","text"],axis=1,inplace=True)

y = data.label
x = np.concatenate((np.array(data.drop("label",axis=1)),title_matrix,text_matrix),axis=1)

X_train,X_test,Y_train,Y_test = train_test_split(x,np.array(y),test_size=0.25,random_state=1)
# train test分别为训练集和检测集
print("--------train set split finished!--------")

'''
模型训练
'''
X_train = torch.tensor(X_train)

Y_train = torch.LongTensor(Y_train)

X_test = torch.from_numpy(X_test)
Y_test = torch.from_numpy(Y_test)


epoch = 20 #迭代20次
print("--------start training--------")
for e in range(epoch):
    optimizer.zero_grad() #清空梯度
    fout = model(X_train) #foward prop
    loss = error(fout, Y_train) #evaluate loss
    loss.backward() #backward prop
    optimizer.step() #update param
    print("epoch {}: loss {}".format(e, loss))

    # prediction and test
    y_head = model(X_test)
    y_pred = torch.max(y_head, 1)[1]
    print("rate of good prediction: ", accuracy_score(y_pred,Y_test))

print("--------end of training--------")

'''
使用外来的数据进行测试， test set
'''