# FAKENEWS-Classifier
cqu,2020fall, big data

### 1. 遇到的问题

#### 0x1、python下nltk包

下载了nltk并不能直接进行调用。因为该资源被国内墙了，所以我们要下载nltk中的自然语言处理数据包，就只能另外单独下载。我们去github下载，并修改对应路径名。

> 1. 使用punkt要把压缩包解压
>
> 2. 使用stopwords应采用以下命令：
>
>    ```python
>    from nltk.corpus import stopwords
>    stop = stopwords.words('english')
>    ```



#### 0x2、spark上部署数据集后运行代码报错

暂未解决。



#### 0x3、pytorch使用上的问题

##### 1. 数据类型的转换

我们的项目是对文字进行矩阵化再进行训练，在从numpy类型转换成torch类型的时候，与平常的方法不一样。

#### 0x4、对Spark中RDD数据类型及方法的理解

##### 1. 对RDDFilter的使用

其规则与python的filter是相反的。python的filter是true就选入，而RDD的filter是false才选入。这一点不搞清楚MapReduce的结果都有问题！

##### 2. 使用textfile读取时遇到问题

原因就是读出来是rdd类型，建议对其使用.collect()方法转换成列表，之后就可以用我们熟悉的python语法进行调用

#### 
