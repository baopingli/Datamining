from bidict import bidict
from tqdm import tqdm
from collections import Counter
import pandas as pd
import numpy as np
from scipy.stats import chisquare
from os.path import isfile, join, isdir
categories = bidict({'科技':1,'体育':2,'军事':3,'娱乐':4,'文化':5,'汽车':6,'能源':7,'房产':8,'健康':9,'金融':10})
def loadData(train = True, count = 50000):
    if train:
        path = 'D:\\datamining\\news1\\train\\'
    else:
        path = 'D:\\datamining\\news1\\test\\'
    for label in categories:
        c = 0
        with open(join(path, label+'.txt'), encoding='utf8') as file:
            for line in file:
                # 返回标签和词汇列表 yeild类似于return返回了一个生成器 返回的形式是 数字+内容，用的yelid比较高级，返回生成器
                #返回的是数字1-10，和所有的对应的每一行分开的词strip去掉首尾空格split就是分开
                yield  categories[label], line.strip().split()
                c += 1
                if c == count:
                    break
def statisticsidf(count = 50000, save=False):
    '''
    计算idf，然后保存
    :param count:
    :param save:
    :return:
    '''
    data = loadData(train=True, count=count)
    # 字典，key是label，value是一个列表，元素是Counter，每1w数据存到一个Counter里，避免字典哈希表（映射） 过大影响性能 应该不是每1w而是每5w存
    frequency = {i:Counter() for i in categories.inv} #i就是1-10 每一个类给一个counter()然后去计算出现的词的数量
    num_of_labels = [0]*11 #构建一个11个0的数组，为什么构建11个？
    for label, words in tqdm(data): #tqdm可以传入任何数组 label是1-10
        frequency[label].update(set(words)) #开始计算出现，这样的话counter变得越来越大，将每一类5万的数据出现的词都统计了出来，每个词出现的次数
        num_of_labels[label] += 1
    # 转换成DataFrame
    df_count = pd.DataFrame(frequency, dtype='int').T#所以说这个df-count就是大约30万维x10的表
    df_count.fillna(0, inplace=True)#用0去补充缺失的值
    # 计算idf
    count = sum(num_of_labels)#所有文档的总数 50w
    df_count.loc['idf'] = df_count.apply(lambda x: x.sum())#这个词在所有文档中出现的总次数
    idf = df_count.ix['idf'].apply(lambda x: np.log(count/x))#计算所有词的idf
    idf = pd.DataFrame(idf, copy=True)
    if save:
        print('Save idf.csv file.', flush=True)
        idf.to_csv('D:\\datamining\\news1\\temp/idf{}.csv'.format(count))
    return idf
def extract_words_with_tfidf(idf, count=50000, train=True):
    '''
    计算tf-idf
    :param idf:
    :param count:
    :param train:
    :return:
    '''
    data = loadData(train=train, count=count)
    wordsInIdf = set(idf.index)
    for label, words in data:
        cwords = Counter(words)
        words = {}
        for word in cwords:
            if  word in wordsInIdf:
                words[word] = cwords[word] * idf['idf'][word]
        yield  label, words
def count_frequency(count = 50000):
    '''
    计算tf
    :param count:
    :return:
    '''
    data = loadData(train=True, count=count)
    frequency = {i:[Counter() for j in range(5)] for i in categories.inv}
    num_of_labels = [0]*11
    for label, words in tqdm(data):
        frequency[label][num_of_labels[label]//10000].update(set(words))
        num_of_labels[label] += 1
    # 汇总每类的Counter
    fre = {}
    for label in categories.inv:
        fre[label] = sum(frequency[label], Counter())
    # 转换成DataFrame
    df_count = pd.DataFrame(fre, dtype='int').T
    df_count.fillna(0, inplace=True)
    df_count.index.name = 'label'
    return df_count, num_of_labels

def chisquare_scipy(df_count:pd.DataFrame,save=False, sizeOfBOW = 5000):
    '''
    卡方运算然后排序然后存储bag
    :param df_count:
    :param save:
    :param sizeOfBOW:
    :return:
    '''
    df_count.loc['chisquare'], p = chisquare(df_count)
    sorted_chi = df_count.ix['chisquare'].sort_values(ascending=False)
    #排序后选择前4000个词
    bag = list(sorted_chi.index[:sizeOfBOW])
    with open('bag{}tfidf.txt'.format(sizeOfBOW), 'w', encoding='utf8') as file:
        for w in bag:
            file.write(w+'\n')
    return bag

def statistic(count=50000):
    '''
    获得所有词的在文本中的tfidf的求和，为后面进行卡方检验
    :param count:
    :return:
    '''
    print('Statistics idf.')
    idf = statisticsidf(count=count)#计算idf
    #idf = pd.read_csv('D:\\datamining\\news1\\temp\\idf500000.csv', index_col=0)
    #计算每一条文本中的tfidf
    data = extract_words_with_tfidf(idf, count=count)
    frequency = {i:[Counter() for j in range(5)] for i in categories.inv}
    num_of_labels = [0]*11
    for label, words in tqdm(data):
        counter_index = num_of_labels[label]//10000
        frequency[label][counter_index].update(words)
        num_of_labels[label] += 1
    # 汇总每类的Counter
    fre = {}
    for label in categories.inv:
        fre[label] = sum(frequency[label], Counter())
    df_count = pd.DataFrame(fre, dtype='int').T
    df_count.fillna(0, inplace=True)
    df_count.index.name = 'label'
    print('Count: ')
    print(df_count)
    print(num_of_labels)
    return df_count, num_of_labels, idf

def pre_treat(count=50000, sizeOfBOW = 50000):
    df_count, nums, idf = statistic(count=count)
    bag = chisquare_scipy(df_count, sizeOfBOW=sizeOfBOW)
    df_count, nums = count_frequency(count=count)
    return df_count, bag, nums, []

if __name__ == '__main__':
    #pre_treat(50000,4000)
    statistic(50000)