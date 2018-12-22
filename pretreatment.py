"""
Description:
数据预处理（已经分词）
我在处理的过程中应该注意金融是处理好的
"""

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from collections import Counter, defaultdict
from os import listdir, remove, mkdir
from os.path import isfile, join, isdir
import data
import jieba.posseg as pseg
import random
from bidict import bidict
from numba import jit
from scipy.stats import chisquare

categories = bidict({'科技':1,
                     '体育':2,
                     '军事':3,
                     '娱乐':4,
                     '文化':5,
                     '汽车':6,
                     '能源':7,
                     '房产':8,
                     '健康':9,
                     '金融':10
                     })

flags = ('n', 'nt', 'nl', 'nz','ng')
stop_words_file = 'D:\\datamining\\stop_words_ch.txt'

def loadStopWords(filename):
    """
    载入停用词
    :param filename:
    :return:
    """
    stop_words = set()
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            stop_words.add(line.strip())
    return stop_words

stop_words = loadStopWords(stop_words_file)

def makeTempDir():
    '''
    创建文件夹
    :return:
    '''
    if not isdir('./news/temp/'):
        mkdir('./news/temp/')

def loadData(train = True, count = 50000):
    """
    载入给定数量的数据，返回生成器
    每一类的数量为count
    这里的count表示的是每一类中载入的数量都是5w，但是要把所有的类都载入
    """
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


def statistics_idf(count = 50000, save=False):
    '''
    统计idf
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
    count = sum(num_of_labels)#所有文档的总数
    df_count.loc['idf'] = df_count.apply(lambda x: x.sum())#这个词在所有文档中出现的总次数
    idf = df_count.ix['idf'].apply(lambda x: np.log(count/(x)))#计算所有词的idf
    idf = pd.DataFrame(idf, copy=True)


    if save:
        print('Save idf.csv file.', flush=True)
        #makeTempDir()
        idf.to_csv('D:\\datamining\\news1\\temp/idf{}.csv'.format(count))
    return idf

def loadidf():
    '''
    载入idf
    :return:
    '''
    idf = pd.read_csv('D:\\datamining\\news1\\temp\\idf500000.csv', index_col=0)
    return idf

def extract_words_with_tfidf(idf, count=50000, train=True):
    '''
    根据tfidf提取词
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

def extract_words_tfidf(idf, count=50000, train=True):

    data = loadData(train=train, count=count)
    wordsInIdf = set(idf.index)
    for label, words in data:
        words = Counter(words)
        com = words.keys() & wordsInIdf
        wdf = pd.Series(words)[com]
        widf = idf['idf'][com]
        w = wdf*widf
        yield  label, w.to_dict()

def count_words_in_label(count = 50000):
    '''
    计算每一类中词在这一类中出现的次数的和
    :param count:
    :return:
    '''
    data = loadData(count=count, train=True)
    frequency = {i:[Counter() for j in range(5)] for i in categories.inv}
    num_of_labels = [0]*11
    for label, words in data:
        frequency[label][num_of_labels[label]//10000].update(words)
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

def count_frequency(count = 50000):
    """
    统计训练集各类词频（每个文档出现的词语）
    保存idf信息
    :return:
    """
    data = loadData(train=True, count=count)
    # 字典，key是label，value是一个列表，元素是Counter，每1w数据存到一个Counter里，避免字典哈希表过大影响性能 所以取了5个这样可以优化性能
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

def splitToTrainAndTest(filename):
    """
    将文件打乱顺序（行），前五万当做训练集，后五万当测试集。
    :param filename:
    :return:
    """
    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        with open(filename+'train.txt','w', encoding='utf8') as train:
            traindata = lines[:50000]
            train.writelines(traindata)
        with open(filename+'test.txt','w', encoding='utf8') as test:
            testdata = lines[-50000:]
            test.writelines(testdata)


def constructBOW(df_count:pd.DataFrame, nums, save=False, sizeOfBOW = 50000):
    """
    使用卡方检验构建词袋
    :param df_count: 每类新闻的词频
    :param save:
    :param minfrequency: 词汇的最小频率，低于这个数值的词汇将不计算卡方值
    :param sizeOfBOW:
    :return:
    """
    datacount = sum(nums)
    print('Size of BOW: ', sizeOfBOW)
    chis = {i:{} for i in df_count.index}
    print('Words count:', df_count.shape)
    print('Calculate chi-square...')

    #计算每一类与所有的词之间的卡方值，然后进行排序
    for word in tqdm(df_count):
        word_sum = df_count[word].sum()
        for label in df_count.index:
            A = df_count[word][label]
            B =  word_sum - A
            C = nums[label] - A
            D = datacount - nums[label] - B
            chis[label][word] = (A*D-B*C)**2/(word_sum*(C+D))
    print('Chi-square get.')
    bag = set()
    print('sort words by chisquare.')
    # 给每类新闻的词汇按卡方值排序
    sortedwords = {}
    for i in df_count.index:
        sortedwords[i] = sorted(chis[i], key=lambda x:chis[i][x])

    # 每类选取sizeofbow
    # for i in sortedwords:
    #     bag = bag|set(sortedwords[i][:sizeOfBOW//3])

    # 控制总量数为sizeofBOW
    print('Select words to build BOW. ')
    s = 0
    bags = []
    is_not_end = [True] * 11
    is_not_end[0] = False
    size = 0

    while size < sizeOfBOW:
        news = set()
        for label,value in sortedwords.items():
            if is_not_end[label]:
                if s < len(value):
                    news = news|set(value[s:s+1000])
                else:
                    is_not_end[label] = False
        if len(bags) <= s//10000:
            bags.append(set())
        bags[s//10000] = bags[s//10000] | news
        s += 1000
        size = 0
        for b in bags:
            size += len(b)
        if not any(is_not_end):
            break

    for b in bags:
        bag = b | bag

    bag = list(bag)
    if save:
        with open('bag.txt', 'w', encoding='utf-8') as file:
            file.write('\n'.join(bag))
    print('Bag is ready.')
    return bag

def chisquare_scipy(df_count:pd.DataFrame,save=False, sizeOfBOW = 5000):
    df_count.loc['chisquare'], p = chisquare(df_count)#计算卡方
    sorted_chi = df_count.ix['chisquare'].sort_values(ascending=False)#进行排序
    bag = list(sorted_chi.index[:sizeOfBOW])#选择前4000个
    with open('bag{}tfidf.txt'.format(sizeOfBOW), 'w', encoding='utf8') as file:
        for w in bag:
            file.write(w+'\n')
    return bag


def loadBOW(file):
    words = []
    with open(file,encoding='utf8') as f:
        for line in f:
            words.append(line.strip())
    return words

def statistic(count=50000):
    print('Statistics idf.')
    idf = statistics_idf(count=count)#计算idf
    data = extract_words_with_tfidf(idf, count=count)#计算tfidf
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
    print(num_of_labels)
    return df_count, num_of_labels, idf

def pre_treat(count=50000, sizeOfBOW = 50000):
    # df_count, nums = count_frequency(count=count)
    df_count, nums, idf = statistic(count=count)
    #bag = constructBOW(df_count, nums, save=True, baseSizeOfBOW=baseSizeOfBOW)
    bag = chisquare_scipy(df_count, sizeOfBOW=sizeOfBOW)
    df_count, nums = count_frequency(count=count)
    return df_count, bag, nums, []

def svm_pre_treat(count=50000, sizeOfBOW = 50000):
    df_count, nums, idf = statistic(count=count)
    #bag = constructBOW(df_count, nums, save=True, baseSizeOfBOW=baseSizeOfBOW)
    bag = chisquare_scipy(df_count, sizeOfBOW=sizeOfBOW)
    df_count, nums = count_frequency(count=count)
    return df_count, bag, nums, idf


def cleanData():
    path = 'D:\datamining\\news'
    labels = list(categories.keys())
    for label in tqdm(labels):
        filename = join(path, label + '_clean.txt')#名字是+clean
        with open(filename, 'r', encoding='utf8') as infile:
            with open('1'+str(label)+'_clean.txt', 'w', encoding='utf8') as outfile:
                for line in tqdm(infile):
                    words = pseg.cut(line)
                    words = [w.word for w in words if w.flag in flags and w.word not in stop_words]
                    if len(words) >= 3:#这里代表的长度是每一行词的个数多余3个就写上
                        outfile.write(' '.join(words) + '\n')




if __name__ == '__main__':

    # cleanData()
    # filenames = ['./news/'+label+'_clean.txt' for label in list(categories.keys())] 这里使用的应该也是分词好的txt
    # for file in filenames:
    #     splitToTrainAndTest(file)
    # splitToTrainAndTest('./news/house.txt')
    # idf = statistics_idf(save=True)
    # print(type(idf))
    # print(idf)
    # df_count, nums = count_frequency(count=100)
    #     # chisquare_scipy(df_count)


    # idf = loadidf()
    # data = extract_words_tfidf(idf)
    # for label, words in data:
    #     print(words.to_dict())
    # pass
    #pre_treat(50000, 5000)
    # cleanData()

    #1.分词去停用词取名词
    #cleanData()
    #2.分训练集和数据集
    '''filenames=['D:\\datamining\\news1\\1'+label+'_clean.txt' for label in list(categories.keys())]
    for file in filenames:
        splitToTrainAndTest(file)'''
    #3.计算idf，计算的是这个词在所有的文档数中出现的总的次数
    '''idf = statistics_idf(save=True)
    print(type(idf))
    print(idf)'''
    #4.计算TF
    #df_count, nums = count_frequency(count=50000)

    print(loadidf())







