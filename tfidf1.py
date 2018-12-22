from bidict import bidict
from os.path import isfile, join, isdir
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import time

categories = bidict({'科技':1,'体育':2,'军事':3,'娱乐':4,'文化':5,'汽车':6,'能源':7,'房产':8,'健康':9,'金融':10})
def loaddata(mode):
    if mode=='train':
        path='D:\\datamining\\news1\\train\\'
    if mode=='test':
        path='D:\\datamining\\news1\\test\\'
    for label in categories:
        with open(join(path, label + '.txt'), encoding='utf8') as file:
            for line in file:
                yield categories[label],line.strip().split()
def calculate_idf(save=False):
    data=loaddata('train')
    contianer = {i: Counter() for i in categories.inv}
    nums=[0]*11
    for label,words in tqdm(data):
        contianer[label].update(set(words))
        nums[label]+=1
    df_count=pd.DataFrame(contianer,dtype='int').T
    df_count.fillna(0,inplace=True)
    count=sum(nums)
    df_count.loc['idf']=df_count.apply(lambda x:x.sum())
    idf=df_count.ix['idf'].apply(lambda x:np.log(count/x))
    idf=pd.DataFrame(idf,copy=True)
    if save:
        print('Save idf.csv file.', flush=True)
        idf.to_csv('D:\\datamining\\news1\\temp/idf{}.csv'.format(count))
    return idf
'''
def calculate_tf():
    data=loaddata('train')
    container={i:[Counter() for j in range(10)] for i in categories.inv}
    nums=[0]*11
    for label,words in tqdm(data):
        #对每一类中每一万的文本进行统计词出现的次数
        container[label][nums[label]//10000].update(set(words))
        nums[label]+=1
    fre={}
    for label in categories.inv:
        fre[label]=sum(container[label],Counter())
    df_count=pd.DataFrame(fre,dtype='int')
'''
def calculate_tfidf(idf,mode):
    '''
    计算的是所有的每一条文档中词的tf-idf
    :param idf:
    :param mode:
    :return:
    '''
    data=loaddata('train')
    wordsInidf=set(idf.index)
    for label,words in data:
        #统计每一行出现的词频
        cwords=Counter(words)
        words={}
        for word in cwords:
            if word in wordsInidf:
                words[word]=cwords[word]*idf['idf'][word]
        yield label,words


if __name__ == '__main__':
    calculate_idf(True)

