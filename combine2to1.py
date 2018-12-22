import random

from bidict import bidict


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
filenames = ['D:\\datamining\\news1\\1' + label + '_clean.txt' for label in list(categories.keys())]
destfile='D:\\datamining\\news1\\lineartest\\'
def createdata(filename):
    with open(filename,'r',encoding='utf8') as f:
        lines=f.readlines()
        random.shuffle(lines)
        with open(filename+'train.txt','w',encoding='utf8') as p:
            traindata=lines[:50000]+lines[-50000:]
            p.writelines(traindata)


if __name__ == '__main__':

    for file in filenames:
        createdata(file)
