from bidict import bidict
from collections import Counter
import pandas as pd
import numpy as np
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
a=[1,4,2,3,5,1,8,1,2,3,4,3,2,4,4,8,2,2,5,4]# 1 2 3 4 5 8
if __name__ == '__main__':
    '''for label in categories:
        print(label)
        print(categories[label])'''

    frequency = {i: Counter() for i in categories.inv}
    frequency[1].update(set([1,2,3,5,7,1,2,3,4,4,123132]))
    frequency[1].update(set([123132,4,4]))
    frequency[2].update(set([45,45,54,54]))
    df_count = pd.DataFrame(frequency, dtype='int').T
    df_count.fillna(0, inplace=True)
    df_count.loc['idf'] = df_count.apply(lambda x: x.sum()) #求和
    print(df_count.loc['idf'])
    idf = df_count.ix['idf'].apply(lambda x: np.log(10 / x))  # 计算所有词的idf
    idf = pd.DataFrame(idf, copy=True)
    #print(df_count.loc['idf'])
    print(idf)
    frequency = {i: [Counter() for j in range(5)] for i in categories.inv}
    print(frequency)
    fre={}
    for label in categories.inv:
        fre[label] = sum(frequency[label], Counter())
    print(fre)
    #print(idf)
    #print(df_count)
    #print(Counter(a))
    #print([0]*11)
