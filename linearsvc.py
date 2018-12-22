import datetime
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from bidict import bidict
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from sklearn .metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score
import statistic_svm
from sklearn.feature_selection import SelectKBest, chi2
floder='D:\\datamining\\news1\\lineartest\\'
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

def getlabel(index):
    return categories[index]
def getdata():
    '''
    x代表的是所有词的数组，y代表的是label
    :return:
    '''
    x=[]
    y=[]
    for temp in os.listdir(floder):
        filepath=os.path.join(floder,temp)
        (name,extension)=os.path.splitext(temp)
        label=getlabel(name)
        with open(filepath,'r',encoding='utf-8') as p:
            data=p.readlines()
            for line in tqdm(data):
                x.append(label)
                y.append(line.rstrip('\n'))
    with open(floder+'label.pickle', 'wb') as handle:
        pickle.dump(x, handle)
    with open(floder+'content.pickle', 'wb') as handle:
        pickle.dump(y, handle)
def linsvc():
    with open(floder+'test_label.pickle', 'rb') as handle:
        data_x=pickle.load(handle)
        #print(data_x)
        print('load label...')
    with open(floder+'test_content.pickle', 'rb') as handle:
        data_y=pickle.load(handle)
        print('load content...')
    X_train, X_test, y_train, y_test = train_test_split(data_y, data_x, test_size=0.5)
    print('训练数据和测试数据分离完毕....')
    print('计算tfidf值...')
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    print('size of 数据字典:'+repr(train_data.shape))
    test_data = tv.transform(X_test)#将测试数据转换为对应维度的
    print('size of 测试集:'+repr(test_data.shape))

  
    #卡方检验
    print('卡方检验选择5000特征词...')
    ch2=SelectKBest(chi2,k=5000)
    ch2=ch2.fit(train_data, y_train)
    train_data=ch2.transform(train_data)
    test_data=ch2.transform(test_data)

    print('start train...')
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(train_data, y_train)
    joblib.dump(clf, "train_model.m")
    print('model saved...')
    print('start predict...')
    y_predict = clf.predict(test_data)
    print(clf.score(test_data, y_test))
    print(clf.coef_)
    conmat=confusion_matrix(y_true=y_test,y_pred=y_predict)
    print(conmat)
    print('Precision: %.3f' % precision_score(y_true=y_test,y_pred=y_predict,average='micro'))
    print('Recall: %.3f' % recall_score(y_true=y_test,y_pred=y_predict,average='micro'))
    print('F1: %.3f' % f1_score(y_true=y_test,y_pred=y_predict,average='micro'))
    displaycon(conmat)
    statistic_svm.analysis(y_test,y_predict)
    statistic_svm.show_confusion_matrix(conmat)


def displaycon(confmat):
    fig,ax=plt.subplots(figsize=(2.5,2.5))
    ax.matshow(confmat,cmap=plt.cm.Blues,alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    plt.show()

def train_model_NB():
    '''
    朴素贝叶斯
    :return:
    '''
    with open(floder+'test_label.pickle', 'rb') as handle:
        data_x=pickle.load(handle)
        #print(data_x)
        print('load label...')
    with open(floder+'test_content.pickle', 'rb') as handle:
        data_y=pickle.load(handle)
        print('load content...')
    X_train, X_test, y_train, y_test = train_test_split(data_y, data_x, test_size=0.5)
    tv=TfidfVectorizer()
    train_data=tv.fit_transform(X_train)
    test_data=tv.transform(X_test)
    clf=MultinomialNB(alpha=0.01)
    clf.fit(train_data,y_train)
    print(clf.score(test_data, y_test))
def train_model_LR():
    with open(floder+'test_label.pickle', 'rb') as handle:
        data_x=pickle.load(handle)
        #print(data_x)
        print('load label...')
    with open(floder+'test_content.pickle', 'rb') as handle:
        data_y=pickle.load(handle)
        print('load content...')
    X_train, X_test, y_train, y_test = train_test_split(data_y, data_x, test_size=0.5)
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)

    lr = LogisticRegression(C=1000)
    lr.fit(train_data, y_train)
    print(lr.score(test_data, y_test))
def train_model_SVM():
    with open(floder+'test_label.pickle', 'rb') as handle:
        data_x=pickle.load(handle)
        #print(data_x)
        print('load label...')
    with open(floder+'test_content.pickle', 'rb') as handle:
        data_y=pickle.load(handle)
        print('load content...')
    X_train, X_test, y_train, y_test = train_test_split(data_y, data_x, test_size=0.5)

    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)
    clf = SVC(C=1000.0)
    clf.fit(train_data, y_train)
    print(clf.score(test_data, y_test))

if __name__ == '__main__':
    #getdata()
    #print(getlabel('科技'))

    #使用linearsvc
    start = datetime.datetime.now()
    linsvc()
    end = datetime.datetime.now()
    print('程序运行时间（训练到测试）单位：s  ')
    print((end - start).seconds)


    '''

    #使用NB：0.92754 141s
    start=datetime.datetime.now()
    train_model_NB()
    end=datetime.datetime.now()
    print((end-start).seconds)
    '''
    '''
    #使用LR:0.946616 运行时间1742s
    start = datetime.datetime.now()
    train_model_LR()
    end = datetime.datetime.now()
    print('程序运行时间（训练到测试）单位：s  ')
    print((end - start).seconds)
    '''
    '''
    
    #使用SVM
    start = datetime.datetime.now()
    train_model_SVM()
    end = datetime.datetime.now()
    print('程序运行时间（训练到测试）单位：s  ')
    print((end - start).seconds)
    '''