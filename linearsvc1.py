import os
import pickle

import numpy as np
from bidict import bidict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from sklearn.externals import joblib
import statistic_svm
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

    with open('label.pickle', 'wb') as handle:
        pickle.dump(x, handle)
    with open('content.pickle', 'wb') as handle:
        pickle.dump(y, handle)

def pretreatment():
    with open('label.pickle', 'rb') as handle:
        data_x=pickle.load(handle)
        #print(data_x)
        print('load label...')
    with open('content.pickle', 'rb') as handle:
        data_y=pickle.load(handle)
        print('load content...')
    X_train, X_test, y_train, y_test = train_test_split(data_y, data_x, test_size=0.5)
    tv = TfidfVectorizer()
    train_data = tv.fit_transform(X_train)
    test_data = tv.transform(X_test)
    #for k in [4000, 5000, 6000, 7000, 8000, 9000, 10000]:
    ch2 = SelectKBest(chi2, k=4000)
    ch2.fit(train_data, y_train)
    train_data1 = ch2.transform(train_data)
    test_data1 = ch2.transform(test_data)
    with open('label_train.pickle', 'wb') as handle:
        pickle.dump(train_data1, handle)
    with open('content_train.pickle', 'wb') as handle:
        pickle.dump(y_train, handle)
    with open('label_test.pickle', 'wb') as handle:
        pickle.dump(test_data1, handle)
    with open('content_test.pickle', 'wb') as handle:
        pickle.dump(y_test, handle)
def linsvc_train():
    with open('label_train.pickle', 'rb') as handle:
        X_train=pickle.load(handle)
        #print(data_x)
        print('load label...')
    with open('content_train.pickle', 'rb') as handle:
        y_train=pickle.load(handle)
        print('load content...')
    clf = LinearSVC(C=3, tol=1e-5)
    rf = clf.fit(X_train, y_train)
    joblib.dump(rf, 'rf.model')

def linsvc_test():
    with open('label_test.pickle', 'rb') as handle:
        X_test=pickle.load(handle)
        #print(data_x)
        print('load label...')
    with open('content_test.pickle', 'rb') as handle:
        y_test=pickle.load(handle)
        print('load content...')
    clf = joblib.load('rf.model')
    y_predict = clf.predict(X_test)
    # print(clf.score(test_data, y_test))
    targer_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8',
                    'class 9', 'class 10']
    targer_names=['科技','体育','军事','娱乐','文化','汽车','能源','房产','健康','金融']
    #print(k)
    print(accuracy_score(y_test, y_predict))
    print(classification_report(y_test, y_predict, target_names=targer_names))
    # print(clf.coef_)
    conmat = confusion_matrix(y_true=y_test, y_pred=y_predict)
    print(conmat)
    statistic_svm.analysis(y_test, y_predict)
    statistic_svm.show_confusion_matrix(conmat)




if __name__ == '__main__':
    #getdata()
    #pretreatment()
    #linsvc_train()
    linsvc_test()