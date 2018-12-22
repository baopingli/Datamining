import warnings

import csv
import svmutil
import time
from collections import Counter
from os.path import exists, isfile
from random import randint

import jieba
import jieba.analyse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from pylab import *
from sklearn.externals import joblib

import data

import pretreatment
from sklearn.svm import LinearSVC

#显示混淆矩阵的图片
def show_confusion_matrix(conf_arr):
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')
    width = len(conf_arr)
    height = len(conf_arr[0])
    cb = fig.colorbar(res)
    alphabet = list(pretreatment.categories.inv.keys());
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    locs, labels = plt.xticks(range(width), alphabet[:width])
    for t in labels:
        t.set_rotation(90)
        # plt.xticks('orientation', 'vertical')
    # locs, labels = xticks([1,2,3,4], ['Frogs', 'Hogs', 'Bogs', 'Slogs'])
    # setp(alphabet, 'rotation', 'vertical')
    plt.yticks(range(height), alphabet[:height])
    plt.savefig('confusion_matrix.png', format='png')
#计算混淆矩阵
def analysis(y_true, y_pred):
    target_names = [pretreatment.categories.inv[i] for i in range(1,11)]
    print(classification_report(y_true, y_pred, target_names=target_names))


    print('计算混淆矩阵...')
    confusion_matrix = pd.DataFrame([[0 for i in range(10)] for j in range(10)], index=list(range(1, 11)),
                                    columns=list(range(1, 11)), dtype='int')
    for t, p in zip(y_true, y_pred):
        confusion_matrix[p][t] += 1


    confusion_matrix.to_csv('confusion_matrix.csv')
    show_confusion_matrix(confusion_matrix.values.tolist())
    confusion_matrix = confusion_matrix.applymap(lambda x:x/500)
    print(confusion_matrix)


class NaiveBayes:
    def __init__(self, c:int):
        print('朴素贝叶斯分类...')
        print('类别:', ', '.join(pretreatment.categories.keys()))
        self.sizeofdata = c
    def load_train_data(self):
        """
        从载入数据
        :param path:
        :return:
        """
        print('Load train data...')
        self.df_count, self.bag, self.count, self.idf = pretreatment.pre_treat(count=self.sizeofdata, sizeOfBOW=4000)
        self.df_count, _ = pretreatment.count_words_in_label(self.sizeofdata)
        self.set_bag = set(self.bag)
        self.df_count = self.df_count[self.bag]
        print('size of bag:',len(self.bag))
        print('Bag get.')

    def load_data_from_file1(self):
        print('Load BOW from file.')
        self.bag = []
        with open('bag4000.txt', encoding='utf8') as f:
            for line in f:
                self.bag.append(line.strip())
        print('Get bag. Size:',len(self.bag))
        self.df_count, self.count = pretreatment.count_words_in_label(self.sizeofdata)
        self.bag = list(set(self.bag) & set(self.df_count.columns))
        self.set_bag = set(self.bag)
        self.df_count = self.df_count[list(self.set_bag)]

    def load_data_from_file2(self):
        print('加载词袋...')
        self.bag = []
        with open('bag4000.txt', encoding='utf8') as f:
            for line in f:
                self.bag.append(line.strip())
        print('词袋大小:',len(self.bag))
        self.df_count, self.count = pretreatment.count_words_in_label(self.sizeofdata)
        self.bag = list(set(self.bag) & set(self.df_count.columns))
        self.set_bag = set(self.bag)
        #降维最后得到特征词对应的类中词出现的数量
        self.df_count = self.df_count[list(self.set_bag)]


    def train(self):
        print('开始训练...')
        #self.P_label = [np.log(c/sum(self.count)) for c in self.count]
        #新建了一个10x4000的表
        self.P_word_label = pd.DataFrame(index=self.df_count.index, columns=self.bag)
        for label in self.df_count.index:
            # label当中含有词袋中的词的数量的总和
            words_count = self.df_count[self.bag].ix[label].sum()
            for word in self.bag:
                #计算每一类中特征词的条件概率，使用拉普拉斯修正
                self.P_word_label[word][label] = (self.df_count[word][label] + 1) / (words_count + len(self.bag))
        # 将贝叶斯概率相乘转换成对数相加防止浮点溢出
        self.P_word_label = self.P_word_label.applymap(lambda x: np.log(x))
        #print(self.P_word_label)
        print('训练完成...')

    def test(self):
        print('开始测试...')
        test_data = pretreatment.loadData(train=False, count=self.sizeofdata)
        y_true = []
        y_pred = []
        for label, words in test_data:
            words = set(words)
            #计算文本中出现的词的概率之和然后取十类中的最大值
            pre = self.P_word_label[list(words&self.set_bag)].sum(axis=1).argmax()
            y_true.append(label)
            y_pred.append(pre)
        analysis(y_true, y_pred)

    def show_test_result(self, statistic):
        total = 0
        correct = 0
        precisions = []
        recalls = []
        f1_scores = []

        pattern = '{:^15}{:^15.2%}{:^15.2%}{:^15.2%}'
        print('{:^15}{:^15}{:^15}{:^15}'.format('lable', 'percision', 'recall', 'f1-score'))
        for label in statistic:
            precision = statistic[label][label] / statistic.ix[label].sum()
            recall = statistic[label][label] / statistic[label].sum()
            f1_score = 2 * precision * recall / (precision + recall)

            print(pattern.format(label, precision, recall, f1_score))

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

            total += statistic.ix[label].sum()
            correct += statistic[label][label]
        print()
        print(pattern.format('合计', correct / total, sum(recalls) / len(recalls), sum(f1_scores) / len(f1_scores)))
        print()
        print('Confusion Matrix:')

        print(statistic/self.sizeofdata*100)
        self.show_confusion_matrix(statistic.values.tolist())




class SVMClassify():
    def __init__(self, c):
        self.save = True
        self.count = c

    def load_statistic_info(self):
        print('Convert train data to libsvm data...')
        #这里也可以直接loadidf和读取词袋，后面再说
        df_count, self.bag, nums, self.idf = pretreatment.svm_pre_treat(count=self.count, sizeOfBOW=5000)
        self.dictbag = {self.bag[i]:i for i in range(len(self.bag))}
        self.set_bag = set(self.bag)
    def load_train_data(self):
        self.main_data = []
        self.y, self.x = svmutil.svm_read_problem('./news/temp/svmtrain.txt')
        '''
        with open('./news/temp/svctrain.txt','r') as file:
            data=file.readlines()

            for item in tqdm(data):
                item=item.rstrip('\n')
                item=item.rstrip(' ')
                items=item.split(' ')
                f_items=[]
                for son_item in items:
                    f_items.append(float(son_item))
                self.main_data.append(f_items)
        self.main_data=np.array(self.main_data)

        print("getY...")
        self.y=self.main_data[:,0]
        print('getX...')
        self.x=self.main_data[:,1:] '''

    def save_data_svmformat(self):
        '''
        将数据转换svm需要的形式
        :return:
        '''
        #计算词袋等信息
        self.load_statistic_info()
        data = pretreatment.extract_words_with_tfidf(self.idf, count=self.count, train=True)
        with open('./news/temp/svmtrain.txt', 'w', encoding='ascii') as file:
            #label 是标签 words是产生的所有的tfidf
            for label, words in tqdm(data):
                sumofvalues = sum(words[w] for w in words)
                words = {self.dictbag[word]:words[word]/sumofvalues for word in words if word in self.dictbag}
                row = str(label) + ' ' + ' '.join('{}:{}'.format(word,words[word]) for word in words) + '\n'
                file.write(row)

        data = pretreatment.extract_words_with_tfidf(self.idf, count=self.count, train=False)
        with open('./news/temp/svmtest.txt', 'w', encoding='ascii') as file:
            for label, words in tqdm(data):
                words = {self.dictbag[word]:words[word] for word in words if word in self.dictbag}
                row = str(label) + ' ' + ' '.join('{}:{}'.format(word,words[word]) for word in words) + '\n'
                file.write(row)


    def load_data_list(self, train = True):
        data = pretreatment.extract_words_with_tfidf(self.idf, count=self.count, train=train)
        y = []
        x = []
        for label, words in tqdm(data):
            if train:
                sumofvalues = sum(words[w] for w in words)
                words = {self.dictbag[w]:words[w]/sumofvalues for w in words if w in self.dictbag}
            else:
                words = {self.dictbag[w]:words[w] for w in words if w in self.dictbag}
            y.append(label)
            x.append(words)
        return y, x
    def train_list(self):
        print('Start to train.')
        paras = '-c 4 -t 0 -h 0 -m 1024'
        self.y, self.x = self.load_data_list(train=True)
        self.model = svmutil.svm_train(self.y, self.x, paras)
        svmutil.svm_save_model('./news/svmmodel',self.model)
        print('Train finished.')

    def test_list(self):
        print('Start to test.')
        self.yt, self.xt = self.load_data_list(train=False)
        p_label, p_acc, p_val = svmutil.svm_predict(self.yt, self.xt, self.model)
        print('Test finished.')
        confusion_matrix = pd.DataFrame([[0 for i in range(10)] for j in range(10)], index=list(range(1,11)), columns=list(range(1,11)), dtype='int')
        print('Calculate confusion matrix.')
        for i in tqdm(range(len(p_label))):
            confusion_matrix[p_label[i]][self.yt[i]] += 1
        confusion_matrix.to_csv('confusion_matrix_svm20000.csv')
        self.show_test_result(confusion_matrix)

    def train(self):
        print('Start to train.')
        paras = '-c 4 -t 0 -h 0'
        self.model = svmutil.svm_train(self.y, self.x, paras)
        svmutil.svm_save_model('./news/svmmodel',self.model)
        '''clf = LinearSVC(random_state=0, tol=1e-5)
        clf.fit(self.x,self.y)
        joblib.dump(clf,"train_model.m")'''


    def save_model(self, filename):
        svmutil.svm_save_model(filename, self.model)

    def load_model(self, filename):
        print('load model.')
        self.model = svmutil.svm_load_model(filename)
        print('load model successfully.')

    def test(self):
        self.model = svmutil.svm_load_model('./news/svmmodel')
        self.yt, self.xt = svmutil.svm_read_problem('./news/temp/svmtest.txt')
        print('Start to predict...')
        p_label, p_acc, p_val = svmutil.svm_predict(self.yt, self.xt, self.model)
        self.yt, self.xt = svmutil.svm_read_problem('./news/temp/svmtest.txt')
        confusion_matrix = pd.DataFrame([[0 for i in range(10)] for j in range(10)], index=list(range(1,11)), columns=list(range(1,11)), dtype='int')
        for i in range(len(p_label)):
            confusion_matrix[p_label[i]][self.yt[i]] += 1
        confusion_matrix.to_csv('confusion_matrix_svm.csv')
        self.show_test_result(confusion_matrix)

    def show_test_result(self, statistic):
        total = 0
        correct = 0
        precisions = []
        recalls = []
        f1_scores = []

        pattern = '{:^15}{:^15.2%}{:^15.2%}{:^15.2%}'
        print('{:^15}{:^15}{:^15}{:^15}'.format('lable', 'accuracy', 'recall', 'f1-score'))
        for label in statistic:
            accuracy = statistic[label][label] / statistic.ix[label].sum()
            recall = statistic[label][label] / statistic[label].sum()
            f1_score = 2 * accuracy * recall / (accuracy + recall)

            print(pattern.format(label, accuracy, recall, f1_score))

            precisions.append(accuracy)
            recalls.append(recall)
            f1_scores.append(f1_score)

            total += statistic.ix[label].sum()
            correct += statistic[label][label]
        print()
        print(pattern.format('合计', correct / total, sum(recalls) / len(recalls), sum(f1_scores) / len(f1_scores)))
        print()
        print('Confusion Matrix:')
        print(statistic/self.count*100)



def testBayes():
    bayes = NaiveBayes(50000)
    bayes.load_train_data()
    bayes.train()
    bayes.test()

def testBayes1():
    bayes = NaiveBayes(50000)
    bayes.load_data_from_file1()
    bayes.train()
    bayes.test()

def testBayes2():
    bayes = NaiveBayes(50000)
    bayes.load_data_from_file2()
    bayes.train()
    bayes.test()

'''
def testSVM():

    mysvm = SVMClassify(50000)
    mysvm.load_train_data()
    mysvm.train()


def testSVM_list():
    mysvm = SVMClassify(20000)
    mysvm.load_statistic_info()
    mysvm.load_model('./news/svmmodel')
    mysvm.test_list()

def convert_to_svmformat():
    mysvm = SVMClassify(50000)
    mysvm.save_data_svmformat()
'''
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    #朴素贝叶斯
    start = datetime.datetime.now()
    testBayes2()
    end = datetime.datetime.now()
    print('程序运行时间（训练到测试）单位：s  ')
    print((end - start).seconds)

    #svm
    #convert_to_svmformat()

    #训练svm模型
    #mysvm = SVMClassify(50000)
    #mysvm.load_train_data()
    #testSVM()

    #SVMClassify(50000).test()

    # con = pd.read_csv('confusion_matrix.csv', index_col=0, dtype='int')
    # # print(con)
    # # cl = NaiveBayes(1)
    # show_confusion_matrix(con.values.tolist())

