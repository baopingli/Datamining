"""
用于统计liblinear测试结果
"""
import pandas as pd
from tqdm import tqdm
from bidict import bidict
from sklearn.metrics import classification_report
import pretreatment
from pylab import *
import matplotlib.pyplot as plt

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

def analysis(y_true, y_pred):
    target_names = [categories.inv[i] for i in range(1, 11)]
    print(classification_report(y_true, y_pred, target_names=target_names))

    confusion_matrix = pd.DataFrame([[0 for i in range(10)] for j in range(10)], index=list(range(1, 11)),
                                    columns=list(range(1, 11)), dtype='int')
    for t, p in tqdm(zip(y_true, y_pred)):
        confusion_matrix[p][t] += 1
    confusion_matrix.to_csv('confusion_matrix_svm50000.csv')
    show_confusion_matrix(confusion_matrix.values.tolist())
    cm = confusion_matrix.applymap(lambda x: x/500)
    print('confusion matrix:')
    print(cm)
    total = 0
    for i in range(1,11):
        total += cm[i][i]
    print()
    print('avg/total = ', total/10)

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
    plt.savefig('confusion_matrix5000.png', format='png')

if __name__ == '__main__':
    y_pred = []
    with open('./svm/output.txt') as file:
        for line in tqdm(file):
            pre = int(line.strip())
            y_pred.append(pre)
    y_true = []
    for i in range(1,11):
        y_true += [i] * 50000
    analysis(y_true, y_pred)


