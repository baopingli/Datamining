import jieba.posseg as pseg
from tqdm import tqdm
stop_words_file='D:\\datamining\\数据集\\数据集\\stop_words_ch.txt'
#切好的文档存放的位置
cutword_file='D:\\datamining\\数据集\\数据集\\运动_clean.txt'
def readFile(addr):
    with open(addr,'rt',encoding="utf-8")as f:
        texts=f.readlines()
    return texts

def cutWord(lines):
    #分词保存长度大于等于2的词
    #去停用词
    stop_words=loadStopWords(stop_words_file)
    fo=open(cutword_file,'r+',encoding="utf-8")
    for line in tqdm(lines):
        I=[]
        words=pseg.cut(line)
        for word,flag in words:
            #这里设置词的长度
            #word是词，flag是词的性质
            if len(word)>1 and word not in stop_words:
                I.append((flag,word))
            #抽取名词,但是没有包括人名，人名是nr
        for element in I:
            if element[0]=='n':
                #判断分好的词的性质
                fo.write(str(element[1]+" "))
        fo.write("\n")
    fo.close()

def loadStopWords(filename):
    stop_words=set()
    with open(filename,'r',encoding='utf-8') as file:
        for line in file:
            stop_words.add(line.strip())
    return stop_words

if __name__ == '__main__':
    #导入需要处理的文本
    text=readFile("D:\\datamining\\数据集\\数据集\\运动.txt")
    cutWord(text)