from tqdm import tqdm
import numpy as np
import pickle


#由于产生的x和y的大小太大了所以决定采用sklearn的方式来进行计算tfidf
if __name__ == '__main__':
    list=[]
    with open('D:\\datamining\\spider\\news\\temp\\svmtrain.txt','r')as file:
    #with open('ceshi1.txt','r') as file:
        data=file.readlines()
        i=0
        s3 = np.zeros((500000,5000))
        s1 = np.zeros(500000)
        for line in tqdm(data):
            s2=np.zeros(5000)
            item=line.rstrip('\n')
            item=item.rstrip(' ')
            items=item.split(' ')
            s1[i]=np.int(items[0])

            leng=len(items)
            for j in range(1,leng):
                first=items[j].split(':')[0]
                second=items[j].split(':')[1]
                s2[np.int(first)-1]=np.float(second)
            s3[i]=s2
            i += 1

        with open('D:\\datamining\\spider\\news\\temp\\svctrainlabel.pickle','wb') as handle:
            pickle.dump(s1,handle,protocol=2)
        with open('D:\\datamining\\spider\\news\\temp\\svctraincontent.pickle','wb') as handle:
            pickle.dump(s3,handle,protocol=2)
        '''with open('ceshi1s1.pickle','rb') as handle:
            s1=pickle.load(handle)
            print(s1[0])
            print(len(s1))
        with open('ceshi1s2.pickle','rb') as handle:
            s3=pickle.load(handle)
            print(s3[0])
            print(s3[1])
            print(s3[2])'''
