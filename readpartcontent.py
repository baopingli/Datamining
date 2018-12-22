import os.path
#file='D:\\datamining\\spider\\news\\temp\\svctrain.txt'
#file='D:\\datamining\\predict_data.txt'
file='D:\\datamining\\news1\\lineartest\\文化.txt'
'''file='D:\\datamining\\数据集\\数据集\\ceshicombine\\combine.txt'''
'''
通过读取txt可知，学长的txt的内容完全是源文本，只是将文本进行了合成。
财经 113216行
房产 93303行
健康 128943行
金融_clean 105492行 是进行切词、去停用词、然后取出名词来，那么就是干净的数据了
军事 88016行
能源 109569行
社会 108224行
生活 107611行
文化 119269行 文化的数据可能存在的问题是有的行是 空的、只有一个名字、有特殊字符，所以说这是清洗数据的时候应该考虑的问题。
汽车  103720行
运动 90950行
娱乐 105504行
科技 160748行 可能换行有点多 所有的数据的总量是一样的
'''
def main():
    with open(file,'r',encoding='utf-8') as p:
        index = 0

        for i in range(1,20):
            data=p.__next__()
            print(data)


        while(1):
            try:
                data=p.__next__()
                index+=1
            except:
                print('共统计%d行'%index)
                break


if __name__=='__main__':
    main()