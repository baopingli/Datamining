import json
import os.path
rootFloder='D:\\datamining\\数据集\\数据集\\tech_f\\tech_f'
destFloder='D:\\datamining\\数据集\\数据集\\'
datalist=[] #初始化一个列表然后存储所有的东西

def combinetxt(floder):

    for temp in os.listdir(floder):
        filepath=os.path.join(floder,temp)
        #如果前面是文件夹那么就继续往里面走
        if(os.path.isdir(filepath)):
            combinetxt(filepath)
        #如果不是文件夹的时候读取txt然后合成
        else:
            #name是名字、extension是类型扩展名
            (name,extension)=os.path.splitext(temp)
            if extension=='.txt':
                with open(filepath,'r',encoding='utf8')as p:
                    data=p.readlines()
                try:
                    print(data[4][8:])
                    datalist.append(data[4][8:]+'\n')

                except Exception as e:
                    print(e)
                    #有特殊的情况 coontent:如果还有其他的情况的话那么倒进了错误的数据
                    #print(data[0][9:])
                    #datalist.append(data[0][9:])
                    try:
                        print(data[3][8:])
                        datalist.append(data[3][8:]+'\n')
                    except Exception as f:
                        print(f)
                        datalist.append(data[0][9:]+'\n')

                continue

def writetxt(floder):
    fileposition = floder+'\\科技.txt'
    with open(fileposition,'wt',encoding='utf8') as p:
        for str in datalist:
            if str[0:4] != 'NULL':
                p.write(str)







if __name__ == '__main__':

    combinetxt(rootFloder)
    writetxt(destFloder)


