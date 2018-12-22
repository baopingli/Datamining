import os.path
rootFolder='D:\datamining\数据集\数据集\ceshicombine'
result={'folder':0}
def statistic(folder):
    for temp in os.listdir(folder):
        filepath=os.path.join(folder,temp)
        #这里还使用了递归遍历
        if(os.path.isdir(filepath)):
            result['folder']+=1
            statistic(filepath)
        else:
            (name,extension)=os.path.splitext(temp)
            if result.__contains__(extension):
                #如果存在这个文件的类型那么就在这个文件数量上+1
                result[extension]+=1
            else:
                #其他类型的文件新建然后数量为1
                result[extension]=1
if __name__=='__main__':
    statistic(rootFolder)
    sum=0
    for name in result.keys():
        if(name==''):
            print('该文件夹下共有类型为【无后缀名】的文件%s个'%(result[name]))
        else:
            print('该文件夹下共有类型【%s】的文件%s个'%(name,result[name]))
        sum+=result[name]
    print("共有目录及文件%s个"%sum)