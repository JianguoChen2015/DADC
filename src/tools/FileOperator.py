'''
文件存取操作类
'''
import numpy as np
 
class FileOperator:
    #read_csv        从文件、URL、文件型对象中加载带分隔符的数据。默认分隔符为,
    #read_table      从文件、URL、文件型对象中加载带分隔符的数据，默认的分隔符为制表符"\t"
    #read_fwf        读取定宽列格式数据 -- 木有分隔符的
    #read_clipboard  读取剪贴板中的数据，可以看做read_table的剪贴板。将网页转换为表格时很有用
    
    #1 读取文件生成点集合
    #  输入参数:无
    #  输出参数:points:数据点集合
    def readDatawithoutLabel(self, fileName):        
        points=[]   #点集合
        for line in open(fileName, "r"):
            items = line.strip("\n").split(",")   #每行数据格式:"x值,y值,标签值"
            tmp = []
            for item in items:
                tmp.append(float(item))
            points.append(tmp)          #提取点的x,y值
        points = np.array(points)
        return points
    
      
    #2 读取文件,生成点集合和标签集合
    #  输入参数:无
    #  输出参数:points:数据点集合, labels:标签集合
    def readDatawithLabel(self, fileName):
        points=[]   #点集合
        labels = []     #标签集合,即每个点所属的类
        for line in open(fileName, "r"):
            items = line.strip("\n").split(",")   #每行数据格式:"x值,y值,标签值"
            labels.append(int(items.pop()))        #提取标签值q
            tmp = []
            for item in items:
                tmp.append(float(item))
            points.append(tmp)          #提取点的x,y值
        points = np.array(points)
        labels = np.array(labels)
        return points,labels
    
    #2 读取单列数据形成集合
    #  输入参数:文件名
    #  输出参数:list:数据集合
    def readList1(self, fileName):
        list = []   #数据集合
        for line in open(fileName, "r"):
            items = line.strip("\n")   #每行数据格式:"[ 数据集  ]"
            list.append(float(str(items).replace('[','').replace(']','')))
        return list
    
    #3 存储数据到文件中
    #  输入参数:data:要写入的数据, fileName:文件名
    def writeData(self, data, fileName):
        f = open(fileName,'a')    #若文件不存在，系统自动创建。'a'表示可连续写入到文件，保留原内容，在原
                                #内容之后写入。可修改该模式（'w+','w','wb'等）
        for d in data:
            f.write(str(d))   #将数据写入文件中
            f.write("\n")   #换行    
        f.close()       #关闭
                