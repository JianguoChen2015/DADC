'''
DPC算法 (自己整理实现)
Created on 2017-9-27
@author: Jianguo Chen
'''
import numpy as np
from tools.PrintFigures import PrintFigures  # 绘图操作类
from tools.FileOperator import FileOperator  # 文件操作类
import math

class MyDPC:
    MAX = 1000000    
    fo = FileOperator()
    pf = PrintFigures() 
    fileurl = 'V:\Cluster_Science\dataset-DACSD\Hexagon\\'
    #Heart-shaped,Aggregation,Compound,Compound2
  
    #1 算法主程序
    def Algorithm(self): 
        
        #读取文件,生成点集合和标签集合        
        #V:\Cluster_Science\data\case-Hexagon\SD-Hexagon.csv
        #V:\Cluster_Science\data\datasets1\g2-2-50-1.csv
        #V:\Cluster_Science\data\cjg-multi-density(label)2.csv
        #V:\Cluster_Science\dataset-DACSD\Heart-shaped.csv
        #V:\Cluster_Science\data\Flame.csv
        fileName = self.fileurl + "dataset.csv" 
        (points,label) = self.fo.readDatawithLabel(fileName)
        length = len(points)
        #print(points)
        #print(label)
        #print("数据点个数:", length)        
        #print("原始图")    #绘制原始数据图 
        self.pf.printScatter_Color_Marker(points,label)
        
        #读取文件,生成点集合(没有标签)
        # CI-datasets-yeast(1-2), cunbalance
        #fileName=self.fileurl +  "dataset.csv"
        #points = self.fo.readDatawithoutLabel(fileName)
        #length = len(points)
        #print(points)
        #print("数据点个数:", length)        
        #print("原始图")
        #self.pf.printScatter(points) #绘制原始数据图               
        
        (ll,dist) = self.getDistance(points)    #1)计算每个点之间的距离（欧氏距离）        
        (rho,delta) = self.ScienceMethod(ll, dist, length)#2)计算rho和delta(原始DPC算法)           
        self.pf.printRhoDelta(rho,delta)  # 绘制决策图
        self.pf.printPolt3(rho)  #单独绘制局部密度
        
        centers = self.identifyCenters(rho, delta, length) #3)识别聚类中心
        result = self.assignDataPoint(dist, rho, centers, length) #4)计算各点所属类簇            
        
        print("结果图") #绘制聚类结果图         
        self.pf.printPoltLenged(points,result)
     
  
    #2 计算rho和delta(原始DPC算法) 
    def ScienceMethod(self, ll, dist,length):
        percent = 0.5               # 确定截断距离
        position = int(len(ll) * percent / 100)  
        print("邻居个数", position)
        sortedll = np.sort(ll)      #对ll降序排列
        dc = sortedll[position]     #获取邻居的最小距离，作为截断距离
        rho = self.getlocalDensity(dist,dc,length)  #计算局部密度
        delta = self.computDelta(rho,dist,length)   #计算Delta距离
        return rho, delta;   
       
       
    #2.1 计算各点之间的距离
    def getDistance(self,points):
        length =len(points)
        dist = np.zeros((length, length))  #定义距离矩阵(值全为0)
        ll = []
        begin = 0
        while begin < length-1:
            end = begin + 1
            while end < length:
                dd = np.linalg.norm(points[begin] - points[end])
                dist[begin][end] = dd
                dist[end][begin] = dd
                ll.append(dd)
                end = end + 1
            begin = begin + 1
        ll = np.array(ll)
        return ll,dist
    
    
    #2.2 local density计算局部密度
    def getlocalDensity(self, dist,dc, length):
        rho = np.zeros((length, 1))
        begin = 0
        while begin < length-1:
            end = begin + 1
            while end < length:
                #k = math.exp(-(dist[begin][end]/dc) ** 2)  #RBF核函数，高斯核函数
                #rho[begin] = rho[begin] + k 
                #rho[end] = rho[end] + k     
                if dist[begin][end] <= dc:
                    rho[begin] = rho[begin] + 1
                    rho[end] = rho[end] + 1
                end = end + 1
            begin = begin + 1  
        self.fo.writeData(rho, self.fileurl +  'DPC-rho.csv')    
        return rho      
    
    
    #2.3 计算Delta距离
    def computDelta(self,rho,dist, length): 
        delta = np.ones((length, 1)) * self.MAX
        maxDensity = np.max(rho)
        begin = 0
        while begin < length:
            if rho[begin] < maxDensity:
                end = 0
                while end < length:
                    if rho[end] > rho[begin] and dist[begin][end] < delta[begin]:
                        delta[begin] = dist[begin][end]
                    end = end + 1
            else:
                delta[begin] = 0.0
                end = 0
                while end < length:
                    if dist[begin][end] > delta[begin]:
                        delta[begin] = dist[begin][end]
                    end = end + 1
            begin = begin + 1
        self.fo.writeData(delta, self.fileurl +  'DPC-delta.csv')  
        return delta


    #3 识别聚类中心
    def identifyCenters(self, rho, delta, length):
        rate1 = 0.2
        #Aggregation,Spiral:0.6; Jain,Flame:0.8; D31:0.75; R15:0.6; Compound:0.5; Pathbased:0.2
        thRho = rate1 * (np.max(rho) - np.min(rho)) + np.min(rho)  #密度阀值

        rate2 = 0.1
        #Aggregation,Spiral:0.2; Jain,Flame:0.2; D31:0.05; R15:0.1; Compound:0.08; Pathbased:0.4
        thDel = rate2 * (np.max(delta) - np.min(delta)) + np.min(delta)  #距离阀值

        centers = np.ones(length, dtype=np.int) * (-1)
        cNum = 0
        #items = range(length)
        #random.shuffle(items)
        for i in range(length): #items:
            if rho[i] > thRho and delta[i] > thDel:
                centers[i] = cNum
                cNum = cNum + 1
        print("聚类中心个数:", cNum)
        #self.fo.writeData(centers, self.fileurl + 'centers.csv') 
        return centers        
 
 
    #4 计算各点所属类簇
    def assignDataPoint(self, dist,rho, result, length):
        for i in range(length):
            dist[i][i] = self.MAX

        for i in range(length):
            if result[i] == -1:
                result[i] = self.nearestNeighbor(i,dist, rho, result, length)
            else:
                continue
        return result
      
      
    #4.1 求最近邻居      
    def nearestNeighbor(self,index, dist, rho, result,length):        
        dd = self.MAX
        neighbor = -1
        for i in range(length):
            if dist[index, i] < dd and rho[index] < rho[i]:
                dd = dist[index, i]
                neighbor = i
        if result[neighbor] == -1:
            result[neighbor] = self.nearestNeighbor(neighbor, dist, rho, result, length)
        return result[neighbor]

         
#主函数
def main():    
    dpc = MyDPC()
    dpc.Algorithm()   #DPC聚类，传入点集合
    
       
if __name__ == "__main__":
    main()   