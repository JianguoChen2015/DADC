'''
DACSD算法(自己提出的算法)
Created on 2017-9-27
@author: Jianguo Chen
'''
import numpy as np
import math
from tools.PrintFigures import PrintFigures
from tools.FileOperator import FileOperator

class MyDADC:
    MAX = 1000000 
    fo = FileOperator()
    pf = PrintFigures()
    maxdist = 0 
    fileurl = 'V:\Cluster_Science\dataset-DACSD\Hexagon\\'
    #Heart-shaped,Aggregation\Compound2
    
    
    
    #1 算法主程序
    def Algorithm(self): 
        
        #读取文件,生成点集合和标签集合        
        #cjg-multi-density(label)2.csv,
        #V:\Cluster_Science\data\case-Hexagon\SD-Hexagon.csv
        #"V:\Cluster_Science\dataset-DACSD\Heart-shaped\Heart-shaped.csv" 
        #V:\Cluster_Science\dataset-DACSD\G25\H.csv
        #fileName = self.fileurl +"dataset.csv"         
        #(points,label) = self.fo.readDatawithLabel(fileName)
        #length = len(points)
        #print(points)
        #print(label)
        #print("数据点个数:", length)
        #print("原始图")
        #self.pf.printScatter_Color(points,label) #绘制原始数据图        
        
        #读取文件,生成点集合(没有标签)
        # CI-datasets-yeast(1-2), cunbalance
        #V:\Cluster_Science\dataset-DACSD\G20\G20.csv
        #V:\Cluster_Science\data\datasets1\g2-2-50.csv
        fileName = self.fileurl +"dataset.csv"       
        points = self.fo.readDatawithoutLabel(fileName)
        length = len(points)
        #print(points)
        #print("数据点个数:", length)        
        #print("原始图")
        #self.pf.printScatter(points) #绘制原始数据图        
        
        (ll,dist) = self.getDistance(points)    #1)计算每个点之间的距离（欧氏距离） 
        self.maxdist = np.max(ll)   #最大距离
        print('最大距离:' , self.maxdist)          
        (rho,delta) = self.DACSDMethod(ll, dist, length)#2)计算rho和delta(DACSD算法)           
        self.pf.printRhoDelta(rho,delta)  # 绘制决策图
        
        centers = self.identifyCenters(rho, delta, length) #3)识别聚类中心
        result = self.assignDataPoint(dist, rho, centers, length) #4)计算各点所属类簇            
        print(points)
        print("结果图") 
        self.pf.printPoltLenged(points,result)  #绘制聚类结果图
     
      
    #2 计算rho和delta(DACSD算法)    
    def DACSDMethod(self, ll, dist, length):
        #1) KNN density计算KNN距离和密度
        percent = 5  #5.5%
        k =int(length * percent / 100) #设置邻居数
        print ("k",k)
        (kls, kDist, kDen) = self.getKNNDensity(dist, k, length)  #获取每个点的KNN邻居集合,距离和密度集合
        #self.pf.printPolt3(kDen[:,0])
        
        #2) domain density计算域密度
        #rho = self.getDomainDensity(kls,kDen)  #计算每个点的域密度差(每个点与所有其邻居点的密度差的和)
        rho = self.getDomainDensity3(kls,kDen,kDist)  #计算每个点的域密度差(每个点与所有其邻居点的密度差的和)
        wrho = self.getWeightedDomainDensity(kls,kDen,kDist)
        #rho = self.getDomainDensity2(kls, kDen, kDist)  #计算每个点的域密度差(每个点与所有其邻居点的密度差的和)
        self.pf.printPolt3(rho)
        self.pf.printPolt3(wrho)
        #rho2 =self.fo.readList1(self.fileurl +'DDD2.csv')
        #self.pf.printPolt3(rho2)
        RD = self.RelativeDomainDensity(wrho, dist,length)  #计算相对域密度
        self.pf.printPolt3(RD) 
        #RD = self.fo.readList1(self.fileurl +'RD决策图修1.csv')     
                
        #3)delta distance计算Delta距离
        delta = self.computDelta(RD,dist,length)   #计算Delta距离
        #delta = self.computDeltaDistance(rho, kls, kDist, dist)
        return RD, delta;
 
     
       
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
        self.fo.writeData(dist,self.fileurl +'dist.csv')     #存储距离 
        return ll,dist
    
    
    #2.2 每个数据点的计算KNN列表,KNN距离和KNN密度    
    def getKNNDensity(self,dist, k, length): 
        kls =np.zeros((length, k),dtype = np.integer)  #邻居集合,整数
        kDist = np.zeros((length, k)) #邻居距离集合
        kDen =  np.zeros((length, 3)) #邻居距离集合
        i =0
        while i < length:
            ll = dist[i]   #得到当前行每一列的数据，即这个数据点到其他所有点的距离
            sortedll = np.sort(ll)      #对ll降序排列
            kDist[i] = sortedll[1:k+1]     #获取k邻居的距离 (因为第0个是自身距离,故舍去)
            j = 0
            kls_temp= []
            while j < k:
                temp = np.where(ll==kDist[i][j])   #获取指定距离对应的索引（可能有多个数据的距离是相同，返回多个值，以array形式返回到tuple元组）
                temp2 = temp[0]  #获取元组的第1个值，里面是一个数组，存多个数据的索引
                j = j+len(temp2)
                #print(temp2)
                kls_temp.extend(list(temp2))
                #print(kls_temp)           
            kls[i] =kls_temp[0:k]   #提取前k个元素，因为有时会多于k个(有时距离相同的数据点有多个)    
            kDen[i][0] = 1 / np.average(kDist[i]) #求k邻居距离平均值的倒数 (密度)
            kDen[i][1] = np.average(kDist[i]) #求k邻居距离平均值
            kDen[i][2] = sortedll[k]      #存储到第k个邻居的距离  
            i = i + 1         
        #print("knn list: ",kls)
        #print("Kdist: ",kDist)
        #print("knn density: ", kDen)       
        #self.fo.writeData(kls, self.fileurl +'kls.csv')     
        #self.fo.writeData(kDist, self.fileurl +'kDist.csv') 
        self.fo.writeData(kDen, self.fileurl +'kDen.csv')    
        return kls, kDist, kDen
    
   
    
    #2.3 计算每个点的域密度差(每个点与所有其邻居点的密度差的和)
    #    输入参数: kls:每个点的邻居列表, kDen,每个点的局部密度列表(3维)
    #    输出参数:域密度差列表
    def getDomainDensity(self, kls,kDen): 
        DDD = []   #存储域密度差
        i=0
        while i < len(kls):
            Di = kDen[i][0]  #当前数据点的密度
            DDi = 0.0  
            for j in kls[i]: #当前数据点的邻居列表
                DDi =DDi + (Di- kDen[j][0]) #当前数据点与各个邻居的密度差
            DDD.append(DDi)
            i = i+1
        #self.fo.writeData(DDD, self.fileurl +'DDD.csv') 
        return DDD       
 
 
    #2.3 计算每个点的域密度差(每个点与所有其邻居点的密度差的和)
    #    输入参数: kls:每个点的邻居列表, kDen,每个点的局部密度列表(3维)
    #    输出参数:域密度差列表
    def getDomainDensity2(self, kls, kDen, kDist): 
        DDD = []   #存储域密度差
        i=0
        while i < len(kls):
            Di = kDen[i][0]  #当前数据点的密度
            DDi = 0.0  
            for j in kls[i]: #当前数据点的邻居列表
                di = sum(kDist[i])/len(kDist[i])
                DDi =DDi + (Di- kDen[j][0]) #当前数据点与各个邻居的密度差
                #k = math.exp(-(kDist[i,np.where(kls[i]==j)]/ 2 * kDist[i,-1]) ** 2)  #RBF核函数，高斯核函数
                #if DDi > 0:
                #    DDi = DDi + k
                #else:
                #    DDi = DDi + k    
            DDi = DDi * di  
            DDD.append(DDi)
            i = i+1
        #self.fo.writeData(DDD, self.fileurl +'DDD.csv') 
        return DDD     
    
       
    #2.3 计算每个点的域密度差(每个点与所有其邻居点的密度差的和)
    #    输入参数: kls:每个点的邻居列表, kDen,每个点的局部密度列表(3维)
    #    输出参数:域密度差列表
    def getDomainDensity3(self, kls,  kDen,   kDist): 
        DD = []   #存储域密度差
        i=0
        while i < len(kls):
            #di = sum(kDist[i])/len(kDist[i])
            #Di = kDen[i][0]  #当前数据点的密度
            Di = kDen[i][0]  
            for j in kls[i]: #当前数据点的邻居列表
                #方法1,直接邻居密度和
                #Di =Di + kDen[j][0] #当前数据点与各个邻居的密度和
                #方法2,加权邻居密度和
                wkDenj = kDen[j][0] * (1/kDist[i,np.where(kls[i]==j)])  #wkDenj是一个数组
                Di =Di + wkDenj[0] #当前数据点与各个邻居的加权密度和
                #方法2,BRF核函数
                #k = math.exp(-(kDist[i,np.where(kls[i]==j)]/ 2 * kDist[i,-1]) ** 2)  #RBF核函数，高斯核函数
                #Di = Di + k
            #DDi =  di /Di 
            #print(type(list(Di)), list(Di))
            DD.append(Di)
            i = i+1
        self.fo.writeData(DD, self.fileurl +'DACSD-DD.csv') 
        return DD  
    
    
    #2.3 计算每个点的域密度差(每个点与所有其邻居点的密度差的和)
    #    输入参数: kls:每个点的邻居列表, kDen,每个点的局部密度列表(3维)
    #    输出参数:域密度差列表
    def getWeightedDomainDensity(self, kls,  kDen,   kDist): 
        DD = []   #存储域密度差
        i=0
        while i < len(kls):
            #di = sum(kDist[i])/len(kDist[i])
            #Di = kDen[i][0]  #当前数据点的密度
            Di = kDen[i][0]  
            for j in kls[i]: #当前数据点的邻居列表
                #方法1,直接邻居密度和
                #Di =Di + kDen[j][0] #当前数据点与各个邻居的密度和
                #方法2,加权邻居密度和
                if(kDen[j][0] <= kDen[i][0]):
                    wkDenj = kDen[j][0] * (1/kDist[i,np.where(kls[i]==j)])  #wkDenj是一个数组
                    Di =Di + wkDenj[0] #当前数据点与各个邻居的加权密度和
                #方法2,BRF核函数
                #k = math.exp(-(kDist[i,np.where(kls[i]==j)]/ 2 * kDist[i,-1]) ** 2)  #RBF核函数，高斯核函数
                #Di = Di + k
            #DDi =  di /Di 
            #print(type(list(Di)), list(Di))
            DD.append(Di)
            i = i+1
        self.fo.writeData(DD, self.fileurl +'DACSD-WDD.csv') 
        return DD  
    
    
    #相对域密度(消除不同密度区域的域密度区别)
    def RelativeDomainDensity(self, WDD, dist, length):
        RD = np.ones((length, 1))* self.MAX
        maxDensity = np.max(WDD)
        begin = 0
        while begin < length:
            if WDD[begin] < maxDensity:
                end = 0
                while end < length:
                    if WDD[end] > WDD[begin] and dist[begin][end] < RD[begin]:
                        RD[begin] = WDD[begin] * dist[begin][end]  #
                    end = end + 1
            else:
                RD[begin] = 0.0
                end = 0
                while end < length:
                    if dist[begin][end] > RD[begin]:
                        RD[begin] =WDD[begin] *  dist[begin][end]# ( dist[begin][end]/self.maxdist)
                    end = end + 1
            begin = begin + 1
        self.fo.writeData(RD, self.fileurl +'RD.csv')
        return RD  
    
    def RelativeDomainDensity2(self,WDD,dist, length):
        RD = np.ones((length, 1))* self.MAX
        maxDensity = np.max(WDD)
        begin = 0
        while begin < length:
            if WDD[begin] < maxDensity:
                end = 0
                while end < length:
                    if WDD[end] > WDD[begin] and dist[begin][end] < RD[begin]:
                        RD[begin] = WDD[begin] * ( dist[begin][end]/self.maxdist)  #
                    end = end + 1
            else:
                RD[begin] = 0.0
                end = 0
                while end < length:
                    if dist[begin][end] > RD[begin]:
                        RD[begin] =WDD[begin] *   dist[begin][end] # ( dist[begin][end]/self.maxdist)
                    end = end + 1
            begin = begin + 1
        self.fo.writeData(RD, self.fileurl +'RD.csv')
        return RD  
    
       
    #2.6 计算Delta距离
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
        self.fo.writeData(delta, self.fileurl +'Delta.csv')
        return delta


    #2.6 计算Delta距离
    def computDeltaDistance(self, rho, kls, kDist, dist): 
        length = len(kls)  
        delta = np.ones((length, 1)) * self.MAX
        for i in range(length):
            rho_knn =[]
            for j in kls[i]:  #获取所有邻居的域密度
                rho_knn.append(rho[j])
            rho_knn2 = np.sort(rho_knn)   #升序排列 
            if rho[i] >= rho_knn2[-1]:   #如果当前点的域密度大于所有邻居的域密度,则为域密度峰值
                delta[i] =np.max(dist[i])
            else:
                delta[i] = np.min(dist[i])     
        return delta

    #3 识别聚类中心
    def identifyCenters(self, rho, delta, length):
        #rate1 = 0.6
        #Aggregation,Spiral:0.6; Jain,Flame:0.8; D31:0.75; R15:0.6; Compound:0.5; Pathbased:0.2
        #thRho = rate1 * (np.max(rho) - np.min(rho)) + np.min(rho)  #密度阀值
        thRho = np.max(rho)/2

        #rate2 = 0.2
        #Aggregation,Spiral:0.2; Jain,Flame:0.2; D31:0.05; R15:0.1; Compound:0.08; Pathbased:0.4
        #thDel = rate2 * (np.max(delta) - np.min(delta)) + np.min(delta)  #距离阀值
        thDel = np.max(delta)/4
 
        centers = np.ones(length, dtype=np.int) * (-1)
        cNum = 0
        #items = range(length)
        #random.shuffle(items)
        for i in range(length): #items:
            if rho[i] > thRho and delta[i] > thDel:
                centers[i] = cNum
                cNum = cNum + 1
        print("聚类中心个数：",cNum)
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
        MAX = 1000000
        dd = MAX
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
    dacsd = MyDADC()
    dacsd.Algorithm()   #聚类，传入点集合
    
       
if __name__ == "__main__":
    main()   