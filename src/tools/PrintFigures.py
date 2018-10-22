'''
提供各种图绘制
Suporting printer services
Create by Jianguo Chen
2017-09-29
'''
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches  import Circle
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


class PrintFigures:   
    
    #1 简单绘图
    # 输入参数:points:数据点集合; length:数据点长度
    def printScatter(self,points):
        plt.figure()
        for i in range(len(points)):
            plt.plot(points[i][0], points[i][1],color='#0049A4', marker = '.')
        plt.xlabel('x'), plt.ylabel('y')
        plt.show()
        
    #简单散点图
    # 输入参数:points:数据点集合; length:数据点长度
    def printPolt2(self,ylist):
        plt.figure()
        for i in range(len(ylist)):
            plt.plot(i, ylist[i], marker = '.')
        plt.xlabel('x'), plt.ylabel('y')
        plt.show()
    
    #简单折线图
    # 输入参数:points:数据点集合; length:数据点长度
    def printPolt3(self,ylist):
        plt.figure()
        plt.plot(ylist)
        plt.xlabel('x'), plt.ylabel('y')
        plt.show()
    
    
    #2 绘图散点图(带簇标签和图例)
    #  设置点的大小,坐标轴范围,坐标标签文本大小 
    def printScatter_Color(self,points,label):
        colors = self.getRandomColor()  #获得随机颜色        
        fig = plt.figure()
        ax = fig.add_subplot(111) #注意:一般都在ax中设置,不再plot中设置
        for i in range(len(points)):
            index = label[i]
            plt.plot(points[i][0], points[i][1], color = colors[index], marker = '.', MarkerSize=15)
        xmin, xmax = plt.xlim()   # return the current xlim
        ymin, ymax = plt.ylim()
        plt.xlim(xmin=int(xmin* 1.0), xmax=int(xmax *1.1))  #设置坐标轴范围
        plt.ylim(ymin = int(ymin * 1.0), ymax=int(ymax * 1.1))
        #plt.xlim(0,20)  #设置坐标轴范围
        #plt.ylim(0,12)  #设置坐标轴范围        
        
        #设置主刻度标签的位置,标签文本的格式  
       
        xmajorLocator   = MultipleLocator(4) #将x主刻度标签设置为20的倍数  
        #xmajorFormatter = FormatStrFormatter('%1.1f') '%1.1f' #设置x轴标签文本的格式  
        ax.xaxis.set_major_locator(xmajorLocator)  
        #ax.xaxis.set_major_formatter(xmajorFormatter)
        plt.xticks(fontsize = 17)#对坐标的值数值，大小限制 
        plt.yticks(fontsize = 17)
        #plt.xlabel('x', fontsize=18)
        #plt.ylabel('y', fontsize=16)
        plt.show()
        
        
    #3 绘图散点图(带簇标签和图例)
    def printScatter_Color_Marker(self,points,label):
        colors = self.getRandomColor()  #获得随机颜色
        markers = self.getRandomMarker() #获得随机形状
        fig = plt.figure()
        ax = fig.add_subplot(111) #注意:一般都在ax中设置,不再plot中设置
        cNum = np.max(label)  #获取簇的个数 
        for j in range(cNum+1):
            idx = np.where(label==j)  
            plt.scatter(points[idx,0], points[idx,1], color =  colors[j%len(colors)], label=('C'+str(j)), marker = markers[j%len(markers)], s = 20)  
        #xmin, xmax = plt.xlim()   # return the current xlim
        #ymin, ymax = plt.ylim()
        #plt.xlim(xmin=int(xmin* 1.0), xmax=int(xmax *1.1))  #设置坐标轴范围
        #plt.ylim(ymin = int(ymin * 1.0), ymax=int(ymax * 1.1))
        #plt.xlim(0,185)  #设置坐标轴范围
        #plt.ylim(0,185)  #设置坐标轴范围        
        
        #设置主刻度标签的位置,标签文本的格式  
       
        #xmajorLocator   = MultipleLocator(4) #将x主刻度标签设置为20的倍数  
        #xmajorFormatter = FormatStrFormatter('%1.1f') '%1.1f' #设置x轴标签文本的格式  
        #ax.xaxis.set_major_locator(xmajorLocator)  
        #ax.xaxis.set_major_formatter(xmajorFormatter)
        plt.xticks(fontsize = 13)#对坐标的值数值，大小限制 
        plt.yticks(fontsize = 13)
        #plt.xlabel('x', fontsize=18)
        #plt.ylabel('y', fontsize=16)
        ## best\upper right\upper left\lower left\lower right\right\center left\center right\lower center\upper center\center
        plt.legend(loc = 'lower left')
        plt.show()    
      
        
    #4 绘制带图例
    def printPoltLenged(self,points,label):
        colors = self.getRandomColor()  #获得随机颜色
        markers = self.getRandomMarker() #获得随机形状
        plt.figure()
        cNum = np.max(label) #获取簇的个数
        #print(cNum)
        for j in range(cNum+1):
            idx = np.where(label==j)  
            plt.scatter(points[idx,0], points[idx,1], color =  colors[j%len(colors)], label=('C'+str(j)), marker = markers[j%len(markers)], s = 30)  
        #plt.xlabel('x'), plt.ylabel('y')
        plt.legend(loc = 'upper left')
        #--for ED_Hexagon (-50,300)
        #plt.xlim(-50,300)  #设置坐标轴范围
        #plt.ylim(-50,300)  #设置坐标轴范围        
        plt.xticks(fontsize = 17)#对坐标的值数值，大小限制 
        plt.yticks(fontsize = 17)
        plt.show()
      
        
    #5 绘制密度峰值聚类算法的决策图
    # 横坐标为rho,纵坐标为delta
    def printRhoDelta(self,rho,delta):
        fig = plt.figure()
        ax = fig.add_subplot(111) #注意:一般都在ax中设置,不再plot中设置
        plt.plot(rho, delta, '.', MarkerSize=15)    
       
        xmin, xmax = plt.xlim()   # return the current xlim
        ymin, ymax = plt.ylim()
        #plt.xlim(xmin=int(xmin* 1.1), xmax=int(xmax *1.5))  #设置坐标轴范围
        #plt.ylim(ymin = int(ymin * 1.1), ymax=int(ymax * 1.1))
        #plt.xlim(-1,4.5)  #设置坐标轴范围
        #plt.ylim(-1,250)  #设置坐标轴范围        
        
        #设置主刻度标签的位置,标签文本的格式  
       
        #xmajorLocator   = MultipleLocator(4) #将x主刻度标签设置为20的倍数  
        #xmajorFormatter = FormatStrFormatter('%1.1f') '%1.1f' #设置x轴标签文本的格式  
        #ax.xaxis.set_major_locator(xmajorLocator)  
        #ax.xaxis.set_major_formatter(xmajorFormatter)
        plt.xticks(fontsize = 17)#对坐标的值数值，大小限制 
        plt.yticks(fontsize = 17)
        plt.xlabel('x', fontsize=17)
        plt.ylabel('y', fontsize=17)
        plt.xlabel('Adaptive domain density'), plt.ylabel('Delta distance')
        plt.show()  
    
    #绘制局部密度和域密度的对比图,上下结构
    def printTwoFig(self,rho,DD):
        plt.figure()
        plt.subplot(211)
        plt.plot(rho)
        plt.xlim(0,213)  #设置坐标轴范围
        #plt.ylim(-1,180)  #设置坐标轴范围        
        plt.xticks(fontsize = 17)#对坐标的值数值，大小限制 
        plt.yticks(fontsize = 17)
        plt.ylabel('Local density',fontsize=17)
        
        plt.subplot(212)
        plt.plot(DD)
        plt.xlim(0,213)  #设置坐标轴范围
        plt.ylim(-1,40)  #设置坐标轴范围        
        plt.xticks(fontsize = 17)#对坐标的值数值，大小限制 
        plt.yticks(fontsize = 17)
        plt.xlabel('Data points', fontsize=17) 
        plt.ylabel('Adaptive domain density',fontsize=17)
        plt.show()
 
    
    
    #6 绘制散点图带圆圈 (带簇标签和图例)
    #  每个点设置一个指定半径的圆圈 
    #  用于GenerateDataPoint
    #  输入参数: points:数据点集合, label:标签集合, rs:每个数据点所需要的圆圈半径
    def printScatterCircle(self,points,label,rs):
        colors = self.getRandomColor()  #获得随机颜色
        markers = self.getRandomMarker() #获得随机形状
        fig = plt.figure()
        ax = fig.add_subplot(111)
        print(label)
        cNum = np.max(label)  #获取簇的个数 
        print(cNum)
        for j in range(cNum+1):
            print("j=",j)
            print("range=",range(cNum))
            idx = np.where(label==j)
            print("idx:",idx)   
            plt.scatter(points[idx,0], points[idx,1], color =  colors[j%len(colors)], label=('C'+str(j)), marker = markers[j%len(markers)], s = 30)  
        #plt.xlabel('x'), plt.ylabel('y')
        for i in range(len(points)): 
            print("rs","i:",rs[i])
            cir1 = Circle(xy = (points[i,0], points[i,1]), radius=rs[i], alpha=0.03)
            ax.add_patch(cir1)
            ax.plot(points[i,0], points[i,1], 'w')
        plt.legend(loc = 'best')
        plt.show()        
        
    #生成随机颜色
    def getRandomColor(self):
        R = list(range(256))  #也可用np.arange(256)
        B = list(range(256))
        G = list(range(256))
        R = np.array(R)/255.0
        G = np.array(G)/255.0
        B = np.array(B)/255.0
        #print(R)
        random.shuffle(R)   #将序列的所有元素随机排序
        random.shuffle(G)
        random.shuffle(B)
        colors = []
        for i in range(256):
            colors.append((R[i], G[i], B[i]))        
        return colors
    
    #生成随机颜色2 指定颜色列表
    def getRandomColor2(self):
        colors =['#00B0F0','#99CC00','#7C0050']   #心形图案的三种颜色
        return colors   
 
 
    #生成随机形状
    def getRandomMarker(self):
        markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
        #markers = ['s','o', '*']   #心形图案的三种图标
        random.shuffle(markers) #随机排序
        return markers   
    
    
    