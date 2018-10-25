'''
Suporting printer services
Created on 2017-9-27
@author: Jianguo Chen
'''
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches  import Circle
from matplotlib.ticker import MultipleLocator


class PrintFigures:   
    
    #1 Draw a curve graph (only one list)
    def printPolt2(self,ylist):
        plt.figure()
        for i in range(len(ylist)):
            plt.plot(i, ylist[i], marker = '.')
        plt.xlabel('x'), plt.ylabel('y')
        plt.show()
    
    #2 Draw a curve graph (two lists)
    def printPolt3(self,ylist):
        plt.figure()
        plt.plot(ylist)
        plt.xlabel('x'), plt.ylabel('y')
        plt.show()
    
    
    #3 Draw a scatter graph
    def printScatter(self,points):
        plt.figure()
        for i in range(len(points)):
            plt.plot(points[i][0], points[i][1],color='#0049A4', marker = '.')
        plt.xlabel('x'), plt.ylabel('y')
        plt.show()
 
    #4 Draw a scatter graph (coloring with label)
    def printScatter_Color(self,points,label):
        colors = self.getRandomColor()          
        fig = plt.figure()
        ax = fig.add_subplot(111) 
        for i in range(len(points)):
            index = label[i]
            plt.plot(points[i][0], points[i][1], color = colors[index], marker = '.', MarkerSize=15)
        xmin, xmax = plt.xlim()   # return the current xlim
        ymin, ymax = plt.ylim()
        plt.xlim(xmin=int(xmin* 1.0), xmax=int(xmax *1.1))  #set the axis range
        plt.ylim(ymin = int(ymin * 1.0), ymax=int(ymax * 1.1))
               
        xmajorLocator   = MultipleLocator(4) 
        ax.xaxis.set_major_locator(xmajorLocator)  
        plt.xticks(fontsize = 17) 
        plt.yticks(fontsize = 17)
        plt.show()
        
        
    #5 Draw a scatter graph (coloring and marking with label)
    def printScatter_Color_Marker(self,points,label):
        colors = self.getRandomColor()  
        markers = self.getRandomMarker() 
        cNum = np.max(label)  #Number of clusters 
        for j in range(cNum+1):
            idx = np.where(label==j)  
            plt.scatter(points[idx,0], points[idx,1], color =  colors[j%len(colors)], label=('C'+str(j)), marker = markers[j%len(markers)], s = 20)  
        
        plt.xticks(fontsize = 13)
        plt.yticks(fontsize = 13)
        plt.legend(loc = 'lower left')
        plt.show()    
      
    #6 Draw a scatter graph with circles (coloring and marking with label)
    #  Set a circle with a specified radius for each point
    #  input: points, label, rs: the radius of the circle required for each data point
    def printScatterCircle(self,points,label,rs):
        colors = self.getRandomColor()  
        markers = self.getRandomMarker() 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        print(label)
        cNum = np.max(label)   
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
        
           
    #7 Draw a curve figure with lenged
    def printPoltLenged(self,points,label):
        colors = self.getRandomColor()  
        markers = self.getRandomMarker() 
        plt.figure()
        cNum = np.max(label) 
        for j in range(cNum+1):
            idx = np.where(label==j)  
            plt.scatter(points[idx,0], points[idx,1], color =  colors[j%len(colors)], label=('C'+str(j)), marker = markers[j%len(markers)], s = 30)  
        plt.legend(loc = 'upper left')
        plt.xticks(fontsize = 17)
        plt.yticks(fontsize = 17)
        plt.show()
      
        
    #8 Draw clustering decision graph
    # X-axis: rho, Y-axis: delta
    def printRhoDelta(self,rho,delta):
        plt.plot(rho, delta, '.', MarkerSize=15)
        plt.xticks(fontsize = 17)
        plt.yticks(fontsize = 17)
        plt.xlabel('x', fontsize=17)
        plt.ylabel('y', fontsize=17)
        plt.xlabel('Domain-adaptive density'), plt.ylabel('Delta distance')
        plt.show()  
    
    
    #9 Draw a comparison of local density and domain density (two sub-figure: top and bottom)
    def printTwoFig(self,rho,DD):
        plt.figure()
        plt.subplot(211)
        plt.plot(rho)
        plt.xlim(0,213)  #Set the axis range
        #plt.ylim(-1,180)       
        plt.xticks(fontsize = 17)
        plt.yticks(fontsize = 17)
        plt.ylabel('Local density',fontsize=17)
        
        plt.subplot(212)
        plt.plot(DD)
        plt.xlim(0,213)  
        plt.ylim(-1,40)    
        plt.xticks(fontsize = 17)
        plt.yticks(fontsize = 17)
        plt.xlabel('Data points', fontsize=17) 
        plt.ylabel('Domain-adaptive  density',fontsize=17)
        plt.show()
           
        
    #10 generate random color
    def getRandomColor(self):
        R = list(range(256))  #np.arange(256)
        B = list(range(256))
        G = list(range(256))
        R = np.array(R)/255.0
        G = np.array(G)/255.0
        B = np.array(B)/255.0
        #print(R)
        random.shuffle(R)   
        random.shuffle(G)
        random.shuffle(B)
        colors = []
        for i in range(256):
            colors.append((R[i], G[i], B[i]))        
        return colors
    
    #10 generate random color
    def getRandomColor2(self):
        colors =['#00B0F0','#99CC00','#7C0050']   
        return colors   
 
 
    #11 generate random marker
    def getRandomMarker(self):
        markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
        #markers = ['s','o', '*']   
        random.shuffle(markers) 
        return markers   