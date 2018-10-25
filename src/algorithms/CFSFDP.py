'''
The CFSFDP algorithm 
Alex Rodriguez et al. 'Clustering by fast search and find of density peaks', Science. 2014.
Created on 2017-9-27
implementation: Jianguo Chen
'''
import numpy as np
from tools.PrintFigures import PrintFigures  # 绘图操作类
from tools.FileOperator import FileOperator  # 文件操作类

class CFSFDP:
    MAX = 1000000    
    fo = FileOperator()
    pf = PrintFigures() 
    #fileurl = '../datasets/ED_Hexagon/'    
    #fileurl = '../datasets/MDDM_D31/' 
    #fileurl = '../datasets/MDDM_G2/'
    #fileurl = '../datasets/VDD_Heartshaped/'  
    #fileurl = '../datasets/Aggregation/' 
    #fileurl = '../datasets/Compound/'
    fileurl = '../datasets/G50/'
  
    #1 main function of CFSFDP
    def runAlgorithm(self): 
        #1) load input data
        fileName = self.fileurl + "dataset.csv" 
        #points, label = self.fo.readDatawithLabel(fileName)  #load input data and label
        #length = len(points)
        #self.pf.printScatter_Color_Marker(points,label)   # print original figure
        
        points = self.fo.readDatawithoutLabel(fileName)   # load input data without label
        length = len(points)
        self.pf.printScatter(points) #print original figure without label            
        
        #2) compute rho density and delta distance
        ll, dist = self.getDistance(points)    #compute distances        
        rho,delta = self.ScienceMethod(ll, dist, length) # compute rho density and delta distance           
        self.pf.printRhoDelta(rho,delta)  # print clustering decision graph 
        self.pf.printPolt3(rho)  # print graph for rho 
        
        #3) identify cluster centers
        centers = self.identifyCenters(rho, delta, length) 
        
        #4) assign the remaining points
        result = self.assignDataPoint(dist, rho, centers, length)             
        
        #print clustering results       
        self.pf.printPoltLenged(points,result)
     
  
    #2 compute rho density and delta distance
    def ScienceMethod(self, ll, dist,length):
        #1) compute the cutoff distance
        percent = 0.5               # percent of  Number of neighbors
        position = int(len(ll) * percent / 100)    # Number of neighbors
        sortedll = np.sort(ll)      #
        dc = sortedll[position]     #Get the minimum distance of the neighbor as the cutoff distance
        #2) compute rho and delta
        rho = self.getlocalDensity(dist,dc,length)  
        delta = self.computDelta(rho,dist,length)
        return rho, delta;   
       
       
    #3 compute distances among data points
    def getDistance(self,points):
        length =len(points)
        dist = np.zeros((length, length))
        ll = []
        for i in range(length-1):
            for j in range(i+1, length):
                dd = np.linalg.norm(points[i] - points[j])
                dist[i][j] = dd
                dist[j][i] = dd
                ll.append(dd)
        ll = np.array(ll)
        return ll,dist
    
    
    #4 compute rho density
    def getlocalDensity(self, dist,dc, length):
        rho = np.zeros((length, 1))
        for i in range(length-1):
            for j in range(i+1, length):
                #k = math.exp(-(dist[i][j]/dc) ** 2)  #using RBF Kernel function
                #rho[i] = rho[i] + k 
                #rho[j] = rho[j] + k     
                if dist[i][j] <= dc:
                    rho[i] = rho[i] + 1
                    rho[j] = rho[j] + 1
        self.fo.writeData(rho, self.fileurl +  'DPC-rho.csv')  #save rho density   
        return rho      
    
    
    #5 compute Delta distance
    def computDelta(self,rho,dist, length): 
        delta = np.ones((length, 1)) * self.MAX
        maxDensity = np.max(rho)
        for i in range(length):
            if rho[i] < maxDensity:
                for j in range(length):
                    if rho[j] > rho[i] and dist[i][j] < delta[i]:
                        delta[i] = dist[i][j]
            else:
                delta[i] = 0.0
                for j in range(length):
                    if dist[i][j] > delta[i]:
                        delta[i] = dist[i][j]
        self.fo.writeData(delta, self.fileurl +  'DPC-delta.csv') #save Delta distance
        return delta


    #6 identify cluster centers
    def identifyCenters(self, rho, delta, length):
        rate1 = 0.2
        thRho = rate1 * (np.max(rho) - np.min(rho)) + np.min(rho)  #set the parameter threshold of rho density

        rate2 = 0.1
        thDel = rate2 * (np.max(delta) - np.min(delta)) + np.min(delta)  #set the parameter threshold of delta distance

        centers = np.ones(length, dtype=np.int) * (-1)
        cNum = 0
        for i in range(length):  
            if rho[i] > thRho and delta[i] > thDel:
                centers[i] = cNum
                cNum = cNum + 1
        print("Number of cluster centers: ", cNum)
        #self.fo.writeData(centers, self.fileurl + 'centers.csv') 
        return centers        
 
 
    #7 assign the remaining points to the corresponding cluster center
    def assignDataPoint(self, dist,rho, result, length):
        for i in range(length):
            dist[i][i] = self.MAX

        for i in range(length):
            if result[i] == -1:
                result[i] = self.nearestNeighbor(i,dist, rho, result, length)
            else:
                continue
        return result
      
      
    #8 Get the nearest neighbor with higher rho density for each point   
    def nearestNeighbor(self, index, dist, rho, result,length):        
        dd = self.MAX
        neighbor = -1
        for i in range(length):
            if dist[index, i] < dd and rho[index] < rho[i]:
                dd = dist[index, i]
                neighbor = i
        if result[neighbor] == -1:
            result[neighbor] = self.nearestNeighbor(neighbor, dist, rho, result, length)
        return result[neighbor]

        
def main():    
    dpc = CFSFDP()
    dpc.runAlgorithm()   #run the main function of CFSFDP 
    
       
if __name__ == "__main__":
    main()   