'''
Class of file operator
Created on 2017-9-27
@author: Jianguo Chen
'''
import numpy as np
 
class FileOperator:
    
    #1 Load data points without label from csv or txt file
    def readDatawithoutLabel(self, fileName):        
        points=[]   
        for line in open(fileName, "r"):
            items = line.strip("\n").split(",")   #data format in each row:"x, y, label"
            tmp = []
            for item in items:
                tmp.append(float(item))
            points.append(tmp)                    #extract values of x and y
        points = np.array(points)
        return points
    
      
    #2 Load data points and labels from csv or txt file
    def readDatawithLabel(self, fileName):
        points=[]   
        labels = []
        for line in open(fileName, "r"):
            items = line.strip("\n").split(",")    #data format in each row:"x, y, label"
            labels.append(int(items.pop()))        #extract value of label
            tmp = []
            for item in items:
                tmp.append(float(item))
            points.append(tmp)                     #extract values of x and y
        points = np.array(points)
        labels = np.array(labels)
        return points,labels
    
    
    #3 Save data into a file
    def writeData(self, data, fileName):
        f = open(fileName,'a')    #append data to the file, create a file if it doesn't exist                                #内容之后写入。可修改该模式（'w+','w','wb'等）
        for d in data:
            f.write(str(d))   
            f.write("\n")       
        f.close()       
                