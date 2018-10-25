#Comparison experiments on ED dataset (Hexagon)

import numpy as np
from tools.FileOperator import FileOperator
from tools.PrintFigures import PrintFigures
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster.optics import optics
from algorithms.DPC import MyDPC
from algorithms.DADC import MyDADC

fo = FileOperator()
pf = PrintFigures()
dpc = MyDPC()
dadc = MyDADC()

#1 load dataset
def loaddata_ED():
    filepath="../datasets/ED_Hexagon.csv"
    # Input data in following format [ [0.1, 0.5], [0.3, 0.1], ... ].
    points, labels = fo.readDatawithLabel(filepath);
    return points, labels


#2.1 clustering by DBSCAN
def clustering_DBSCAN(points):
    eps=10  #eps (double): Connectivity radius between points, points may be connected if distance between them less then the radius.
    neighbors =5 #neighbors (uint): minimum number of shared neighbors that is required for establish links between points.
    
    dbscan_instance = dbscan(points, eps, neighbors);
    dbscan_instance.process();
    clusters = dbscan_instance.get_clusters();
    return clusters

#2.2 clustering by OPTICS
def clustering_OPTICS(points):
    optics_instance = optics(points, 10, 3);
    optics_instance.process();
    clusters = optics_instance.get_clusters();
    return clusters

#2.3 clustering by DPC
def clustering_DPC(points):
    (ll,dist) = dpc.getDistance(points)    #1)计算每个点之间的距离（欧氏距离）        
    (rho,delta) = dpc.ScienceMethod(ll, dist, len(points))#2)计算rho和delta(原始DPC算法)           
        
    centers = dpc.identifyCenters(rho, delta, len(points)) #3)识别聚类中心
    result = dpc.assignDataPoint(dist, rho, centers, len(points)) #4)计算各点所属类簇            
     
    print("结果图") #绘制聚类结果图         
    pf.printPoltLenged(points,result)    


#2.4 clustering by DADC
def clustering_DADC(points):
    (ll,dist) = dadc.getDistance(points)    #1)计算每个点之间的距离（欧氏距离）        
    (rho,delta) = dadc.DACSDMethod(ll, dist, len(points))#2)计算rho和delta(DACSD算法)         
        
    centers = dadc.identifyCenters(rho, delta, len(points)) #3)识别聚类中心
    result = dadc.assignDataPoint(dist, rho, centers, len(points)) #4)计算各点所属类簇            
     
    print("结果图") #绘制聚类结果图         
    pf.printPoltLenged(points,result) 
    
#2.5 clustering by DADC
def clustering_DADC2(points):    
    optics_instance = optics(points, 22, 5);
    optics_instance.process();
    clusters = optics_instance.get_clusters();
    return clusters
    

def visulaizer(clusters, points):
    # Visualize clusters:
    visualizer = cluster_visualizer();
    visualizer.append_clusters(clusters, points);
    visualizer.show();


def printresults(clusters, points):
    #print(len(clusters))
    labels_results=[-1]*len(labels)   #create labels for clustering results
    for cluster in clusters:
        i = clusters.index(cluster)
        for index in cluster:
            labels_results[index] = i
    
    #print(labels_DBSCAN)
    #print(input_data)
    labels_results = np.array(labels_results)
    pf.printPoltLenged(points,labels_results)  # print the clustering results



points, labels = loaddata_ED()  
#clusters = clustering_DBSCAN(points)
clusters = clustering_OPTICS(points)
print(len(clusters))
#clusters = clustering_DADC2(points)
printresults(clusters, points)
#clustering_DPC(points)
#clustering_DADC(points)


