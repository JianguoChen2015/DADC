import numpy as np

# DADC - compute inter-cluster density similarity   
def test_IDS():
    DD    =[0.98, 0.76, 0.71, 0.33, 0.45, 0.23, 0.32,  0.66, 0.65, 0.65]
    result=[0,0,0,1,1,1,1,2,2,2]
    cNum=np.max(result)+1
    Cdens = np.zeros(cNum)   #cluster density
    for i in range(cNum):
        #print(list(j for j in range(len(result)) if result[j]==i))
        Cdens[i] = np.average(list(DD[j] for j in list(j for j in range(len(result)) if result[j]==i)))
    #print(Cdens)
    
    ids = np.zeros((cNum, cNum))
    for i in range(cNum):
        for j in range(i+1, cNum):
            ids[i,j] = ids[j,i] = (np.sqrt(Cdens[i] * Cdens[j])*2)/(Cdens[i]+Cdens[j]) 
    print(ids)  
    return ids       
    
#print(np.sqrt(13.5)*2/7.5  )  
#test_IDS()   


def test_CCD():
    result = [0,0,0,1,1,1,1,2,2,2]
    kls = [[1,2],[0,2],[1,3],[2,4],[3,5],[4,6],[4,5],[5,8],[7,9],[8,0]]
    pointNum = len(result)   #Number of data points
    cNum = np.max(result)+1  #Number of clusters
    #1) crossover degree for each point
    c = np.zeros((pointNum, cNum))  
    for i in range(pointNum):
        ci = result[i] #the cluster of xi
        cn =list(result[j] for j in kls[i])   #the cluster of each neighbor
        for j in range(cNum):
            cj = cn.count(j)   #number of neighbors in j-th cluster
            ci2= cn.count(ci)  #number of neighbors in i-th cluster 
            c[i,j] = (np.sqrt(ci2 * cj)*2)/(ci2+cj) 
    print("crossover degree of each point:\n",c)
    
    #2)compute cluster crossover degree (CCD)
    ccd = np.zeros((cNum, cNum))
    for i in range(cNum):
        for j in range(i+1, cNum):
            ccd[i,j] = ccd[j,i] = (sum(list(c[k,j] for k in range(len(result)) if result[k]==i))   #c(x, i->j) points in cluster i
                                    + sum(list(c[k,i] for k in range(len(result)) if result[k]==j)))  #c(x, j->i) points in cluster j
    print(ccd)
    return ccd
    
#test_CCD()    


#14 compute cluster density stability
def test_CDS():
    result = [0,0,0,1,1,1,1,2,2,2]
    DD = [0.98, 0.76, 0.71, 0.33, 0.45, 0.23, 0.32,  0.66, 0.65, 0.65]   
    cNum = np.max(result)+1  #Number of clusters
    #1) compute the cds of each cluster
    da = np.zeros(cNum)
    for i in range(cNum):
        li = list(k for k in range(len(result)) if result[k]==i) #points in cluster ci
        den_avg = np.average(list(DD[k] for k in li))  #the average domain density of cluster ci
        #print(np.square(len(li)))
        #print(np.sum(list(np.square(DD[k] - den_avg) for k in li)))
        #print(np.square(len(li))/np.sum(list(np.square(DD[k] - den_avg) for k in li)))
        #print(np.sqrt(np.square(len(li))/np.sum(list(np.square(DD[k] - den_avg) for k in li)))) 
        da[i] = np.sqrt(np.square(len(li))/np.sum(list(np.square(DD[k] - den_avg) for k in li)))
                
    #2) compute the cds among clusters
    cds = np.zeros((cNum, cNum))
    for i in range(cNum):
        for j in range(i+1, cNum):
            li = list(k for k in range(len(result)) if (result[k]==i or result[k]==j)) #points in cluster ci
            denavg_ab = np.average(list(DD[k] for k in li))  #the average domain density of cluster ci 
            den_ab =  np.sqrt(np.square(len(li))/np.sum(list(np.square(DD[k] - denavg_ab) for k in li)))
            cds[i,j] = cds[j,i] = (da[i]+da[j])/(2* den_ab)
    return cds

test_CDS()

def test_CFD(ids, ccd, cds):
    cfd = np.zeros(np.array(ids).shape)
    cfd = (np.sqrt(3)/4)*(ids * ccd + ccd * cds + cds * ids)
    
    ids_max =np.max(ids)
    ccd_max =np.max(ccd)
    cds_max =np.max(cds)
    cfd_max = (np.sqrt(3)/4)*(ids_max * ccd_max + ccd_max * cds_max + cds_max * ids_max)
    print("cfd:\n", cfd)
    print("cfd_max:\n", cfd_max)
    cfd2=cfd/cfd_max
    print("cfd2:", cfd2)
    return cfd2
    
ids = test_IDS()
ccd = test_CCD()
cds = test_CDS()
cfd = test_CFD(ids, ccd, cds)
#print("ids:\n", ids)
#print("ccd:\n", ccd)
#print("cds:\n", cds)
#print("cdf:\n", cdf)


def test_cluset_ensemble(cfd, result):
    cNum = np.max(result)+1  #Number of clusters
    threshod_cfd = 0.6 
    for i in range(cNum):
        for j in range(i+1, cNum):
            if cfd[i,j] > threshod_cfd:
                #ensemble clusters ci and cj                    
                ks = list(k for k in range(len(result)) if result[k]==j)
                for k in ks:
                    result[k] =i
                ks2 = list(k for k in range(len(result)) if result[k]>j)
                for k in ks2:
                    result[k]= result[k]-1
    print(result)
    return result    
        
result = [0,0,0,1,1,1,1,2,2,2]        
test_cluset_ensemble(cfd, result)


#ids2= [[0, 0.91, 0.991],[0.91, 0, 0.94], [0.99, 0.94, 0]]
#ids_max =np.max(ids2)
#print(ids_max) 
