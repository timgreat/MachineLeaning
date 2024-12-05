import numpy as np
import matplotlib.pyplot as plt
def loadData(fileName):
    dataSet=[]
    labels=[]
    with open(fileName,'r')as fr:
        for line in fr.readlines():
            data=line.strip('\n').split('\t')
            dataSet.append([float(data[0]),float(data[1])])
            labels.append(float(data[-1]))
    return  dataSet,labels

def plotData(dataSet,labels):
    data_pos=[]
    data_neg=[]
    for i in range(len(labels)):
        if labels[i]>0:
            data_pos.append(dataSet[i])
        else:
            data_neg.append(dataSet[i])
    data_pos=np.array(data_pos)
    data_neg=np.array(data_neg)
    plt.scatter(np.transpose(data_pos)[0],np.transpose(data_pos)[1])
    plt.scatter(np.transpose(data_neg)[0], np.transpose(data_neg)[1])
    plt.show()

def svm():
    pass
if __name__=='__main__':
    dataSet,labels=loadData('dataset/testSet.txt')
    plotData(dataSet,labels)
    enumerate