import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LogisticRegression

def loadData(fileName):
    traits=[]
    labels=[]
    with open(fileName,'r')as fr:
        for line  in fr.readlines():
            data=line.strip('\n').split('\t')
            traits.append([1.0,float(data[0]),float(data[1])])
            labels.append(int(data[2]))
    return traits,labels
def sigmoid(x):
    return 1.0/(1+np.exp(-x))
def plotData(traits,labels,weight):
    length=len(labels)
    x0=[];y0=[]
    x1=[];y1=[]
    for i in range(length):
        if labels[i]==0:
            x0.append(traits[i,1])
            y0.append(traits[i,2])
        else:
            x1.append(traits[i, 1])
            y1.append(traits[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=20, c='red', marker='s', alpha=.5)
    ax.scatter(x0, y0, s=20, c='green', alpha=.5)
    x=np.arange(-3.0,3.0,0.1)
    y=(-weight[0]-weight[1]*x)/weight[2]
    ax.plot(x,y)
    plt.title('DataSet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
def gradAscent(traits,labels):
    x=np.mat(traits)
    y=np.mat(labels).transpose()
    m,n=x.shape
    seta=np.ones((n,1))
    alpha=0.001
    maxCycle=500
    # seta_array = np.array([])
    for i in range(maxCycle):
        error=y-sigmoid(x*seta)
        seta=seta+alpha*x.transpose()*error
        # seta_array = np.append(seta_array,seta)
    # seta_array = seta_array.reshape(maxCycle, n)
    # return seta.getA(),seta_array
    return seta.getA().reshape((n,))

def SGD(traits,labels):
    m,n=traits.shape
    maxCycle=150
    seta=np.ones(n)
    # seta_array = np.array([])
    for i in range(maxCycle):
        dataIndex=list(range(m))
        for j in range(len(dataIndex)):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            error=labels[dataIndex[randIndex]]-sigmoid(sum(traits[dataIndex[randIndex]]*seta))
            seta=seta+alpha*error*traits[dataIndex[randIndex]]
            del(dataIndex[randIndex])
            # seta_array = np.append(seta_array,seta,axis=0)
    # seta_array = seta_array.reshape(maxCycle*m, n)
    # return seta,seta_array
    return seta


def plotWeights(weights_array1, weights_array2):
    # 设置汉字格式
    font = FontProperties(fname=r"simsun.ttc", size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系', fontproperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', fontproperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', fontproperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', fontproperties=font)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2', fontproperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'梯度上升算法：回归系数与迭代次数关系', fontproperties=font)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0', fontproperties=font)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1', fontproperties=font)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', fontproperties=font)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W2', fontproperties=font)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()

def classify():
    trainSet=[];trainLabels=[]
    with open('dataset/horseColicTraining.txt','r')as fr:
        for line in fr.readlines():
            data=line.strip('\n').split('\t')
            mid=[]
            for i in data[:-1]:
                mid.append(float(i))
            trainSet.append(mid)
            trainLabels.append(int(float(data[-1])))
    weight=gradAscent(np.array(trainSet),trainLabels)
    testSet=[];testLabels=[]
    with open('dataset/horseColicTest.txt','r')as fr:
        for line in fr.readlines():
            data=line.strip('\n').split('\t')
            mid=[]
            for i in data[:-1]:
                mid.append(float(i))
            testSet.append(mid)
            testLabels.append(int(float(data[-1])))
    num=0.0
    length=len(testSet)
    for i in range(length):
        h=sigmoid(sum(testSet[i]*weight))
        label=0
        if h > 0.5:
            label=1
        if label!=testLabels[i]:
            num+=1.0
    error=num/length
    print(error)
def SKclassify():
    trainSet = [];
    trainLabels = []
    with open('dataset/horseColicTraining.txt', 'r') as fr:
        for line in fr.readlines():
            data = line.strip('\n').split('\t')
            mid = []
            for i in data[:-1]:
                mid.append(float(i))
            trainSet.append(mid)
            trainLabels.append(int(float(data[-1])))
    testSet = [];
    testLabels = []
    with open('dataset/horseColicTest.txt', 'r') as fr:
        for line in fr.readlines():
            data = line.strip('\n').split('\t')
            mid = []
            for i in data[:-1]:
                mid.append(float(i))
            testSet.append(mid)
            testLabels.append(int(float(data[-1])))
    classifier = LogisticRegression(solver='sag', max_iter=5000).fit(trainSet, trainLabels)
    test_accurcy = classifier.score(testSet, testLabels) * 100
    print('正确率:%f%%' % test_accurcy)
if __name__=='__main__':
    # traits,labels=loadData('dataset/testSet.txt')
    # traits=np.array(traits)
    # weight1=SGD(traits,labels)
    # weight2=gradAscent(traits,labels)

    # plotData(traits, labels, weight1)
    # plotData(traits,labels,weight2)
    # plotWeights(weight1_array,weight2_array)
    SKclassify()
