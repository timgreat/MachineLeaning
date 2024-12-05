import math
import sys
import os

import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.neighbors import KNeighborsClassifier as kNN

def readData(file):
    traits = []
    labels = []
    with open(file, 'r') as fr:
        for line in fr.readlines():
            data = line.strip('\n').split('\t')
            traits.append(data[:-1])
            if data[-1] == 'didntLike':
                labels.append(1)
            elif data[-1] == 'smallDoses':
                labels.append(2)
            else:
                labels.append(3)
    traits = np.array(traits).astype(float)
    labels = np.array(labels)
    return traits,labels

def showData(traits,labels):
    font = FontProperties(fname=r"simsun.ttc", size=14)
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False,figsize=(13, 8))
    sampleSize=len(labels)
    labelColor=[]
    for i in labels:
        if i ==1:
            labelColor.append('black')
        elif i==2:
            labelColor.append('orange')
        elif i==3:
            labelColor.append('red')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=traits[:,0], y=traits[:,1], color=labelColor, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比', fontproperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数', fontproperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占', fontproperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=traits[:, 0], y=traits[:, 2], color=labelColor, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数', fontproperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数', fontproperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数', fontproperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=traits[:, 1], y=traits[:, 2], color=labelColor, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',fontproperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比', fontproperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数', fontproperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()

def autoNorm(traits):
    max_traits=traits.max(0)
    min_traits=traits.min(0)
    range_traits=max_traits-min_traits
    norm_traits=(traits-min_traits)/range_traits
    return norm_traits,range_traits,min_traits

def classfy(test,trainSet,labels,k):
    distances=(((trainSet-test)**2).sum(axis=1))**0.5
    loc=np.argsort(distances)
    classMap={}
    for i in range(k):
        if labels[loc[i]] in classMap:
            classMap[labels[loc[i]]]+=1
        else:
            classMap.update({labels[loc[i]]:1})
    label=max(classMap, key=lambda k: classMap[k])
    return label

def knnTest(fileName):
    traits, labels = readData(fileName)
    #showData(traits, labels)
    norm_traits, range_traits, min_traits = autoNorm(traits)

    ratio=0.1
    size=labels.shape[0]
    testNum=int(size*ratio)
    errorNum=0
    k=4
    for i in range(testNum):
        label=classfy(norm_traits[i,:],norm_traits[testNum:size,:],labels[testNum:size],k)
        print("分类结果:%d\t真实类别:%d" % (label, labels[i]))
        if label != labels[i]:
            errorNum+=1
    errorRatio=errorNum/testNum
    return errorRatio


# 收集数据：可以使用爬虫进行数据的收集，也可以使用第三方提供的免费或收费的数据。一般来讲，数据放在txt文本文件中，按照一定的格式进行存储，便于解析及处理。
# 准备数据：使用Python解析、预处理数据。
# 分析数据：可以使用很多方法对数据进行分析，例如使用Matplotlib将数据可视化。
# 测试算法：计算错误率。
# 使用算法：错误率在可接受范围内，就可以运行k-近邻算法进行分类。
def txt2vector(ff):
    vector=np.zeros((1,1024))
    with open(ff,'r')as fr:
        for i in range(32):
            ss=fr.readline().strip('\n')
            for j in range(32):
                vector[0,32*i+j]=int(ss[j])
    return vector
if __name__=='__main__':
    # fileName='dataset/datingTestSet.txt'
    # # erroRatio=knnTest(fileName)
    # # print(erroRatio)
    # traits, labels = readData(fileName)
    # # showData(traits, labels)
    # norm_traits, range_traits, min_traits = autoNorm(traits)
    #
    # k=3
    # info=(np.array([44000,12,0.5])-min_traits)/range_traits
    # label=classfy(info,norm_traits,labels,k)
    # if label==1:
    #     print('dislike')
    # elif label==2:
    #     print('smalllike')
    # else:
    #     print('largelike')
    trainList=os.listdir('dataset/trainingDigits/')
    length=len(trainList)
    testList=os.listdir('dataset/testDigits')
    trainSet=np.zeros((length,1024))
    #trainLabels=np.zeros((length,),dtype=int)
    trainLabels =[]
    for i,ff in enumerate(trainList):
        vector=txt2vector('dataset/trainingDigits/'+ff)
        trainSet[i,:]=vector
        #trainLabels[i]=int(ff.strip('_')[0])
        trainLabels.append(int(ff.strip('_')[0]))
    neigh = kNN(n_neighbors=3, algorithm='auto')
    neigh.fit(trainSet, trainLabels)
    errorNum=0.0
    for ff in testList:
        vector=txt2vector('dataset/testDigits/'+ff)
        predic_label=neigh.predict(vector)
        label=int(ff.split('_')[0])
        if predic_label!=label:
            errorNum+=1
    print(errorNum/len(testList))
