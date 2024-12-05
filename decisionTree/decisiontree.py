import math
import pickle

import numpy as np




def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    return np.array(dataSet), labels
def countEntropy(vector):
    vector_map={}
    for i in vector:
        if i in vector_map:
            vector_map[i]+=1
        else:
            vector_map.update({i:1})
    length=vector.shape[0]
    entropy=0.0
    for value in  vector_map.values():
        entropy=entropy-(value/length)*(math.log2(value/length))
    return entropy

def bestTrait(data):
    M=data.shape[0]
    N=data.shape[1]
    H_D=countEntropy(data[:,N-1])

    max=0
    traitNum=0
    for i in range(N-1):
        trait=data[:, i]
        trait_map={}
        for j in range(M):
            if trait[j] in trait_map:
                trait_map[trait[j]].append(data[j,N-1])
            else:
                trait_map.update({trait[j]:[data[j,N-1]]})
        H_D_A=0.0
        for value in trait_map.values():
            H_D_A=H_D_A+(len(value)/M)*countEntropy(np.array(value))
        g_D_A=H_D-H_D_A
        print("第%d个特征的增益为%.3f" % (i, g_D_A))
        if g_D_A > max:
            traitNum=i
            max=g_D_A
    return traitNum,max

# 收集数据：可以使用任何方法。
# 准备数据：收集完的数据，我们要进行整理，将这些所有收集的信息按照一定规则整理出来，并排版，方便我们进行后续处理。
# 分析数据：可以使用任何方法，决策树构造完成之后，我们可以检查决策树图形是否符合预期。
# 训练算法：这个过程也就是构造决策树，同样也可以说是决策树学习，就是构造一个决策树的数据结构。
# 测试算法：使用经验树计算错误率。当错误率达到了可接收范围，这个决策树就可以投放使用了。
# 使用算法：此步骤可以使用适用于任何监督学习算法，而使用决策树可以更好地理解数据的内在含义。
def decisionTree(key,dataSet,sublabels):
    # for i in dataSet:
    #     if i[-1] in dataMap:
    #         dataMap[i[-1]].append(i[:-1])
    #         if len(dataMap[i[-1]]) > max:
    #             max=len(dataMap[i[-1]])
    #             pos=i[-1]
    #     else:
    #         dataMap.update({i[-1]:i[:-1]})
    dataMap=splitData(dataSet,-1)
    #print(dataMap)
    index=0
    max=0
    for k in dataMap.keys():
        length=len(dataMap[k])
        if length>max:
            max=length
            index=k

    if len(dataMap) ==1 or dataSet.shape[1]==1:
        return {key:index}

    best,g_D_A=bestTrait(dataSet)
    myTree = {sublabels[best]: {}}
    ll=[]
    sb=[]
    for n in range(len(sublabels)):
        if n!=best:
            ll.append(n)
            sb.append(sublabels[n])
    ll.append(len(sublabels))
    traitMap=splitData(dataSet,best)
    for k in traitMap.keys():
        myTree[sublabels[best]].update(decisionTree(k,np.array(traitMap[k])[:,np.array(ll)],sb))

    if key=='-1':
        return myTree
    return {key:myTree}




def splitData(dataSet,i):
    traitMap={}
    for sample in dataSet:
        if sample[i] in traitMap:
            traitMap[sample[i]].append(sample)
        else:
            traitMap.update({sample[i]:[sample]})
    return traitMap
def classfy(tree,trait,labels):
    midtree=tree
    for i,label in enumerate(labels):
        if type(midtree).__name__ == 'dict':
            midtree=midtree[label][trait[i]]
        else:
            break
    if midtree=='yes':
        print('放贷')
    else:
        print('不放贷')

def storgeTree(tree,fileName):
    with open(fileName,'wb')as fw:
        pickle.dump(tree,fw)
def loadTree(fileName):
    with open(fileName,'rb')as fr:
        return pickle.load(fr)
if __name__=='__main__':
    dataSet,labels=createDataSet()
    tree=decisionTree('-1',dataSet,labels)
    storgeTree(tree,'decisiontree.txt')
    tree=loadTree('decisiontree.txt')
    classfy(tree,['0','0'],['有自己的房子','有工作'])