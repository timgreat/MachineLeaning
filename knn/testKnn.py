import math

import  numpy as np

def createSet():
    group = [[1, 101], [5, 89], [108, 5], [115, 8]]
    labels = ['爱情片', '爱情片', '动作片', '动作片']

    return group,labels

def classfy(test,group,labels,k):
    distances=[]
    for i in range(len(labels)):
        distance=math.sqrt((test[0]-group[i][0])**2+(test[1]-group[i][1])**2)
        distances.append(distance)
    b=np.argsort(np.array(distances))
    count1=0
    count2=0
    for i in range(k):
        if labels[b[i]]=='爱情片':
            count1+=1
        else:
            count2+=1
    if count1 > count2:
        return '爱情片'
    else:
        return '动作片'

if __name__=='__main__':
    group,labels=createSet()

    test=[101,20]

    label=classfy(test,group,labels,3)

    print(label)
