import random
import re
import numpy as np


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0 ,1 ,0 ,1 ,0 ,1]  # 类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList ,classVec

def setOfWords2Vec(vocabList, inputSet):
    vector=[0]*len(vocabList)
    for i,word in enumerate(vocabList):
        if  word in inputSet:
            vector[i]=1
    return vector
def createVocabList(dataSet):
    vacabList=set([])
    for data in dataSet:
        vacabList=vacabList | set(data)
    return vacabList
def navBayes(trainMatrix,labels):
    numSample=len(labels)
    numWord=len(trainMatrix[0])
    p_y=sum(labels)/float(numSample)
    p_x1=np.ones(numWord);p_x0=np.ones(numWord)
    numy1=2.0;numy0=2.0
    for i in range(numSample):
        if labels[i]==1:
            p_x1+=trainMatrix[i]
            numy1+=sum(trainMatrix[i])
        else:
            p_x0+=trainMatrix[i]
            numy0+=sum(trainMatrix[i])
    p_x1=np.log(p_x1/numy1)
    p_x0=np.log(p_x0/numy0)
    return p_y,p_x1,p_x0
def classify(wordVector,p_y,p_x1,p_x0):
    p1=np.log(p_y)
    p0=np.log(1-p_y)
    for loc,i in enumerate(wordVector):
        if i !=0:
            p1+=p_x1[loc]
            p0+=p_x0[loc]
    if p1>p0:
        return 1
    else:
        return 0
def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 1]
def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):
        wordList = textParse(open('dataset/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)
        wordList = textParse(open('dataset/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = navBayes(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classify(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("分类错误的测试集：",docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))
if __name__ == '__main__':
    # postingList, classVec = loadDataSet()
    # myVocabList = createVocabList(postingList)
    # trainMat = []
    # for postinDoc in postingList:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # p_y,p_x1,p_x0=navBayes(trainMat,classVec)
    # testEntry = ['love', 'my', 'dalmation']
    # vector = setOfWords2Vec(myVocabList, testEntry)
    # result=classify(vector,p_y,p_x1,p_x0)
    # print(result)
    spamTest()
