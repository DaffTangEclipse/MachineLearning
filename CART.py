
import operator
import treePlotter
def read_dataset(filename):
    fr = open(filename, encoding="UTF-8")
    all_lines = fr.readlines()   # list形式,每行为1个str
    labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    # featname=all_lines[0].strip().split(',')  #list形式
    # featname=featname[:-1]
    labelCounts = {}
    dataset = []
    for line in all_lines[0:]:
        line = line.strip().split()   # 以逗号为分割符拆分列表
        dataset.append(line)
    return dataset, labels


def read_testset(testfile):
    fr = open(testfile,encoding="UTF-8")
    all_lines = fr.readlines()
    testset=[]
    for line in all_lines[0:]:
        line=line.strip().split()   #以逗号为分割符拆分列表
        testset.append(line)
    return testset


# 划分数据集，只要指定 特征值 组成的数据集，并且顺便去掉该特征值了
def splitdataset(dataset,axis,value):
    """
    青绿  蜷缩  浊响  清晰  凹陷  硬滑    是
    乌黑  蜷缩  沉闷  清晰  凹陷  硬滑    是
        />>> 不要“清晰”特征值
    青绿  蜷缩  浊响  凹陷  硬滑    是
    乌黑  蜷缩  沉闷  凹陷  硬滑    是
    """
    retdataset=[]#创建返回的数据集列表
    for featVec in dataset:#抽取符合划分特征的值
        if featVec[axis]==value:    # 提取指定特征值所在的行，并且去掉特征值所在的列
            reducedfeatVec=featVec[:axis] #去掉axis特征
            reducedfeatVec.extend(featVec[axis+1:])#将符合条件的特征添加到返回的数据集列表
            retdataset.append(reducedfeatVec)
    return retdataset

# 计算特征的基尼系数
def CART_chooseBestFeatureToSplit(dataset):

    numFeatures = len(dataset[0]) - 1  # 特征的个数
    bestGini = 999999.0
    bestFeature = -1
    for i in range(numFeatures):    # 特征数目
        featList = [example[i] for example in dataset]  # 遍历特征1
        uniqueVals = set(featList)      # 找到特征1所有的特征值
        gini = 0
        # 需要此特征下所有的特征值都参与计算！
        for value in uniqueVals:
            # 各个特征值所在的行组成的数据
            subdataset = splitdataset(dataset,i,value)    # 提取此列含有该特征值的行，i和特征所在的列一致，正好直接用
            p = len(subdataset)/float(len(dataset))       # 所有数据中此特征值个数Dv/D

            # 指定特征值数据中“坏瓜”的个数占比 p0
            subp = len(splitdataset(subdataset, -1, '否')) / float(len(subdataset))  # 继续提取刚才的数据集中值为0的数据
            gini += p * (1.0 - pow(subp, 2) - pow(1 - subp, 2))
        print(u"CART中第%d个特征的基尼值为：%.3f" % (i, gini))
        if gini < bestGini:
            bestGini = gini     # 这里是两层循环，内层遍历计算所有特征值，得到的结果累加才是此特征的基尼值！
            bestFeature = i     # 从所有特征的基尼值中选择最小的
    return bestFeature

# 遍历完所有特征时返回出现次数最多的，CART算法中并没有用到
def majorityCnt(classList):
    '''
    数据集已经处理了所有属性，但是类标签依然不是唯一的，
    此时我们需要决定如何定义该叶子节点，在这种情况下，我们通常会采用多数表决的方法决定该叶子节点的分类
    '''
    classCont={}
    for vote in classList:  #
        if vote not in classCont.keys():    # 如果
            classCont[vote]=0
        classCont[vote] += 1    # 字典中此vote的值+1，即计数
    sortedClassCont=sorted(classCont.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCont[0][0]

def CART_createTree(dataset,labels):
    """

    :param dataset: 样本数据
    :param labels: 标题，即各个特征
    :return:
    """
    classList=[example[-1] for example in dataset]  # 所有样本好/坏的数据集
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分，不需要区分好/坏，因为只要相同，肯定就是class[0]和所有样本都一样啊！
        print("value = " + classList[0])
        return classList[0]     # 直接返回上级迭代，顺便传个最佳
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = CART_chooseBestFeatureToSplit(dataset)
    #print(u"此时最优索引为："+str(bestFeat))
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为："+(bestFeatLabel))
    CARTTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        CARTTree[bestFeatLabel][value] = CART_createTree(splitdataset(dataset, bestFeat, value), subLabels)
    return CARTTree


def classify(inputTree, featLabels, testVec):
    """
    处理测试
    :param inputTree: 决策树
    :param featLabels: 分类标签
    :param testVec: 测试数据
    :return: 决策结果跑决策树
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = '0'
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def classifytest(inputTree, featLabels, testDataSet):
    """

    :param inputTree: 决策树
    :param featLabels: 分类标签
    :param testDataSet: 测试数据集
    :return: 决策结果
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll


def main():
    filename = 'water.txt'
    testfile = 'water_test.txt'

    dataset, labels = read_dataset(filename)
    print(u"CART算法的最优特征索引为:" + str(CART_chooseBestFeatureToSplit(dataset)))
    print(u"首次寻找最优索引结束！")
    labels_tmp = labels[:]  # 拷贝，createTree会改变labels
    CARTdesicionTree = CART_createTree(dataset, labels_tmp)
    print('CARTdesicionTree:\n', CARTdesicionTree)
    treePlotter.CART_Tree(CARTdesicionTree)
    testSet = read_testset(testfile)
    print("下面为测试数据集结果：")
    print('CART_TestSet_classifyResult:\n', classifytest(CARTdesicionTree, labels, testSet))


if __name__ == '__main__':
    main()
