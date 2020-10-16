import operator
import treePlotter


def read_dataset(filename):
    """
    读取文件
    :return: 数据组10x7 没有标题行
    """
    dataset = []
    with open(filename, encoding='UTF-8') as fs:
        for line in fs.readlines():
            dataset.append(line.strip().split()[1:])  # 去掉每一行第0个值
    labels = dataset[0]
    # ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    return dataset[1:], labels


def read_testset(testfile):
    """
    读取测试数据，因为有最后一列
    :param testfile:
    :return:
    """
    testset = []
    reu = []
    with open(testfile, encoding='UTF-8') as fs:
        for line in fs.readlines():
            lie = line.strip().split()
            testset.append(lie[1:-1])  # 去掉每一行第0个值和最后一个值
            reu.append(lie[-1])
    return testset[1:], reu


def splitdataset(dataset, axis, value):
    """
    划分数据集，只要指定 特征值 组成的数据集，并且顺便去掉该特征值了
    青绿  蜷缩  浊响  清晰  凹陷  硬滑    是
    乌黑  蜷缩  沉闷  清晰  凹陷  硬滑    是
    />>> 不要“清晰”特征值
    青绿  蜷缩  浊响  凹陷  硬滑    是
    乌黑  蜷缩  沉闷  凹陷  硬滑    是
    :param dataset: 给定的数据集
    :param axis:  指定列名，即已经找到的最佳特征所在的列
    :param value: 最佳特征中需要取各个特征值的样本
    :return: 二维数组
    """
    retdataset = []  # 创建返回的数据集列表
    for featVec in dataset:  # 抽取符合划分特征的值
        if featVec[axis] == value:  # 提取指定特征值所在的行，并且去掉特征值所在的列
            reducedfeatVec = featVec[:axis]  # 去掉axis特征
            reducedfeatVec.extend(featVec[axis + 1:])  # 将符合条件的特征添加到返回的数据集列表
            retdataset.append(reducedfeatVec)
    return retdataset


def CART_chooseBestFeatureToSplit(dataset):
    """
    计算特征的基尼系数
    :param dataset: 给定数据集
    :return: 最佳特征的序列号
    """
    numFeatures = len(dataset[0]) - 1  # 特征的个数
    bestGini = 999999.0
    bestFeature = -1
    for i in range(numFeatures):  # 特征数目
        featList = [example[i] for example in dataset]  # 遍历特征1
        uniqueVals = set(featList)  # 找到特征1所有的特征值
        gini = 0
        # 需要此特征下所有的特征值都参与计算！
        for value in uniqueVals:
            # 各个特征值所在的行组成的数据
            subdataset = splitdataset(dataset, i, value)  # 提取此列含有该特征值的行，i和特征所在的列一致，正好直接用
            p = len(subdataset) / float(len(dataset))  # 所有数据中此特征值个数Dv/D
            # 指定特征值数据中“坏瓜”的个数占比 p0
            subp = len(splitdataset(subdataset, -1, '否')) / float(len(subdataset))  # 继续提取刚才的数据集中值为0的数据
            gini += p * (1.0 - pow(subp, 2) - pow(1 - subp, 2))
        # print(u"CART中第%d个特征的基尼值为：%.3f" % (i, gini))
        if gini < bestGini:
            bestGini = gini  # 这里是两层循环，内层遍历计算所有特征值，得到的结果累加才是此特征的基尼值！
            bestFeature = i  # 从所有特征的基尼值中选择最小的
    return bestFeature


def majorityCnt(classList):
    """
    遍历完所有特征时返回出现次数最多的
    数据集已经处理了所有属性，但是类标签依然不是唯一的，
    此时我们需要决定如何定义该叶子节点，在这种情况下，我们通常会采用多数表决的方法决定该叶子节点的分类
    :param classList: 所有的特征组成的列表
    :return:
    """
    classCont = {}
    for vote in classList:  #
        if vote not in classCont.keys():  # 如果
            classCont[vote] = 0
        classCont[vote] += 1  # 字典中此vote的值+1，即计数
    sortedClassCont = sorted(classCont.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCont[0][0]


def CART_createTree(dataset, labels):
    """
    每一层都选择最佳的特征，然后从其各个特征值往下迭代，以字典作为上一层字典的value！
    如果传入的数据集全是相同判断结果(好/坏)，就把最后一列作为此特征值的判断结果
    :param dataset: 样本数据
    :param labels: 标题，即各个特征
    :return: 决策树，迭代的复合字典
    """
    classList = [example[-1] for example in dataset]  # 所有样本好/坏的数据集
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分，不需要区分好/坏，因为只要相同，肯定就是class[0]和所有样本都一样啊！
        return classList[0]  # 直接返回上级迭代，顺便传个最佳
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = CART_chooseBestFeatureToSplit(dataset)
    # print(u"此时最优索引为："+str(bestFeat))
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为：" + (bestFeatLabel))
    CARTTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        CARTTree[bestFeatLabel][value] = CART_createTree(splitdataset(dataset, bestFeat, value), subLabels)
    return CARTTree


def classify(inputTree, featLabels, testVec):
    """
    预测单个结果
    :param inputTree: 决策树
    :param featLabels: 所有样本名的list
    :param testVec: 测试数据如：[青绿  蜷缩  浊响  清晰  凹陷  硬滑]
    :return: 决策单个样本好坏
    """
    # 因为是树状字典结构，第一层就这一个key
    firstStr = list(inputTree.keys())[0]  # 获取key——特征名
    secondDict = inputTree[firstStr]  # 获取对应value，即第二层字典，开始找是否到尽头
    featIndex = featLabels.index(firstStr)  # 列表中此特征的序列
    classLabel = '0'
    for key in secondDict.keys():  # 二层既是上一层的value，也是个字典，所以有keys['稍糊', '模糊', '清晰']
        if testVec[featIndex] == key:  # 这个是测试数据中此特征下的值
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def classifytest(inputTree, featLabels, testDataSet):
    """
    判断每个测试数据的决策结果
    :param inputTree: 决策树
    :param featLabels: 所有特征的列表
    :param testDataSet: 测试数据集
    :return: 决策结果
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll


def assess(testReu, testJudge):
    """
    计算此时的正确率
    :param testReu: 给定测试集的最后一列，list类型
    :param testJudge: 对测试集判断结果
    :return: 正确率
    """
    correct = 0
    for i in range(len(testReu)):
        if testJudge[i] == testReu[i]:
            correct += 1
    print("正确率为：".format(correct / len(testReu)))

    # 获取迭代器长度，即查找到的特征数量
def get_length(genera):
    return sum(1 for _ in genera)

# 迭代器里存储着每个样本特征值对应的好/坏
def get_judge(genera):
    yield next(genera)
    # for i in genera:
    #     print(i)
    
    # print("好瓜个数%d" % right)
    # print("此特征值判断结果")
        

def pre_prune(dataset, labels, preTree):
    """
    每一层都选择最佳的特征，然后从其各个特征值往下迭代，以字典作为上一层字典的value！
    如果传入的数据集全是相同判断结果(好/坏)，就把最后一列作为此特征值的判断结果
    每生成一个判定标准后，都进行一次测试集的检测
    :param dataset: 样本数据
    :param labels: 标题，即各个特征
    :return: 决策树，迭代的复合字典
    """
    preTree = {}    # 记录所有特征值的个数，即使迭代到下一层去了，回溯回这一层仍会保留字典信息
    preJudge = {}   # 记录各个特征值对应的好瓜/坏瓜结果
    classList = [example[-1] for example in dataset]  # 所有样本好/坏的数据集
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分，不需要区分好/坏，因为只要相同，肯定就是class[0]和所有样本都一样啊！
        return classList[0]  # 直接返回上级迭代，顺便传个最佳
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    # 投票选举

    bestFeat = CART_chooseBestFeatureToSplit(dataset)

    # print(u"此时最优索引为："+str(bestFeat))
    bestFeatLabel = labels[bestFeat]
    print(u"此时最优索引为：" + (bestFeatLabel))

    CARTTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    # 得到列表包括节点所有的属性值
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        # 预剪枝部分
        gene = (line[-1] for line in dataset if line[bestFeat] == value)
        correspond = get_length(gene)
        print("{}的个数有{}".format(value, correspond))
        preTree[value] = correspond
        # get_judge(gene)
        
        # 迭代生成决策树部分
        subLabels = labels[:]
        CARTTree[bestFeatLabel][value] = pre_prune(splitdataset(dataset, bestFeat, value), subLabels, preTree)
    
    print("本层所有特征值的个数为：{}".format(preTree))     # 在这一步其实就找到了一层所有特征值个数
    print("本层各个特征值对应判断结果为".format(preJudge))  # 
    return CARTTree


def main():
    filename = 'watermelon2_train.txt'
    testfile = 'watermelon2_test.txt'

    dataset, labels = read_dataset(filename)
    # demo
    print(u"首次计算得到的最优特征:" + labels[CART_chooseBestFeatureToSplit(dataset)])
    # print(u"==============首次寻找最优索引结束！===============\n")
    labels_tmp = labels[:]  # 拷贝，createTree会改变labels，不能直接等于，那还是引用
    # 训练
    preTree = {}
    CARTdesicionTree = pre_prune(dataset, labels_tmp, preTree)
    print('字典格式的决策树:\n', CARTdesicionTree)

    # 绘图
    # treePlotter.CART_Tree(CARTdesicionTree)
    # 测试
    testSet, testReu = read_testset(filename)
    testJudge = classifytest(CARTdesicionTree, labels, testSet)
    print('测试集的决策结果:\n', testJudge)


if __name__ == '__main__':
    main()
