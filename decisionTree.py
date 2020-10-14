"""
    name:汤子健
    date:2020/10/06
    基于基尼指数划分的决策树
"""

from math import log2
import numpy as np


class Train:
    # './watermelon2_train.txt'
    def __init__(self, train_file, test_file):
        self.original_sample = self.read_data(train_file)    # 得到一个包含标题行的np数组
        self.test_file = test_file
        self.title = self.original_sample[0].tolist()

    count = 0            # 迭代计算计数
    decision_node = []   # 每次决策选择的特征值
    seze_feature = ["青绿", "乌黑", "浅白"]
    gendi_feature = ["蜷缩", "稍蜷", "硬挺"]
    qiaosheng_feature = ["浊响", "沉闷", "清脆"]
    wenli_feature = ["清晰", "稍糊", "模糊"]
    qibu_feature = ["凹陷", "稍凹", "平坦"]
    chugan_feature = ["硬滑", "软粘"]

    @staticmethod
    def read_data(txtname):
        """
        读取文件
        :return: 数据组17x8 有标题行
        """
        data = []
        with open(txtname, encoding='UTF-8') as fs:
            for line in fs.readlines():
                data.append(line.strip().split()[1:])   # 去掉每一行第0个值
        return np.array(data)

    # 读取的文件转义为各个单独的17x2列表
    @staticmethod
    def txt_interpretation(data):
        seze = []
        gendi = []
        qiaosheng = []
        wenli = []
        qibu = []
        chugan = []
        for item in data:
            seze.append([item[0], item[6]])
            gendi.append([item[1], item[6]])
            qiaosheng.append([item[2], item[6]])
            wenli.append([item[3], item[6]])
            qibu.append([item[4], item[6]])
            chugan.append([item[5], item[6]])
        return np.array(seze), np.array(gendi), np.array(qiaosheng),np.array(wenli),np.array(qibu),np.array(chugan)

    # 返回标题行，没创建类，就直接用函数返回了
    @staticmethod
    def get_title(sample):
        return sample[0].tolist()

    # 获取迭代器长度，即查找到的特征数量
    @staticmethod
    def get_length(genera):
        return sum(1 for _ in genera)

    def cal_entropy(self, D, n_true):
        """
        计算信息熵ent = 求和（pk*logpk）只有正反两种情况
        :param D: 总数
        :param n_true:  正例的个数
        :return: ent信息熵
        """
        pk = n_true/D
        ent = pk*log2(pk)*(-1)+(1-pk)*log2(1-pk)*(-1)
        return ent

    def gini_index(self, feature, feature_str):
        """
        计算基尼系数
        :param feature:  目标特征组成的17x2矩阵
        :param feature_str: 特征值字符串
        :return: 基尼系数值GI
        """
        # 样本总数
        n_all = feature.shape[0]
        # 计青绿和非青绿的个数
        n_feature = str(feature[:, 0]).count(feature_str)
        # print("样本总数{},色泽特征为{}的个数{}".format(n_all, feature_str, n_feature))
        n_nofea = n_all - n_feature

        # 计算青绿特征中好瓜的个数
        seze1_true_temp = (x for x in feature if x[0] == feature_str and x[1] == "是")    # 创建迭代器查找
        # for i in n:
        #     print(i)  # 这样才能显示迭代器的内容
        n_featrue_good = self.get_length(seze1_true_temp)

        # 计算非此特征下好瓜的个数
        wrong_feature = (y for y in feature if y[0] != feature_str and y[1] == "是")
        n_nofea_good = self.get_length(wrong_feature)
        # 计算基尼系数
        if n_feature != 0 and n_nofea != 0:

            # 青绿特征的好坏瓜比例
            p1 = (n_featrue_good/n_feature)**2
            p0 = ((n_feature - n_featrue_good)/n_feature)**2
            # 非青绿特征的好坏瓜比例
            p1_no = (n_nofea_good/n_nofea)**2
            p0_no = ((n_nofea - n_nofea_good)/n_nofea)**2
            GI = n_feature/n_all*(1-p1-p0)+n_nofea/n_all*(1-p1_no-p0_no)
            return GI
        else:
            # 包括此特征值不存在,个数为0，或特征值全都是这个，反特征=0
            return 0.99

    def adjust_train(self, sample, gini_feature):
        """
        根据传入的特征值调整样本，筛选出样本特征值和基尼最小特征值相同的数据行，
        使得到的样本在基尼值那一列纯度为100%
        但是如果调整后的样本全是好瓜/坏瓜，就不要往下了，这就是叶节点！
        :param sample: 整体样本，只保留数据部分，不要标题
        :param gini_feature: 上一步计算得到的最小基尼值的特征值
        :return: 调整筛选后的样本
        """
        adjust_sample = []
        gini_order = self.title.index(gini_feature[:2])     # 找到纹理的序列号
        for line in sample:
            if line[gini_order] == gini_feature[-2:]:
                adjust_sample.append(line)
            else:
                continue
        
        goods = (i for i in adjust_sample if i[-1] == "是")
        good_nums = self.get_length(goods)
        if good_nums == len(adjust_sample):
            return 0
        return np.array(adjust_sample)

    # 对训练集数据统一计算基尼系数保存到字典中，并找到最佳特征值
    def train_statistics(self, train_sample):
        # temp_sample = self.sample   # './watermelon2_train.txt'

        seze, gendi, qiaosheng, wenli, qibu, chugan = self.txt_interpretation(train_sample)
        # <class 'numpy.ndarray'>
        gi = {}
        for item0 in self.seze_feature:
            gi.update({"色泽" + item0: self.gini_index(seze, item0)})
        for item1 in self.gendi_feature:
            gi.update({"根蒂"+item1: self.gini_index(gendi, item1)})
        for item2 in self.qiaosheng_feature:
            gi.update({"敲声"+item2: self.gini_index(qiaosheng, item2)})
        for item3 in self.wenli_feature:
            gi.update({"纹理"+item3: self.gini_index(wenli, item3)})  # 第一个特征值最小
        for item4 in self.qibu_feature:
            gi.update({"脐部"+item4: self.gini_index(qibu, item4)})
        for item5 in self.chugan_feature:
            gi.update({"触感"+item5: self.gini_index(chugan, item5)})

        gi_min = sorted(gi.items(), key=lambda i: i[1])  # 基尼最小——纹理清晰
        # 找到基尼值最小的特征值
        gimin_name: str = gi_min[0][0]   # 纹理清晰
        print("基尼值最小的特征值:{}，其值为：{}".format(gimin_name, gi_min[0][1]))

        # 根据特征值调整样本数据
        self.count += 1

        # 找到基尼最小特征值所在的特征其他特征值,每次只能从本循环样本中查找唯一特征值
        gini_index = self.title.index(gimin_name[:2])
        gini_values = set(train_sample[:, gini_index])
        print("此特征下各个特征值有{}".format(gini_values))
        # 每一层都遍历最佳特征的所有特征值，并在进入下一层之前判断调整数据集是否全为好瓜/坏瓜，如果是就return
        for values in gini_values:
            if gi_min[0][1] != 0:
                temp = gimin_name[:2]+values
                print("----本层特征值:"+temp)
                if self.adjust_train(train_sample, temp) == 0:
                    continue 
                adjust_sample = self.adjust_train(train_sample, temp)  # 得到调整后样本
                # 迭代
                print(adjust_sample)
                gimin_name = self.train_statistics(adjust_sample)
        return gimin_name

    def test_statistics(self):
        """
        用找到的特征值检测测试集数据
        :return:
        """
        gini_feature1: str = self.train_statistics(self.original_sample[1:])    # 最开始提供的数据去掉标题行

        print("决策次数：%d" % self.count)
        test_set = self.read_data(self.test_file)      # "watermelon2_test.txt"
        gini_order: str = self.title.index(gini_feature1[:2])     # 定位测试集判断依据的特征在文件中的序列号
        feature_name: str = gini_feature1[-2:]                 # 定位测试集判断依据的特征值
        sample = test_set[1:]                        # 只要数据
        # 样本总数
        n_all = sample.shape[0]
        true_judge = 0
        for i in range(n_all):
            if sample[i][gini_order] == feature_name and sample[i][6] == "是" or \
                    sample[i][gini_order] != feature_name and sample[i][6] == "否":
                true_judge += 1
            else:
                continue
        print("测试集中样本总数有%d个,特征值匹配要求的样本数有%d个" % (n_all, true_judge))
        print("样本正确率{}".format(true_judge/n_all))


def main():
    tra = Train("watermelon2_train.txt", "watermelon2_test.txt")
    tra.test_statistics()


if __name__ == '__main__':
    main()
